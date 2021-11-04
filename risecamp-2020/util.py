import io
import os
import json
from collections import defaultdict
from itertools import cycle
from sqlite3 import connect
from tqdm import tqdm, trange
import requests

import numpy as np
import pandas as pd
import faiss

import ray
#from ray.experimental.metrics import Gauge

ROOT_PATH = "/root/rise-camp-tutorial"
if not os.path.exists(ROOT_PATH):
    ROOT_PATH = "."
with open(os.path.join(ROOT_PATH, "all-image-ids.json")) as f:
    MOVIE_IDS = json.load(f)


def get_db_connection():
    path = os.path.join(ROOT_PATH, "tutorial.sqlite3")
    if not os.path.exists(path):
        raise Exception("""
It seems like the database file doesn't exist. Did you forget
to download it? You can try running
    make download-data
    """)

    return connect(path)


@ray.remote(num_cpus=0)
class ImpressionStore:
    def __init__(self):
        # session_key -> {id: model}
        self.impressions = defaultdict(dict)

        # session_key -> {model_name: int}
        self.session_liked_model_count = defaultdict(
            lambda: defaultdict(lambda: 0))

    def record_impressions(self, session_key, impressions):
        # Record impressions we are sending out
        for model, ids in impressions.items():
            for movie_payload in ids:
                movie_id = movie_payload["id"]
                self.impressions[session_key][movie_id] = model

    def model_distribution(self, session_key, liked_id):
        if session_key == "":
            return {}
        # Record feedback from the user
        src_model = self.impressions[session_key].get(liked_id)
        if src_model is not None:
            self.session_liked_model_count[session_key][src_model] += 1

        return self.session_liked_model_count[session_key]


def choose_ensemble_results(model_distribution, model_results):
    model_results = {
            model: results[:] for model, results in model_results.items()
            }
    # Normalize dist
    if len(model_distribution) != 2:
        default_dist = {model: 1 for model in ["color", "plot"]}
        for name, count in model_distribution.items():
            default_dist[name] += count
    else:
        default_dist = model_distribution
    total_weights = sum(default_dist.values())
    normalized_distribution = {
        k: v / total_weights
        for k, v in default_dist.items()
    }

    # Generate num returns
    chosen = []
    impressions = defaultdict(list)
    dominant_group = max(
        list(normalized_distribution.keys()),
        key=lambda k: normalized_distribution[k])
    sorted_group = list(
        sorted(
            normalized_distribution.keys(),
            key=lambda k: -normalized_distribution[k]))
    if normalized_distribution[sorted_group[0]] > normalized_distribution[sorted_group[1]]:
        sorted_group = [dominant_group] + sorted_group

    # Rank based on weights
    groups = cycle(sorted_group)
    while len(chosen) <= 10:
        model = next(groups)
        preds = model_results[model]

        if len(preds) == 0:
            if model == dominant_group:
                break
            else:
                continue

        movie_id = preds.pop(0)

        if movie_id not in chosen:
            impressions[model].append(movie_id)
            chosen.append(movie_id)

    return normalized_distribution, impressions, chosen


class LRMovieRanker:
    def __init__(self, lr_model, features):
        self.lr_model = lr_model
        self.features = features

    def rank_movies(self, recommended_movies):
        vectors = np.array([self.features[i] for i in recommended_movies])
        ranks = self.lr_model.predict_proba(vectors)[:, 1].flatten()
        high_to_low_idx = np.argsort(ranks).tolist()[::-1]
        return [recommended_movies[i] for i in high_to_low_idx]


class KNearestNeighborIndex:
    def __init__(self, db_cursor):
        # Query all the cover image palette
        self.id_to_arr = {
            row[0]: np.array(json.loads(row[1])).flatten()
            for row in db_cursor
        }

        vector_length = len(next(iter(self.id_to_arr.values())))
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(vector_length))

        # Build the index
        arr = np.stack(list(self.id_to_arr.values())).astype('float32')
        ids = np.array(list(self.id_to_arr.keys())).astype('int')
        self.index.add_with_ids(arr, ids)

    def search(self, liked_id, num_returns):
        # Perform nearest neighbor search
        source_color = self.id_to_arr[liked_id]
        source_color = np.expand_dims(source_color, 0).astype('float32')
        _, ids = self.index.search(source_color, num_returns)
        neighbors = ids.flatten().tolist()

        return [str(n) for n in neighbors]


def load_image(image_id):
    r = requests.get(f"https://rise-camp-ray-data.s3-us-west-2.amazonaws.com/movie-cover-assets/{image_id}.jpg", stream=True)
    return io.BytesIO(r.content)

def progress_bar(obj_refs):
    ready = []
    with tqdm(total=len(obj_refs)) as pbar:
        while len(obj_refs) > 0:
            new_ready, obj_refs = ray.wait(obj_refs, num_returns=min(10, len(obj_refs)))
            pbar.update(len(new_ready))
            ready.extend(new_ready)
    return ready

def delete_palettes(n):
    db = get_db_connection()
    c = db.execute("select id, palette_json from movies where palette_json != ''")
    rows = c.fetchmany(n)
    for r in rows:
        db.execute("update movies set palette_json = ('') where id == ({})".format(r[0]))
    db.commit()

