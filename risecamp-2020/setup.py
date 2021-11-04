import ray
from ray import serve
import pickle

from deploy_color import ColorRecommender
from deploy_plot import PlotRecommender
from util import get_db_connection

# Connecting to background Ray cluster
ray.init(address="auto")

# Create a Serve instance
serve.start(detached=True)


client = serve.connect()
client.create_backend("color:v1", ColorRecommender)
client.create_endpoint("color", backend="color:v1", route="/rec/color")
model_weights = get_db_connection().execute(
    "SELECT weights FROM models WHERE key='ranking/lr:base'").fetchone()[0]
base_lr_model = pickle.loads(model_weights)
client.create_backend("plot:v1", PlotRecommender, base_lr_model)
client.create_endpoint("plot", backend="plot:v1", route="/rec/plot")
