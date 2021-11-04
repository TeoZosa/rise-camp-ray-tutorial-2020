#!/bin/bash

# TODO: try this out

pip install virtualenv
virtualenv --python=python3.7 risecamp
source activate risecamp && pip install -r requirements.txt
