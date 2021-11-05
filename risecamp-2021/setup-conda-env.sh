#!/bin/bash

conda create -n risecamp python=3.7
source activate risecamp && pip install -r requirements.txt
source activate risecamp && ray install-nightly
