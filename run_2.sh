#!/bin/bash

ray stop --force
ray start --head --num-cpus 16
python setup.py
