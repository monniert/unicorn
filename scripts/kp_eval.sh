#!/bin/bash
set -e
CUDA_VISIBLE_DEVICES=$cuda python src/kp_eval.py --tag $tag
