#!/bin/bash
set -e
CUDA_VISIBLE_DEVICES=$cuda python src/reconstruct.py --model $model --input $input
