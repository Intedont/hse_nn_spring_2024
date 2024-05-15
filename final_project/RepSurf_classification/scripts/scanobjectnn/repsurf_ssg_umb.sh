#!/usr/bin/env bash
export PYTHONPATH='/home/madusov/nn_project/hse_nn_spring_2024/final_project/RepSurf_classification'

set -v

python3 tool/train_cls_scanobjectnn.py \
          --cuda_ops \
          --batch_size 220 \
          --model repsurf.repsurf_ssg_umb \
          --epoch 100 \
          --log_dir repsurf_cls_ssg_umb \
          --gpus 0 \
          --n_workers 12 \
          --return_center \
          --return_dist \
          --return_polar \
          --group_size 8 \
          --umb_pool sum \
          --num_point 1024 \
          --min_val 5