#! /bin/bash
export CUDA_VISIBLE_DEVICES=1
python my_test2.py \
    --model_dir='cascade_rcnn' \
    --model='cascade_rcnn' \
    --backbone='r50' \
    --neck='fpn' \
    --samples_per_gpu=40 \
    --val_ratio=0.2 \
    --seed='0, 1, 2, 3' \
    --n_gpus=1