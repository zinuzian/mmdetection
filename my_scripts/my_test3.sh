#! /bin/bash
export CUDA_VISIBLE_DEVICES=2
python my_test1.py \
    --model_dir='faster_rcnn' \
    --model='faster_rcnn' \
    --backbone='r50' \
    --neck='fpn' \
    --samples_per_gpu=32 \
    --val_ratio=0.2 \
    --seed='0, 1, 2, 3' \
    --n_gpus=1