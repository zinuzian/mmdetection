#! /bin/bash
export CUDA_VISIBLE_DEVICES=2
python my_test3.py \
    --model_dir='yolox' \
    --model='yolox_s' \
    --backbone='8x8' \
    --neck='300e' \
    --samples_per_gpu=32 \
    --val_ratio=0.2 \
    --seed='0, 1, 2, 3' \
    --n_gpus=1