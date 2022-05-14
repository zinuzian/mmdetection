# 모듈 import
import argparse
import os

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor

import torch.distributed as dist
import torch.multiprocessing as mp

from my_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    #       Template      #
    # parser.add_argument('--xxx', type=str, default='xxx', help='xxx')
    # parser.add_argument('--xxx', type=int, default=0, help='xxx')
    # parser.add_argument('--xxx', action='store_true', help='xxx')
    # parser.add_argument('--xxx', type=lambda x: list(map(int, x.split(', '))))
    
    parser.add_argument('--model_dir', type=str, default='faster_rcnn', help='Object detection base model')
    parser.add_argument('--model', type=str, default='faster_rcnn', help='Object detection base model')
    parser.add_argument('--backbone', type=str, default='r50', help='Object detection backbone network')
    parser.add_argument('--neck', type=str, default='fpn', help='Object detection connection between backbone and heads')
    parser.add_argument('--dense_head', type=str, default='', help='Object detection dense head')
    parser.add_argument('--roi_head', type=str, default='', help='Object detection roi head')
    
    parser.add_argument('--samples_per_gpu', type=int, default=4, help='Object detection samples_per_gpu')
    parser.add_argument('--val_ratio', type=float, default=.2, help='Validation ratio')
    parser.add_argument('--seed', type=lambda x: list(map(int, x.split(', '))), help='Random seed')
    parser.add_argument('--n_gpus', type=int, help='# of gpus')
    
    args = parser.parse_args()
    return args
    
def get_cfg(args):
    cfg_str = "_".join(filter(None, [
        args.model,
        args.backbone,
        args.neck if args.neck else "",
        args.dense_head if args.dense_head else "",
        args.roi_head if args.roi_head else "",
    ]))
    print(cfg_str)
    return Config.fromfile(f'./configs/{args.model_dir}/{cfg_str}_coco.py'), cfg_str
    
def main(args):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg, cfg_str = get_cfg(args)

    root='../dataset/'
    input_json = "train.json"
    ratio = args.val_ratio
    
    for seed in args.seed:
        split_dataset(root, input_json, ratio, seed)
            
        # dataset config 수정
        cfg.data.train.dataset.classes = classes
        cfg.data.train.dataset.img_prefix = root
        cfg.data.train.dataset.ann_file = root + f'train_{seed}.json' # train json 정보
        cfg.data.train.pipeline[0]['img_scale'] = (512,512) # Resize
        cfg.data.train.pipeline[1]['border'] = (-512 // 2, -512 // 2) # Resize
        cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize
        
        cfg.data.val.classes = classes
        cfg.data.val.img_prefix = root
        cfg.data.val.ann_file = root + f'val_{seed}.json' # val json 정보
        cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize

        cfg.data.test.classes = classes
        cfg.data.test.img_prefix = root
        cfg.data.test.ann_file = root + f'test_{seed}.json' # test json 정보
        cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

        cfg.data.samples_per_gpu = args.samples_per_gpu

        cfg.seed = seed
        cfg.gpu_ids = list(range(args.n_gpus))
        cfg.work_dir = f'./work_dirs/{cfg_str}_trash_{seed}'

        if type(cfg.model.bbox_head) == type([1]):
            for i in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.bbox_head[i].num_classes = 10
        else:
            cfg.model.bbox_head.num_classes = 10

        cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
        cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)


        # build_dataset
        datasets = [build_dataset(cfg.data.train)]

        # 모델 build 및 pretrained network 불러오기
        model = build_detector(cfg.model)
        model.init_weights()


        # 모델 학습
        # dist.init_process_group("gloo", rank=seed+1, world_size=args.n_gpus)
        train_detector(model, datasets[0], cfg, distributed=False, validate=True)
    
    
    
if __name__ == "__main__":
    args = get_args()
    main(args)