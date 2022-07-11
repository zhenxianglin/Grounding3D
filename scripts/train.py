import os
import sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import random
import torch
from datasets import create_dataset
from models import create_model
from torch.utils.data import DataLoader



def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector')
    parser.add_argument('--dataset', default='sunrefer', type=str, help='sunrefer')
    parser.add_argument('--data_path', default='data', type=str, help='point cloud path')
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--sample_points_num', default=3000, type=int, help='number of sampling points')
    parser.add_argument('--img_shape', default=(256, 256), type=tuple, help='image shape')
    # parser.add_argument('--use_color', action='store_true', help="If true, use 6 dim point cloud including RGB")
    # parser.add_argument('--use_aug', action='store_true', help="If true, use augmentation when training")
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # parser.add_argument('--lr', default=1e-4, type=float)
    # parser.add_argument('--lr_bert', default=1e-5, type=float)

    parser.add_argument('--no_pretrained_resnet', action='store_true', help="If true, not use pretrained resnet")
    parser.add_argument('--resnet_layer_num', default=34, type=int, help="number of rest layers")
    parser.add_argument('--roi_size', default=(7, 7), type=tuple, help="roi align output size")
    parser.add_argument('--spatial_scale', default=1/32, type=float, help="roi align spatial_scale")
    parser.add_argument('--objects_aggregator', default='pool', type=str, help="approach of objects aggregatro", choices=['linear', 'pool'])
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--linear_layer_num', default=3, type=int, help="number of linear layers")
    parser.add_argument('--pc_out_dim', default=512, type=int, help="point cloud output feature dimension")
    parser.add_argument('--vis_out_dim', default=1024, type=int, help="visual information (point cloud and image) output feature dimension")
    
    # parser.add_argument('--no_evaluate', action='store_true', help="If true, evaluate when training")
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    return args

def main(args):
    seed = random.randint(0, 200)
    print("seed: ", 150)
    set_random_seed(150)

    print("Create dataset")
    train_dataset = create_dataset(args, 'train')
    # val_dataset = create_dataset(args, 'val')

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, 
                                    num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    # val_dataloader = DataLoader(val_dataset, 1, shuffle=False, 
    #                                 num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)

    model = create_model(args).cuda()

    for image, boxes2d, points, spatial, vis_mask, token, mask, segment_ids, target in train_dataloader:
        image = image.cuda()
        boxes2d = boxes2d.cuda()
        points = points.cuda()
        spatial = spatial.cuda()
        vis_mask = vis_mask.cuda()
        token = token.cuda()
        mask = mask.cuda()
        segment_ids = segment_ids.cuda()
        target = target.cuda()

        model(image, boxes2d, points, spatial, vis_mask, token, mask, segment_ids)
        break
    return





if __name__ == '__main__':
    args = get_args_parser()
    main(args)