# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
import os
from xml.sax.handler import all_features
import numpy as np
import torch
import datasets
import clip
import util.misc as utils
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset
from datasets import build_dataset, build_processed_dataset
from engine import preprocess_neighbors
from engine import encoder_evaluate_swig, evaluate_swig, evaluate_swig_zero_shot
from engine import encoder_train_one_epoch, decoder_train_one_epoch
from models import build_encoder_model, build_decoder_model
from util.myutils import load_mask_rcnn_model
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set Grounded Situation Recognition Transformer', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm',
                        default=0.1,
                        type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--encoder_batch_size', default=16, type=int)
    parser.add_argument('--decoder_batch_size', default=4, type=int)
    parser.add_argument('--encoder_epochs', default=0, type=int)
    parser.add_argument('--decoder_epochs', default=0, type=int)

    # Backbone parameters
    parser.add_argument('--backbone',
                        default='resnet50',
                        type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument(
        '--position_embedding',
        default='learned',
        type=str,
        choices=('sine', 'learned'),
        help="Type of positional embedding to use on top of the image features"
    )

    # Transformer parameters
    parser.add_argument('--num_enc_layers',
                        default=6,
                        type=int,
                        help="Number of encoding layers in GSRFormer")
    parser.add_argument('--num_dec_layers',
                        default=5,
                        type=int,
                        help="Number of decoding layers in GSRFormer")
    parser.add_argument(
        '--dim_feedforward',
        default=2048,
        type=int,
        help=
        "Intermediate size of the feedforward layers in the transformer blocks"
    )
    parser.add_argument(
        '--hidden_dim',
        default=512,
        type=int,
        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout',
                        default=0.15,
                        type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument(
        '--nheads',
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions")

    # Loss coefficients
    parser.add_argument('--noun_loss_coef', default=2, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--bbox_conf_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=5, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_file', default='swig')
    parser.add_argument('--swig_path', type=str, default="SWiG")
    parser.add_argument('--dev', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')

    # Etc...
    parser.add_argument('--inference', default=False)
    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--encoder_start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='encoder start epoch')
    parser.add_argument('--decoder_start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='decoder start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--encoder_saved_model',
                        default='GSRFormer/encoder_checkpoint.pth',
                        help='path where saved encoder model is')
    parser.add_argument('--decoder_saved_model',
                        default='GSRFormer/decoder_checkpoint.pth',
                        help='path where saved decoder model is')
    parser.add_argument('--load_saved_encoder', default=False, type=bool)
    parser.add_argument('--load_saved_decoder', default=False, type=bool)
    parser.add_argument('--preprocess', default=False, type=bool)
    parser.add_argument('--images_per_segment', default=9463, type=int)
    parser.add_argument('--images_per_eval_segment', default=12600, type=int)

    # Distributed training parameters
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # check dataset
    if args.dataset_file == "swig":
        from datasets.swig import collater, processed_collater
    else:
        assert False, f"dataset {args.dataset_file} is not supported now"

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    args.num_noun_classes = dataset_train.num_nouns()
    dataset_test = build_dataset(image_set='test', args=args)

    # load clip model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    text_prompts = []
    for verb_name in dataset_test.verb_to_idx:
        text_prompts.append("An image of {}".format(verb_name))
    text = clip.tokenize(text_prompts).to(device)
    text_features = clip_model.encode_text(text)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    # The ⟨name⟩ is a ⟨role⟩ of ⟨verb⟩

    # load mask-rcnn model
    mask_rcnn_model = load_mask_rcnn_model()
    
    # build Encoder Transformer model
    
    # build Decoder Transformer Model
    _, decoder_criterion = build_decoder_model(args)

    # Dataset Sampler
    # For Encoder Transformer
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    # For preprocessing
    if args.preprocess == True:
        preprocess_sampler_test = torch.utils.data.RandomSampler(dataset_test)

    output_dir = Path(args.output_dir)
    # dataset loader
    # For Encoder Transformer
    data_loader_test = DataLoader(dataset_test,
                                    num_workers=args.num_workers,
                                    drop_last=False,
                                    collate_fn=collater,
                                    sampler=sampler_test)
    # For Preprocessing
    if args.preprocess == True:
        batch_preprocess_sampler_test = torch.utils.data.BatchSampler(
            preprocess_sampler_test, args.encoder_batch_size, drop_last=False)
        preprocess_data_loader_test = DataLoader(
            dataset_test,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=collater,
            batch_sampler=batch_preprocess_sampler_test)

    # use saved model for evaluation (using dev set or test set)
    if args.dev or args.test:
        # build dataset
        if args.test:
            with open("__storage__/testDict.json") as test_json:
                test_dict = json.load(test_json)
            # processed_dataset_test = build_processed_dataset(
            #     image_set='test', args=args, neighbors_dict=test_dict)
            if args.distributed:
                sampler_test = DistributedSampler(dataset_test,
                                                  shuffle=False)
            else:
                sampler_test = torch.utils.data.SequentialSampler(
                    dataset_test)
            data_loader = DataLoader(dataset_test,
                                     num_workers=args.num_workers,
                                     drop_last=False,
                                     collate_fn=collater,
                                     sampler=sampler_test)
        else:
            print("zero shot only support test.")

        test_stats = evaluate_swig_zero_shot(
                                   decoder_criterion, data_loader, dataset_test, device, 
                                   args.output_dir, clip_model, clip_preprocess, text, text_features, mask_rcnn_model)
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / ("res_dev.txt"
                  if args.dev else "res_test.txt")).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            with open(
                    output_dir /
                    ("res_dev.json" if args.dev else "res_test.json"),
                    "w") as f1:
                json.dump(log_stats, f1)

        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'GSRFormer training and evaluation script',
        parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
