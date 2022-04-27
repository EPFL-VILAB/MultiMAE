# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import math
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from einops import rearrange

import utils
import utils.data_constants as data_constants
from multimae import multimae
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import ConvNeXtAdapter, DPTOutputAdapter
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import NYU_MEAN, NYU_STD
from utils.dataset_regression import build_regression_dataset, nyu_transform
from utils.log_images import log_taskonomy_wandb
from utils.optim_factory import LayerDecayValueAssigner, create_optimizer
from utils.pos_embed import interpolate_pos_embed_multimae


def masked_mse_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)
    element_wise_loss = (preds - target)**2
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()


def masked_l1_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()


def masked_berhu_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)

    diff = preds - target
    diff[~mask_valid] = 0
    with torch.no_grad():
        c = max(torch.abs(diff).max() * 0.2, 1e-5)

    l1_loss = torch.abs(diff)
    l2_loss = (torch.square(diff) + c**2) / 2. / c
    berhu_loss = l1_loss[torch.abs(diff) < c].sum() + l2_loss[torch.abs(diff) >= c].sum()

    return berhu_loss / mask_valid.sum()

@torch.no_grad()
def masked_nyu_metrics(preds, target, mask_valid=None):
    # map to the original scale 
    preds = preds * NYU_STD + NYU_MEAN
    target = target * NYU_STD + NYU_MEAN

    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)

    n = mask_valid.sum()
    
    diff = torch.abs(preds - target)
    diff[~mask_valid] = 0
    
    max_rel = torch.maximum(preds/torch.clamp_min(target, 1e-6), target/torch.clamp_min(preds, 1e-6))
    max_rel = max_rel[mask_valid]

    log_diff = torch.log(torch.clamp_min(preds, 1e-6)) - torch.log(torch.clamp_min(target, 1e-6))
    log_diff[~mask_valid] = 0

    metrics = {
        'rmse': (diff.square().sum() / n).sqrt(),
        'rel': (diff/torch.clamp_min(target, 1e-6))[mask_valid].mean(),
        'srel': (diff**2/torch.clamp_min(target, 1e-6))[mask_valid].mean(),
        'log10': (log_diff.square().sum() / n).sqrt(),
        'delta_1': (max_rel < 1.25).float().mean(),
        'delta_2': (max_rel < (1.25**2)).float().mean(),
        'delta_3': (max_rel < (1.25**3)).float().mean(),
    }
    return metrics

DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'aug_type': 'image',
    },
    'depth': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'aug_type': 'mask',
    },
    'mask_valid': {
        'stride_level': 1,
        'aug_type': 'mask',
    },
}


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE depth fine-tuning script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--save_ckpt_freq', default=200, type=int)
    parser.add_argument('--tmp', default=False, action='store_true')

    # Task parameters
    parser.add_argument('--in_domains', default='rgb', type=str,
                        help='Input domain names, separated by hyphen')
    parser.add_argument('--decoder_main_tasks', type=str, default='rgb',
                        help='for convnext & DPT adapters, separate tasks with a hyphen')
    parser.add_argument('--standardize_depth', action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=False)

    # Model parameters
    parser.add_argument('--model', default='multivit_base', type=str, metavar='MODEL',
                        help='Name of MultiViT model to train')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='number of global tokens to add to encoder')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='base patch size for image-like modalities')
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size for backbone')
    parser.add_argument('--drop_path_encoder', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    parser.add_argument('--output_adapter', type=str, default='dpt',
                        choices=['dpt', 'convnext'],
                        help='One of [dpt, convnext] (default: dpt)')
    parser.add_argument('--decoder_dim', default=6144, type=int,
                        help='Token dimension inside the decoder layers (for convnext adapter)')
    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='Depth of decoder (for convnext adapter)')
    parser.add_argument('--drop_path_decoder', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--decoder_preds_per_patch', type=int, default=64,
                        help='Predictions per patch for convnext adapter')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--decoder_decay', type=float, default=None,
                        help='decoder weight decay')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate  (default: 1e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (0.0)')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA')
    parser.add_argument('--scale_input_lr', action='store_true')
    parser.add_argument('--no_scale_input_lr', action='store_false', dest='scale_input_lr')
    parser.set_defaults(scale_input_lr=True)
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--no_freeze_transformer', action='store_false', dest='freeze_transformer')
    parser.set_defaults(freeze_transformer=False)

    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Finetuning parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--loss', default='berhu',
                        help='Loss to use. One of [l1, l2, berhu] (default: berhu)')

    # Dataset parameters
    parser.add_argument('--data_path', default=data_constants.NYU_TRAIN_PATH, type=str, help='dataset path')
    parser.add_argument('--eval_data_path', default=data_constants.NYU_TEST_PATH, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--aug_name', default='nyu-augs', type=str)
    parser.add_argument('--color_augs', default=False, action='store_true')
    parser.add_argument('--no_color_augs', dest='color_augs', default=False, action='store_false')
    parser.add_argument('--eval_freq', default=1, type=int, help="frequency of evaluation")
    parser.add_argument('--max_train_images', default=1000, type=int, help='number of train images')
    parser.add_argument('--max_val_images', default=100, type=int, help='number of validation images')
    parser.add_argument('--max_test_images', default=100, type=int, help='number of test images')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--test', action='store_true',
                        help='Perform testing only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                    help='Enabling distributed evaluation')
    parser.add_argument('--no_dist_eval', action='store_false', dest='dist_eval',
                    help='Disabling distributed evaluation')
    parser.set_defaults(dist_eval=False)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', dest='log_wandb', default=False, action='store_false',
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='user or team name of wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='run name on wandb')
    parser.add_argument('--wandb_group', default='', type=str)
    parser.add_argument('--log_images_wandb', action='store_true')
    parser.add_argument('--log_images_freq', default=5, type=int,
                        help="Frequency of image logging (in epochs)")
    parser.add_argument('--show_user_warnings', default=False, action='store_true')


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args


def main(args):

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    args.in_domains = args.in_domains.split('-')
    args.out_domains = ['depth']
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))
    if args.use_mask_valid:
        args.all_domains.append('mask_valid')
    if 'rgb' not in args.all_domains:
        args.all_domains.append('rgb')

    args.decoder_main_tasks = args.decoder_main_tasks.split('-')
    for task in args.decoder_main_tasks:
        assert task in args.in_domains, f'Readout task {task} must be in in_domains.'

    additional_targets = {domain: DOMAIN_CONF[domain]['aug_type'] for domain in args.all_domains}

    if args.aug_name == 'nyu-augs':
        train_transform = nyu_transform(train=True, additional_targets=additional_targets, input_size=args.input_size, color_aug=args.color_augs)
        val_transform = nyu_transform(train=False, additional_targets=additional_targets, input_size=args.input_size)
    else:
        raise ValueError(f"Invalid aug: {args.aug_name}")

    dataset_train = build_regression_dataset(args, data_path=args.data_path, transform=train_transform)
    dataset_val = build_regression_dataset(args, data_path=args.eval_data_path, transform=val_transform, max_images=args.max_val_images)
    dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True,
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if dataset_test is not None:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            if dataset_test is not None:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if dataset_test is not None:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5*args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,
    )

    # Model

    # Rename depth task
    if 'depth_zbuffer' in args.in_domains:
        args.in_domains.remove('depth_zbuffer')
        args.in_domains.append('depth')
    if 'depth_zbuffer' in args.out_domains:
        args.out_domains.remove('depth_zbuffer')
        args.out_domains.append('depth')

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            image_size=args.input_size,
        )
        for domain in args.in_domains
    }

    # DPT settings are fixed for ViT-B. Modify them if using a different backbone.
    if args.model != 'multivit_base' and args.output_adapter == 'dpt':
        raise NotImplementedError('Unsupported backbone: DPT head is fixed for ViT-B.')

    adapters_dict = {
        'dpt': DPTOutputAdapter,
        'convnext': partial(ConvNeXtAdapter, preds_per_patch=64),
    }

    output_adapters = {
        domain: adapters_dict[args.output_adapter](
            num_classes=DOMAIN_CONF[domain]['channels'],
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size=args.patch_size,
            main_tasks=args.decoder_main_tasks
        )
        for domain in args.out_domains
    }

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        drop_path_rate=args.drop_path_encoder,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu')
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']

        # # Remove keys for semantic segmentation
        # for k in list(checkpoint_model.keys()):
        #     if "semseg" in k:
        #         del checkpoint_model[k]

        # Remove output adapters
        for k in list(checkpoint_model.keys()):
            if "output_adapters" in k:
                del checkpoint_model[k]

        # Interpolate position embedding
        interpolate_pos_embed_multimae(model, checkpoint_model)

        # Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    # Optionally freeze the encoder
    if args.freeze_transformer:
        raise NotImplementedError
        for param in model.encoder.parameters():
            param.requires_grad = False

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    if args.loss == 'l1':
        tasks_loss_fn = {
            'depth': masked_l1_loss
        }
    elif args.loss == 'berhu':
        tasks_loss_fn = {
            'depth': masked_berhu_loss
        }
    else:
        raise NotImplementedError

    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        # idx=0: input adapters, idx>0: transformer layers 
        layer_decay_values = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        if not args.scale_input_lr:
            layer_decay_values[0] = 1.0
        assigner = LayerDecayValueAssigner(layer_decay_values)
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler(enabled=False)

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # Specifies if transformer encoder should only return last layer or all layers for DPT
    return_all_layers = args.output_adapter in ['dpt']

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        val_stats = evaluate(model=model, tasks_loss_fn=tasks_loss_fn, data_loader=data_loader_val,
                             device=device, epoch=-1, in_domains=args.in_domains, mode='val', log_images=True,
                             return_all_layers=return_all_layers, standardize_depth=args.standardize_depth)
        print(f"Performance of the network on the {len(dataset_val)} validation images")
        print(f"Loss {val_stats['loss']:.3f}")
        if log_writer is not None:
            log_writer.set_step(args.start_epoch * num_training_steps_per_epoch)
            log_writer.update({**{f'val/{k}': v for k, v in val_stats.items()}, 'epoch': args.start_epoch})
        exit(0)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_val_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        log_images = args.log_wandb and args.log_images_wandb and (epoch % args.log_images_freq == 0)
        train_stats = train_one_epoch(
            model=model, tasks_loss_fn=tasks_loss_fn, data_loader=data_loader_train,
            optimizer=optimizer, device=device, epoch=epoch, loss_scaler=loss_scaler,
            max_norm=args.clip_grad, log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
            in_domains=args.in_domains, return_all_layers=return_all_layers,
            standardize_depth=args.standardize_depth,
            log_images=log_images,
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            log_images = args.log_wandb and args.log_images_wandb and (epoch % args.log_images_freq == 0)
            val_stats = evaluate(model=model, tasks_loss_fn=tasks_loss_fn, data_loader=data_loader_val,
                                 device=device, epoch=epoch, in_domains=args.in_domains, log_images=log_images,
                                 mode='val', return_all_layers=return_all_layers, standardize_depth=args.standardize_depth)
            if val_stats["loss"] < min_val_loss:
                min_val_loss = val_stats["loss"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best")
                print(f'New best val loss: {min_val_loss:.3f}')

            log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                         **{f'val/{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if log_writer is not None:
            log_writer.update(log_stats)

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Test with best checkpoint
    print('Loading model with best validation loss')
    checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pth'), map_location='cpu')
    state_dict = {}
    for k,v in checkpoint['model'].items():
        state_dict[f'module.{k}'] = v
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


def train_one_epoch(model: torch.nn.Module, tasks_loss_fn: Dict[str, torch.nn.Module], data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    loss_scaler, max_norm: float = 0, log_writer=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, in_domains=None,
                    return_all_layers=False, standardize_depth=True, log_images=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    pred_images = None
    gt_images = None

    for step, x in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        x = x[0]
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if 'depth_zbuffer' in x:
            x['depth'] = x['depth_zbuffer']
            del x['depth_zbuffer']

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        # Robust depth standardization
        if standardize_depth and 'depth' in input_dict:
            # Flatten depth and remove bottom and top 10% of non-masked values
            nan_depth = input_dict['depth'].clone()
            nan_depth[~tasks_dict['mask_valid']] = np.nan
            trunc_depth = torch.sort(rearrange(nan_depth, 'b c h w -> b (c h w)'), dim=1)[0]
            n_valid = (~torch.isnan(trunc_depth)).sum(dim=1)
            from_idxs, to_idxs = (n_valid * 0.1).long(), (n_valid * 0.9).long()
            robust_means = torch.stack([
                trunc_depth[batch_idx, from_idx:to_idx].mean() 
                for batch_idx, (from_idx, to_idx) in enumerate(zip(from_idxs, to_idxs))
            ])
            robust_vars = torch.stack([
                trunc_depth[batch_idx, from_idx:to_idx].var() 
                for batch_idx, (from_idx, to_idx) in enumerate(zip(from_idxs, to_idxs))
            ])
            input_dict['depth'] = (input_dict['depth'] - robust_means[:,None,None,None]) / torch.sqrt(robust_vars[:,None,None,None] + 1e-6)
            input_dict['depth'][~tasks_dict['mask_valid']] = 0.0
        
        # Mask invalid input values
        for task in input_dict:
            if task in ['rgb']:
                continue
            channels = input_dict[task].shape[1]
            input_dict[task][~tasks_dict['mask_valid'].repeat_interleave(repeats=channels, dim=1)] = 0.0

        # Forward + backward
        with torch.cuda.amp.autocast(enabled=False):
            # print('=====> input_dict:', {k: v.shape for k, v in input_dict.items()})

            preds = model(input_dict, return_all_layers=return_all_layers)
            task_losses = {
                task: tasks_loss_fn[task](preds[task].float(), tasks_dict[task], mask_valid=tasks_dict['mask_valid'])
                for task in preds
            }
            loss = sum(task_losses.values())

        loss_value = loss.item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        metrics = masked_nyu_metrics(preds['depth'], tasks_dict['depth'], mask_valid=tasks_dict['mask_valid'])

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        # loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # Metrics and logging
        metric_logger.update(**metrics)
        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        # metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_images and pred_images is None and utils.is_main_process():
            # Just log images of first batch
            pred_images = {task: v.detach().cpu().float() for task, v in preds.items()}
            gt_images = {task: v.detach().cpu().float() for task, v in input_dict.items()}
            gt_images.update({task: v.detach().cpu().float() for task, v in tasks_dict.items() if task not in gt_images})

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.set_step()

    if log_images and utils.is_main_process():
        prefix = 'train/img'
        log_taskonomy_wandb(pred_images, gt_images, prefix=prefix, image_count=8)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, tasks_loss_fn, data_loader, device, epoch, in_domains,
             log_images=False, mode='val', return_all_layers=False, standardize_depth=True):
    # Switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if mode == 'val':
        header = '(Eval) Epoch: [{}]'.format(epoch)
    elif mode == 'test':
        header = '(Test) Epoch: [{}]'.format(epoch)
    else:
        raise ValueError(f'Invalid eval mode {mode}')
    print_freq = 20

    pred_images = None
    gt_images = None

    for x in metric_logger.log_every(data_loader, print_freq, header):
        x = x[0]

        if 'depth_zbuffer' in x:
            x['depth'] = x['depth_zbuffer']
            del x['depth_zbuffer']
        
        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        # Robust depth standardization
        if standardize_depth and 'depth' in input_dict:
            # Flatten depth and remove bottom and top 10% of non-masked values
            nan_depth = input_dict['depth'].clone()
            nan_depth[~tasks_dict['mask_valid']] = np.nan
            trunc_depth = torch.sort(rearrange(nan_depth, 'b c h w -> b (c h w)'), dim=1)[0]
            n_valid = (~torch.isnan(trunc_depth)).sum(dim=1)
            from_idxs, to_idxs = (n_valid * 0.1).long(), (n_valid * 0.9).long()
            robust_means = torch.stack([
                trunc_depth[batch_idx, from_idx:to_idx].mean() 
                for batch_idx, (from_idx, to_idx) in enumerate(zip(from_idxs, to_idxs))
            ])
            robust_vars = torch.stack([
                trunc_depth[batch_idx, from_idx:to_idx].var() 
                for batch_idx, (from_idx, to_idx) in enumerate(zip(from_idxs, to_idxs))
            ])
            input_dict['depth'] = (input_dict['depth'] - robust_means[:,None,None,None]) / torch.sqrt(robust_vars[:,None,None,None] + 1e-6)
            input_dict['depth'][~tasks_dict['mask_valid']] = 0.0

        # Mask invalid input values
        for task in input_dict:
            if task in ['rgb']:
                continue
            channels = input_dict[task].shape[1]
            input_dict[task][~tasks_dict['mask_valid'].repeat_interleave(repeats=channels, dim=1)] = 0.0

        # Forward + backward
        with torch.cuda.amp.autocast(enabled=False):
            preds = model(input_dict, return_all_layers=return_all_layers)
            task_losses = {
                task: tasks_loss_fn[task](preds[task], tasks_dict[task], mask_valid=tasks_dict['mask_valid'])
                for task in preds
            }
            loss = sum(task_losses.values())

        loss_value = loss.item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        metrics = masked_nyu_metrics(preds['depth'], tasks_dict['depth'], mask_valid=tasks_dict['mask_valid'])

        metric_logger.update(**metrics)

        if log_images and pred_images is None and utils.is_main_process():
            # Just log images of first batch
            pred_images = {task: v.detach().cpu().float() for task, v in preds.items()}
            gt_images = {task: v.detach().cpu().float() for task, v in input_dict.items()}
            gt_images.update({task: v.detach().cpu().float() for task, v in tasks_dict.items() if task not in gt_images})

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)

    # Do before metrics so that void is not replaced
    if log_images and utils.is_main_process():
        prefix = 'val/img' if mode == 'val' else 'test/img'
        log_taskonomy_wandb(pred_images, gt_images, prefix=prefix, image_count=8)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(f'* Loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    opts = get_args()
    if opts.tmp:
        opts.output_dir = f'{opts.output_dir}-tmp'
    else:
        opts.output_dir = f'{opts.output_dir}-loss={opts.loss}-lr={opts.lr}-adapter={opts.output_adapter}-weight_decay={opts.weight_decay}-input_size={opts.input_size}-drop_path_encoder={opts.drop_path_encoder}-color_augs={opts.color_augs}'
    opts.wandb_run_name = f'{opts.wandb_run_name}-loss={opts.loss}-lr={opts.lr}-adapter={opts.output_adapter}-weight_decay={opts.weight_decay}'
    if opts.tmp:
        opts.wandb_run_name = f'tmp-{opts.wandb_run_name}'
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
