# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on BEiT, timm, DINO, DeiT and MAE-priv code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from utils import create_transform

from .data_constants import (IMAGE_TASKS, IMAGENET_DEFAULT_MEAN,
                             IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN,
                             IMAGENET_INCEPTION_STD)
from .dataset_folder import ImageFolder, MultiTaskImageFolder


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )


class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        trans = [transforms.RandomResizedCrop(args.input_size)]
        if args.hflip > 0.0:
            trans.append(transforms.RandomHorizontalFlip(args.hflip))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))])

        self.transform = transforms.Compose(trans)

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


class DataAugmentationForMultiMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        self.rgb_mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.rgb_std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.input_size = args.input_size
        self.hflip = args.hflip

    def __call__(self, task_dict):
        flip = random.random() < self.hflip # Stores whether to flip all images or not
        ijhw = None # Stores crop coordinates used for all tasks
        
        # Crop and flip all tasks randomly, but consistently for all tasks
        for task in task_dict:
            if task not in IMAGE_TASKS:
                continue
            if ijhw is None:
                # Official MAE code uses (0.2, 1.0) for scale and (0.75, 1.3333) for ratio
                ijhw = transforms.RandomResizedCrop.get_params(
                    task_dict[task], scale=(0.2, 1.0), ratio=(0.75, 1.3333)
                )
            i, j, h, w = ijhw
            task_dict[task] = TF.crop(task_dict[task], i, j, h, w)
            task_dict[task] = task_dict[task].resize((self.input_size, self.input_size))
            if flip:
                task_dict[task] = TF.hflip(task_dict[task])
                
        # Convert to Tensor
        for task in task_dict:
            if task in ['depth']:
                img = torch.Tensor(np.array(task_dict[task]) / 2 ** 16)
                img = img.unsqueeze(0)  # 1 x H x W
            elif task in ['rgb']:
                img = TF.to_tensor(task_dict[task])
                img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
            elif task in ['semseg', 'semseg_coco']:
                # TODO: add this to a config instead
                # Rescale to 0.25x size (stride 4)
                scale_factor = 0.25
                img = task_dict[task].resize((int(self.input_size * scale_factor), int(self.input_size * scale_factor)))
                # Using pil_to_tensor keeps it in uint8, to_tensor converts it to float (rescaled to [0, 1])
                img = TF.pil_to_tensor(img).to(torch.long).squeeze(0)
                
            task_dict[task] = img
        
        return task_dict

    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        #repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr

def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)

def build_multimae_pretraining_dataset(args):
    transform = DataAugmentationForMultiMAE(args)
    return MultiTaskImageFolder(args.data_path, args.all_domains, transform=transform)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
