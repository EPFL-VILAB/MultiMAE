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
from typing import Dict, Tuple

import numpy as np
import torch

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except:
    print('albumentations not installed')
import cv2
import torch.nn.functional as F

from utils import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PAD_MASK_VALUE,
                   SEG_IGNORE_INDEX)

from .dataset_folder import ImageFolder, MultiTaskImageFolder


def simple_transform(train: bool,
                     additional_targets: Dict[str, str],
                     input_size: int =512,
                     pad_value: Tuple[int, int, int] = (128, 128, 128),
                     pad_mask_value: int =PAD_MASK_VALUE):
    """Default transform for semantic segmentation, applied on all modalities

    During training:
        1. Random horizontal Flip
        2. Rescaling so that longest side matches input size
        3. Color jitter (for RGB-modality only)
        4. Large scale jitter (LSJ)
        5. Padding
        6. Random crop to given size
        7. Normalization with ImageNet mean and std dev

    During validation / test:
        1. Rescaling so that longest side matches given size
        2. Padding
        3. Normalization with ImageNet mean and std dev
     """

    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.LongestMaxSize(max_size=input_size, p=1),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5),  # Color jittering from MoCo-v3 / DINO
            A.RandomScale(scale_limit=(0.1 - 1, 2.0 - 1), p=1),  # This is LSJ (0.1, 2.0)
            A.PadIfNeeded(min_height=input_size, min_width=input_size,
                          position=A.augmentations.PadIfNeeded.PositionType.TOP_LEFT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=pad_value, mask_value=pad_mask_value),
            A.RandomCrop(height=input_size, width=input_size, p=1),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    else:
        transform = A.Compose([
            A.LongestMaxSize(max_size=input_size, p=1),
            A.PadIfNeeded(min_height=input_size, min_width=input_size,
                          position=A.augmentations.PadIfNeeded.PositionType.TOP_LEFT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=pad_value, mask_value=pad_mask_value),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    return transform


class DataAugmentationForSemSeg(object):
    """Data transform / augmentation for semantic segmentation downstream tasks.
    """

    def __init__(self, transform, seg_num_classes, seg_ignore_index=SEG_IGNORE_INDEX, standardize_depth=True,
                 seg_reduce_zero_label=False, seg_use_void_label=False):

        self.transform = transform
        self.seg_num_classes = seg_num_classes
        self.seg_ignore_index = seg_ignore_index
        self.standardize_depth = standardize_depth
        self.seg_reduce_zero_label = seg_reduce_zero_label
        self.seg_use_void_label = seg_use_void_label

    @staticmethod
    def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
        img[img == PAD_MASK_VALUE] = torch.nan
        if mask_valid is not None:
            # This is if we want to apply masking before standardization
            img[~mask_valid] = torch.nan
        sorted_img = torch.sort(torch.flatten(img))[0]
        # Remove nan, nan at the end of sort
        num_nan = sorted_img.isnan().sum()
        if num_nan > 0:
            sorted_img = sorted_img[:-num_nan]
        # Remove outliers
        trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
        trunc_mean = trunc_img.mean()
        trunc_var = trunc_img.var()
        eps = 1e-6
        # Replace nan by mean
        img = torch.nan_to_num(img, nan=trunc_mean)
        # Standardize
        img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
        return img

    def seg_adapt_labels(self, img):
        if self.seg_use_void_label:
            # Set void label to num_classes
            if self.seg_reduce_zero_label:
                pad_replace = self.seg_num_classes + 1
            else:
                pad_replace = self.seg_num_classes
        else:
            pad_replace = self.seg_ignore_index
        img[img == PAD_MASK_VALUE] = pad_replace

        if self.seg_reduce_zero_label:
            img[img == 0] = self.seg_ignore_index
            img = img - 1
            img[img == self.seg_ignore_index - 1] = self.seg_ignore_index

        return img

    def __call__(self, task_dict):

        # Need to replace rgb key to image
        task_dict['image'] = task_dict.pop('rgb')
        # Convert to np.array
        task_dict = {k: np.array(v) for k, v in task_dict.items()}

        task_dict = self.transform(**task_dict)

        # And then replace it back to rgb
        task_dict['rgb'] = task_dict.pop('image')

        for task in task_dict:
            if task in ['depth']:
                img = task_dict[task].to(torch.float)
                if self.standardize_depth:
                    # Mask valid set to None here, as masking is applied after standardization
                    img = self.standardize_depth_map(img, mask_valid=None)
                if 'mask_valid' in task_dict:
                    mask_valid = (task_dict['mask_valid'] == 255).squeeze()
                    img[~mask_valid] = 0.0
                task_dict[task] = img.unsqueeze(0)
            elif task in ['rgb']:
                task_dict[task] = task_dict[task].to(torch.float)
            elif task in ['semseg']:
                img = task_dict[task].to(torch.long)
                img = self.seg_adapt_labels(img)
                task_dict[task] = img
            elif task in ['pseudo_semseg']:
                # If it's pseudo-semseg, then it's an input modality and should therefore be resized
                img = task_dict[task]
                img = F.interpolate(img[None,None,:,:], scale_factor=0.25, mode='nearest').long()[0,0]
                task_dict[task] = img

        return task_dict


def build_semseg_dataset(args, data_path, transform, max_images=None):
    transform = DataAugmentationForSemSeg(transform=transform, seg_num_classes=args.num_classes,
                                          standardize_depth=args.standardize_depth,
                                          seg_reduce_zero_label=args.seg_reduce_zero_label,
                                          seg_use_void_label=args.seg_use_void_label)
    prefixes = {'depth': 'pseudo_'} if args.load_pseudo_depth else None
    return MultiTaskImageFolder(data_path, args.all_domains, transform=transform, prefixes=prefixes, max_images=max_images)


def ade_classes():
    """ADE20K class names for external use."""
    return [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag'
    ]


def hypersim_classes():
    """Hypersim class names for external use."""
    return [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 
        'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 
        'curtain', 'dresser', 'pillow', 'mirror', 'floor-mat', 'clothes', 
        'ceiling', 'books', 'fridge', 'TV', 'paper', 'towel', 'shower-curtain', 
        'box', 'white-board', 'person', 'night-stand', 'toilet', 'sink', 'lamp',
        'bathtub', 'bag', 'other-struct', 'other-furntr', 'other-prop'
    ]


def nyu_v2_40_classes():
    """NYUv2 40 class names for external use."""
    return [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 
        'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 
        'curtain', 'dresser', 'pillow', 'mirror', 'floor-mat', 'clothes', 
        'ceiling', 'books', 'fridge', 'TV', 'paper', 'towel', 'shower-curtain', 
        'box', 'white-board', 'person', 'night-stand', 'toilet', 'sink', 'lamp',
        'bathtub', 'bag', 'other-struct', 'other-furntr', 'other-prop'
    ]
