# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb

import utils
from utils.datasets_semseg import (ade_classes, hypersim_classes,
                                   nyu_v2_40_classes)


def inv_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of the normalization that was done during pre-processing
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    return inv_normalize(tensor)


@torch.no_grad()
def log_semseg_wandb(
        images: torch.Tensor, 
        preds: List[np.ndarray], 
        gts: List[np.ndarray],
        depth_gts: List[np.ndarray],
        dataset_name: str = 'ade20k',
        image_count=8, 
        prefix=""
    ):

    if dataset_name == 'ade20k':
        classes = ade_classes()
    elif dataset_name == 'hypersim':
        classes = hypersim_classes()
    elif dataset_name == 'nyu':
        classes = nyu_v2_40_classes()
    else:
        raise ValueError(f'Dataset {dataset_name} not supported for logging to wandb.')

    class_labels = {i: cls for i, cls in enumerate(classes)}
    class_labels[len(classes)] = "void"
    class_labels[utils.SEG_IGNORE_INDEX] = "ignore"

    image_count = min(len(images), image_count)

    images = images[:image_count]
    preds = preds[:image_count]
    gts = gts[:image_count]
    depth_gts = depth_gts[:image_count] if len(depth_gts) > 0 else None

    semseg_images = {}

    for i, (image, pred, gt) in enumerate(zip(images, preds, gts)):
        image = inv_norm(image)
        pred[gt == utils.SEG_IGNORE_INDEX] = utils.SEG_IGNORE_INDEX

        semseg_image = wandb.Image(image, masks={
            "predictions": {
                "mask_data": pred,
                "class_labels": class_labels,
            },
            "ground_truth": {
                "mask_data": gt,
                "class_labels": class_labels,
            }
        })

        semseg_images[f"{prefix}_{i}"] = semseg_image

        if depth_gts is not None:
            semseg_images[f"{prefix}_{i}_depth"] = wandb.Image(depth_gts[i])

    wandb.log(semseg_images, commit=False)


@torch.no_grad()
def log_taskonomy_wandb(
        preds: Dict[str, torch.Tensor], 
        gts: Dict[str, torch.Tensor], 
        image_count=8, 
        prefix=""
    ):
    pred_tasks = list(preds.keys())
    gt_tasks = list(gts.keys())
    if 'mask_valid' in gt_tasks:
        gt_tasks.remove('mask_valid')

    image_count = min(len(preds[pred_tasks[0]]), image_count)

    all_images = {}

    for i in range(image_count):

        # Log GTs
        for task in gt_tasks:
            gt_img = gts[task][i]
            if task == 'rgb':
                gt_img = inv_norm(gt_img)
            if gt_img.shape[0] == 1:
                gt_img = gt_img[0]
            elif gt_img.shape[0] == 2:
                gt_img = F.pad(gt_img, (0,0,0,0,0,1), mode='constant', value=0.0)

            gt_img = wandb.Image(gt_img, caption=f'GT #{i}')
            key = f'{prefix}_gt_{task}'
            if key not in all_images:
                all_images[key] = [gt_img]
            else:
                all_images[key].append(gt_img)

        # Log preds
        for task in pred_tasks:
            pred_img = preds[task][i]
            if task == 'rgb':
                pred_img = inv_norm(pred_img)
            if pred_img.shape[0] == 1:
                pred_img = pred_img[0]
            elif pred_img.shape[0] == 2:
                pred_img = F.pad(pred_img, (0,0,0,0,0,1), mode='constant', value=0.0)

            pred_img = wandb.Image(pred_img, caption=f'Pred #{i}')
            key = f'{prefix}_pred_{task}'
            if key not in all_images:
                all_images[key] = [pred_img]
            else:
                all_images[key].append(pred_img)

    wandb.log(all_images, commit=False)
