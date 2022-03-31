# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the timm and MAE-priv code base
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

CIFAR_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_DEFAULT_STD = (0.2023, 0.1994, 0.2010)

SEG_IGNORE_INDEX = 255
PAD_MASK_VALUE = 254
COCO_SEMSEG_NUM_CLASSES = 133

IMAGE_TASKS = ['rgb', 'depth', 'semseg', 'semseg_coco']

NYU_MEAN = 2070.7764
NYU_STD = 777.5723

# Data paths
IMAGENET_TRAIN_PATH = 'ADD_DATA_PATH_HERE'
IMAGENET_VAL_PATH = 'ADD_DATA_PATH_HERE'

ADE_TRAIN_PATH = 'ADD_DATA_PATH_HERE'
ADE_VAL_PATH = 'ADD_DATA_PATH_HERE'

HYPERSIM_TRAIN_PATH = 'ADD_DATA_PATH_HERE'
HPYERSIM_VAL_PATH = 'ADD_DATA_PATH_HERE'
HYPERSIM_TEST_PATH = 'ADD_DATA_PATH_HERE'

NYU_TRAIN_PATH = 'ADD_DATA_PATH_HERE'
NYU_TEST_PATH = 'ADD_DATA_PATH_HERE'

TASKONOMY_PATH = 'ADD_DATA_PATH_HERE'
