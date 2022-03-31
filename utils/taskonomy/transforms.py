from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .task_configs import task_parameters

MAKE_RESCALE_0_1_NEG1_POS1   = lambda n_chan: transforms.Normalize([0.5]*n_chan, [0.5]*n_chan)
RESCALE_0_1_NEG1_POS1        = transforms.Normalize([0.5], [0.5])  # This needs to be different depending on num out chans
MAKE_RESCALE_0_MAX_NEG1_POS1 = lambda maxx: transforms.Normalize([maxx / 2.], [maxx * 1.0])
RESCALE_0_255_NEG1_POS1      = transforms.Normalize([127.5,127.5,127.5], [255, 255, 255])
MAKE_RESCALE_0_MAX_0_POS1 = lambda maxx: transforms.Normalize([0.0], [maxx * 1.0])
STD_IMAGENET = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    

# For semantic segmentation
transform_dense_labels = lambda img: torch.Tensor(np.array(img)).long()  # avoids normalizing

# Transforms to a 3-channel tensor and then changes [0,1] -> [0, 1]
transform_8bit = transforms.Compose([
        transforms.ToTensor(),
    ])
    
# Transforms to a n-channel tensor and then changes [0,1] -> [0, 1]. Keeps only the first n-channels
def transform_8bit_n_channel(n_channel=1, crop_channels=True):
    if crop_channels:
        crop_channels_fn = lambda x: x[:n_channel] if x.shape[0] > n_channel else x
    else: 
        crop_channels_fn = lambda x: x
    return transforms.Compose([
            transforms.ToTensor(),
            crop_channels_fn,
        ])

# Transforms to a 1-channel tensor and then changes [0,1] -> [0, 1].
def transform_16bit_single_channel(im):
    im = transforms.ToTensor()(np.array(im))
    im = im.float() / (2 ** 16 - 1.0) 
    return im

def make_valid_mask(mask_float, max_pool_size=4):
    '''
        Creates a mask indicating the valid parts of the image(s).
        Enlargens masked area using a max pooling operation.

        Args:
            mask_float: A (b x c x h x w) mask as loaded from the Taskonomy loader.
            max_pool_size: Parameter to choose how much to enlarge masked area.
    '''
    squeeze = False
    if len(mask_float.shape) == 3:
        mask_float = mask_float.unsqueeze(0)
        squeeze = True
    _, _, h, w = mask_float.shape
    mask_float = 1 - mask_float
    mask_float = F.max_pool2d(mask_float, kernel_size=max_pool_size)
    mask_float = F.interpolate(mask_float, (h, w), mode='nearest')
    mask_valid = mask_float == 0
    mask_valid = mask_valid[0] if squeeze else mask_valid
    return mask_valid


def task_transform(file, task: str, image_size=Optional[int]):
    transform = None

    if task in ['rgb']:
        transform = transforms.Compose([
            transform_8bit,
            STD_IMAGENET
        ])
    elif task in ['normal']:
        transform = transform_8bit
    elif task in ['mask_valid']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            make_valid_mask
        ])
    elif task in ['keypoints2d', 'keypoints3d', 'depth_euclidean', 'depth_zbuffer', 'edge_texture']:
        transform = transform_16bit_single_channel
    elif task in ['edge_occlusion']:
        transform = transforms.Compose([
            transform_16bit_single_channel,
            transforms.GaussianBlur(3, sigma=1)
        ])
    elif task in ['principal_curvature', 'curvature']:
        transform = transform_8bit_n_channel(2)
    elif task in ['reshading']:
        transform = transform_8bit_n_channel(1)
    elif task in ['segment_semantic', 'segment_instance', 'segment_panoptic', 'fragments', 'segment_unsup2d', 'segment_unsup25d']:  # this is stored as 1 channel image (H,W) where each pixel value is a different class
        transform = transform_dense_labels
    elif task in ['class_object', 'class_scene']:
        transform = torch.Tensor
        image_size = None
    else:
        transform = None
    
    if 'threshold_min' in task_parameters[task]:
        threshold = task_parameters[task]['threshold_min']
        transform = transforms.Compose([
            transform,
            lambda x: torch.threshold(x, threshold, 0.0)
        ])
    if 'clamp_to' in task_parameters[task]:
        minn, maxx = task_parameters[task]['clamp_to']
        if minn > 0:
            raise NotImplementedError("Rescaling (min1, max1) -> (min2, max2) not implemented for min1, min2 != 0 (task {})".format(task))
        transform = transforms.Compose([
            transform,
            lambda x: torch.clamp(x, minn, maxx),
            MAKE_RESCALE_0_MAX_0_POS1(maxx)
        ])
    

    if image_size is not None:
        if task == 'fragments':
            resize_frag = lambda frag: F.interpolate(frag.permute(2,0,1).unsqueeze(0).float(), image_size, mode='nearest').long()[0].permute(1,2,0)
            transform = transforms.Compose([
                transform,
                resize_frag
            ])
        else:
            resize_method = transforms.InterpolationMode.BILINEAR if task in ['rgb'] else transforms.InterpolationMode.NEAREST
            transform = transforms.Compose([
                transforms.Resize(image_size, resize_method),
                transform
            ])

    if transform is not None:
        file = transform(file)
        
    return file
