# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from .checkpoint import *
from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .data_constants import *
from .dist import *
from .logger import *
from .metrics import AverageMeter, accuracy
from .mixup import FastCollateMixup, Mixup
from .model import freeze, get_state_dict, unfreeze, unwrap_model
from .model_builder import create_model
from .model_ema import ModelEma, ModelEmaV2
from .native_scaler import *
from .optim_factory import create_optimizer
from .registry import model_entrypoint, register_model
from .task_balancing import *
from .taskonomy import *
from .transforms import *
from .transforms_factory import create_transform
