# --------------------------------------------------------
# Based on BEiT, timm, DINO DeiT and MAE-priv code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------
import json

import torch
from torch import optim as optim

try:
    from apex.optimizers import FusedAdam, FusedLAMB, FusedNovoGrad, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("input_adapters"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks") or var_name.startswith("encoder"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(
        model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, 
        decoder_decay=None, decoder_list=(), no_lr_scale_list=[]):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        elif decoder_decay is not None and (name.startswith("decoder.") or name in decoder_list):
            group_name = "decoder_decay"
            this_weight_decay = decoder_decay
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # Assign layer ID for LR scaling
        skip_scale = False
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f'{group_name}_no_lr_scale'
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None and not skip_scale:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    '''
    Model can either be a single nn.Module, or a dictionary with {'model': model, 'balancer': balancer}.
    '''
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    try:
        decoder_decay = args.decoder_decay
    except:
        decoder_decay = None
    try:
        no_lr_scale_list = args.no_lr_scale_list.split('-')
    except:
        no_lr_scale_list = []

    def get_parameters(m):
        if weight_decay and filter_bias_and_bn:
            skip = {}
            if skip_list is not None:
                skip = skip_list
            elif hasattr(m, 'no_weight_decay'):
                skip = m.no_weight_decay()
            decoder={}
            if hasattr(m, 'decoder_weight_decay'):
                decoder = m.decoder_weight_decay()
            parameters = get_parameter_groups(m, weight_decay, skip, get_num_layer, get_layer_scale, decoder_decay, decoder, no_lr_scale_list)
            wd = 0.
        else:
            parameters = m.parameters()
            wd = weight_decay
        return parameters, wd
    
    if isinstance(model, torch.nn.Module):
        parameters, weight_decay = get_parameters(model)
    elif isinstance(model, dict):
        parameters = [
            {
                "params": [p for n, p in model['model'].named_parameters()
                        if p.requires_grad],
                "lr_scale": 1.,
            },
            {
                "params": [p for n, p in model['balancer'].named_parameters()
                        if p.requires_grad],
                "lr_scale": args.balancer_lr_scale,
            },
        ]

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer
