# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
import torch.nn.functional as F
from einops import rearrange


def multimae_to_vit(multimae_state_dict):
    """
    Converts MultiMAE weights to timm ViT weights.
    Assumes that there is only 1 global token in the MultiMAE.
    """
    state_dict = {}
    for k,v in multimae_state_dict.items():
        if k == 'global_tokens':
            state_dict['cls_token'] = v
        elif k == 'input_adapters.rgb.pos_emb':
            state_dict['pos_embed'] = rearrange(v, 'b d h w -> b (h w) d')
            state_dict['pos_embed'] = F.pad(state_dict['pos_embed'], (0,0,1,0,0,0), mode='constant', value=0.0)
        elif k == 'input_adapters.rgb.proj.weight':
            state_dict['patch_embed.proj.weight'] = v
        elif k == 'input_adapters.rgb.proj.bias':
            state_dict['patch_embed.proj.bias'] = v
        elif 'encoder' in k:
            state_dict[k.replace('encoder', 'blocks')] = v
    return state_dict
    
def multimae_to_vitmultimae(multimae_state_dict):
    """
    Converts MultiMAE weights to timm-style ViTMultiMAE weights.
    Works with arbitrary number of global tokens.
    """
    state_dict = {}
    for k,v in multimae_state_dict.items():
        if k == 'global_tokens':
            state_dict['global_tokens'] = v
        elif k == 'input_adapters.rgb.pos_emb':
            state_dict['pos_embed'] = rearrange(v, 'b d h w -> b (h w) d')
        elif k == 'input_adapters.rgb.proj.weight':
            state_dict['patch_embed.proj.weight'] = v
        elif k == 'input_adapters.rgb.proj.bias':
            state_dict['patch_embed.proj.bias'] = v
        elif 'encoder' in k:
            state_dict[k.replace('encoder', 'blocks')] = v
    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="MultiMAE to ViT checkpoint converter")
    parser.add_argument(
        "--multimae_ckpt_path", type=str,
        help="Path to MultiMAE checkpoint"
    )
    parser.add_argument(
        "--vit_ckpt_path", type=str,
        help="Path to converted ViT(MultiMAE) checkpoint"
    )
    args = parser.parse_args()
    
    print(f'Loading weights at {args.multimae_ckpt_path}')
    ckpt = torch.load(args.multimae_ckpt_path)
    print('Converting from MultiMAE weights to ViT weights...')
    ckpt['model'] = multimae_to_vit(ckpt['model'])
    torch.save(ckpt, args.vit_ckpt_path)
    print(f'Saved converted weights at {args.vit_ckpt_path}')
