# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math

import torch
from einops import rearrange


def vit_to_multimae(multimae_state_dict):
    """
    Converts timm ViT weights to MultiMAE weights.
    """
    state_dict = {}
    state_dict['global_tokens'] = multimae_state_dict['cls_token']
    for k,v in multimae_state_dict.items():
        if k == 'pos_embed':
            n = int(math.sqrt(v.shape[1]))
            pos_embed = rearrange(v[:,1:], 'b (n1 n2) d -> b d n1 n2', n1=n, n2=n)
            state_dict['global_tokens'] += v[:,0]
            state_dict['input_adapters.rgb.pos_emb'] = pos_embed
        elif k == 'patch_embed.proj.weight':
            state_dict['input_adapters.rgb.proj.weight'] = v
        elif k == 'patch_embed.proj.bias':
            state_dict['input_adapters.rgb.proj.bias'] = v
        elif 'blocks.' in k:
            state_dict[k.replace('blocks.', 'encoder.')] = v
    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="ViT to MultiMAE checkpoint converter")
    parser.add_argument(
        "--vit_ckpt_path", type=str,
        help="Path to converted ViT(MultiMAE) checkpoint"
    )
    parser.add_argument(
        "--multimae_ckpt_path", type=str,
        help="Path to MultiMAE checkpoint"
    )
    args = parser.parse_args()
    
    print(f'Loading weights at {args.vit_ckpt_path}')
    ckpt = torch.load(args.vit_ckpt_path)
    print('Converting from ViT weights to MultiMAE weights...')
    ckpt['model'] = vit_to_multimae(ckpt['model'])
    torch.save(ckpt, args.multimae_ckpt_path)
    print(f'Saved converted weights at {args.multimae_ckpt_path}')
