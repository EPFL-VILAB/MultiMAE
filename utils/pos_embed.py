# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on BEiT, timm, DINO DeiT and MAE-priv code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

import re

import torch


def interpolate_pos_embed_vit(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def interpolate_pos_embed_multimae(model, checkpoint_model):
    pattern = "input_adapters\.(.*)\.pos_emb"
    matched_keys = [k for k in checkpoint_model if bool(re.match(pattern, k))]

    for key in matched_keys:
        domain = re.match(pattern, key).group(1)  # group(0) is entire matched regex
        if getattr(model.input_adapters, domain, None) is not None:
            pos_embed_checkpoint = checkpoint_model[key]
            _, _, orig_H, orig_W = pos_embed_checkpoint.shape
            _, _, new_H, new_W = getattr(model.input_adapters, domain).pos_emb.shape
            if (orig_H != new_H) or (orig_W != new_W):
                print(f"Key {key}: Position interpolate from {orig_H}x{orig_W} to {new_H}x{new_W}")
                pos_embed_checkpoint = torch.nn.functional.interpolate(
                    pos_embed_checkpoint, size=(new_H, new_W), mode='bicubic', align_corners=False)
                checkpoint_model[key] = pos_embed_checkpoint
