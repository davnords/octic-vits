# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from types import MethodType
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from dinov2.models.vision_transformer import *

def custom_fwd(
        self,
        x: torch.Tensor,
    ):

    outputs = self._get_intermediate_layers_not_chunked(x, 1)
    outputs = [self.norm(out) for out in outputs]
    class_tokens = [out[:, 0] for out in outputs]
    outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
    registers = [out[:, 1 : 1 + self.num_register_tokens] for out in outputs]

    B, _, w, h = x.shape
    outputs = [
        out.reshape(B, w // self.patch_size, h // self.patch_size, -1).contiguous()
        for out in outputs
    ]
    return class_tokens[-1], registers[-1], outputs[-1] 

def custom_fwd_d8(
        self,
        x: torch.Tensor,
    ):

    xs = self.prepare_tokens_with_masks(x)

    for i, blk in enumerate(self.blocks):
        xs = blk(xs)
    xs = self.norm(xs)
    out = self.invariant_proj(self.invariant(xs))
    
    class_tokens = out[:, 0]
    out = out[:, 1 + self.num_register_tokens :]
    registers = out[:, 1 : 1 + self.num_register_tokens]


    B, _, w, h = x.shape
    out = out.reshape(B, w // self.patch_size, h // self.patch_size, -1).contiguous()
       
    return class_tokens, registers, out 

model_dict = {
    "dinov2_vitl16": (vit_large, custom_fwd),
    "dinov2_hybrid_vitl16": (hybrid_vit_large, custom_fwd),
    "dinov2_vitl16_inv_early": (vit_large_inv_early, custom_fwd),
    "dinov2_vith16": (vit_huge, custom_fwd),
    "dinov2_vith16_inv_early": (vit_huge_inv_early, custom_fwd),
    "dinov2_hybrid_vith16": (hybrid_vit_huge, custom_fwd),
}

def __model_loader__(model_name: str, weights:str, device: str = "cuda") -> nn.Module:

    model_func, custom_fwd_func = model_dict[model_name]
    model: nn.Module = model_func()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(checkpoint["teacher"], strict=False)
    print(msg)

    model.forward = MethodType(custom_fwd_func, model)
    model.eval()
    model.to(device=device)
    return model