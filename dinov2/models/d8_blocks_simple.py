# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from d8_components.d8_layers_5tuple import (
    LayerNormD8v2,
    AttentionD8,
    DropPathD8,
    MlpD8,
    LayerScaleD8,
    TritonGeluD8Five,
)

class BlockD8(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = TritonGeluD8Five,
        norm_layer: Callable[..., nn.Module] = LayerNormD8v2,
        attn_class: Callable[..., nn.Module] = AttentionD8,
        ffn_layer: Callable[..., nn.Module] = MlpD8,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScaleD8(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPathD8(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScaleD8(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPathD8(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, xs: Tuple[Tensor]) -> Tuple[Tensor]:
        def attn_residual_func(xs: Tuple[Tensor]) -> Tuple[Tensor]:
            return self.ls1(self.attn(self.norm1(xs)))

        def ffn_residual_func(xs: Tuple[Tensor]) -> Tuple[Tensor]:
            return self.ls2(self.mlp(self.norm2(xs)))

        if self.training and self.sample_drop_ratio > 0.0:
            residual = self.drop_path1(attn_residual_func(xs))
            xs = tuple(x + r for x, r in zip(xs, residual))
            residual = self.drop_path2(ffn_residual_func(xs))
            xs = tuple(x + r for x, r in zip(xs, residual))
        else:
            residual = attn_residual_func(xs)
            xs = tuple(x + r for x, r in zip(xs, residual))
            residual = ffn_residual_func(xs)
            xs = tuple(x + r for x, r in zip(xs, residual))
        return xs

class NestedTensorBlockD8(BlockD8):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        return [super(NestedTensorBlockD8, self).forward(x) for x in x_list]

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, tuple):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            return self.forward_nested(x_or_x_list)
        else:
            print(f"Unsupported type: {type(x_or_x_list)}")
            raise AssertionError

# batch_size = 2
# embed_dim = 1024
# device = "cuda:0"
# x = torch.randn([batch_size, 196, embed_dim], device=device)
# xs_D8 = tuple(t.clone().contiguous() for t in x.chunk(8, dim=-1))
# xs= (
#     xs_D8[0].clone().contiguous(),
#     xs_D8[1].clone().contiguous(),
#     xs_D8[2].clone().contiguous(),
#     xs_D8[3].clone().contiguous(),
#     torch.cat((
#         torch.stack(xs_D8[4:6], dim=-2),
#         torch.stack(xs_D8[6:], dim=-2),
#     ), dim=-1).clone().contiguous()
# )

# block = NestedTensorBlockD8(embed_dim, 8)
# block.to(device)
# block.train()
# out = block([xs, xs])

# print(out[0][0].shape)
