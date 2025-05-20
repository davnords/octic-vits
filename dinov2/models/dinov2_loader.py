# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from types import MethodType
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from dinov2.models.vision_transformer import vit_base, vit_large
from dinov2.utils.utils import chunk_block_weight, revert_block_chunk_weight, load_pretrained_weights
from dinov2.models import build_model_from_cfg
from omegaconf import OmegaConf

def custom_fwd(
    self: nn.Module,
    x: Float[Tensor, "b c h w"],
) -> tuple[
    Float[Tensor, "b d"],
    Float[Tensor, "b k d"] | None,
    Float[Tensor, "b ih iw d"],
]:
    bs, _, h, w = x.shape
    x = self.prepare_tokens_with_masks(x, None)
    for blk in self.blocks:
        x = blk(x)
    x_norm = self.norm(x)
    cls_tok = x_norm[:, 0]
    registers = x_norm[:, 1 : 1 + self.num_register_tokens]
    patches = x_norm[:, 1 + self.num_register_tokens :]
    patches = patches.reshape(bs, h // self.patch_size, w // self.patch_size, -1)
    return cls_tok, registers, patches


model_dict = {
    "dinov2_vitb16": ("vit_base", "pretrained_models/simdinov2/vitb16_reg4_DINOv2_ep100.pth"),
    "dinov2_vitl16": ("vit_large", "pretrained_models/simdinov2/vitl16_reg4_DINOv2_ep100.pth"),

    "simdinov2_vitb16": ("vit_base", "pretrained_models/simdinov2/vitb16_reg4_SimDNIOv2_ep100.pth"),
    "simdinov2_vitl16": ("vit_large", "pretrained_models/simdinov2/vitl16_reg4_SimDINOv2_100ep.pth"),
}

def __model_loader__(model_name: str, device: str = "cuda") -> nn.Module:
    cfg = OmegaConf.load('octo/models/simdinov2/configs/ssl_default_config.yaml')
    model_n, pretrained_weights = model_dict[model_name]
    cfg.student.arch = model_n
    cfg.student.block_chunks = 4
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    target_block_chunks = cfg.student.block_chunks
    load_pretrained_weights(model, pretrained_weights, ("model", "teacher"), target_block_chunks )
    model.forward = MethodType(custom_fwd, model)
    model.eval()
    model.to(device=device)
    return model

if __name__ == "__main__":
    model = __model_loader__("simdinov2_vitl16")
    x = torch.randn(1, 3, 224, 224).cuda()
    out = model(x)