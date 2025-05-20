# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from dinov2.models.d8_blocks_simple import NestedTensorBlockD8 as BlockD8
from d8_components.d8_utils import SQRT2_OVER_2, convert_5tuple_to_8tuple, convert_8tuple_to_5tuple
from d8_components.d8_layers import isotypic_dim_interpolation, interpolate_spatial_tuple


from d8_components.d8_layers_5tuple import (
    LayerNormD8v2,
    AttentionD8,
    DropPathD8,
    MlpD8,
    LayerScaleD8,
    TritonGeluD8Five,
    PatchEmbedD8
)


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class HybridDinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=partial(PatchEmbedD8, strict_img_size=False),
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,

        # Specific args to the equi part
        equi_embed_dim=None,
        equi_mlp_ratio=None,
        equi_num_heads=None,

        invariant=None,

        # To work with timm
        **kwargs,

    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        initial_embed_dim = embed_dim if equi_embed_dim is None else equi_embed_dim
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=initial_embed_dim)
        num_patches = self.patch_embed.num_patches
        self.depth = depth
        
        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, initial_embed_dim // 8), requires_grad=(i == 0))for i in range(8)])
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.empty(img_size//patch_size//2, img_size//patch_size//2, initial_embed_dim//8)) for _ in range(6)])
        assert num_register_tokens >= 0
        self.register_tokens = (
           nn.ParameterList([nn.Parameter(torch.zeros(1, self.num_register_tokens, initial_embed_dim // 8), requires_grad=(i == 0))for i in range(8)]) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            print("using MLP layer as FFN")
            ffn_layer = Mlp
        else:
            raise NotImplementedError

        assert depth % 2 == 0, "depth should be even!"

        blocks_list = [
            BlockD8(
                dim=initial_embed_dim,
                num_heads=num_heads if equi_num_heads is None else equi_num_heads,
                mlp_ratio=mlp_ratio if equi_mlp_ratio is None else equi_mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=LayerNormD8v2,
                act_layer=TritonGeluD8Five,
                ffn_layer=MlpD8,
                init_values=init_values,
            ) if i < depth//2 else

            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            assert (self.depth//2) % block_chunks == 0, f"depth {self.depth//2} should be divisible by block_chunks {block_chunks}"
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            print('Block chunksize:', chunksize)
            print('Block chunks: ', block_chunks)
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.proj = nn.Linear(equi_embed_dim, embed_dim) if equi_embed_dim is not None else nn.Identity()

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()


        self.invariant = invariant(initial_embed_dim) if invariant is not None else None
        if self.invariant is not None:
            self.invariant_proj = nn.Linear(self.invariant.output_dim, embed_dim)

        # self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.mask_token = nn.ParameterList([nn.Parameter(torch.zeros(1, initial_embed_dim // 8), requires_grad=(i == 0))for i in range(8)])

        self.init_weights()

    def init_weights(self):
        std = 8*0.02  

        # trunc_normal_(self.pos_embed, std=0.02)
        for p in list(self.pos_embed): trunc_normal_(p, std=std*SQRT2_OVER_2) 
        nn.init.normal_(self.cls_token[0], std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens[0], std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, h, w = x.shape # changing height / width order from dinov2
        xs = convert_5tuple_to_8tuple(self.patch_embed(x))
        if masks is not None:
            xs = tuple(torch.where(masks.unsqueeze(-1), self.mask_token[i].to(xs[i].dtype).unsqueeze(0), xs[i]) for i in range(8))
        
        pos_embed = isotypic_dim_interpolation(self.pos_embed, dim=0)
        pos_embed = interpolate_spatial_tuple(xs, pos_embed, h, w, self.patch_size)
        xs = tuple(x+v.flatten(0,1) for x,v in zip(xs, pos_embed))

        # Add cls token after pos_embed, deviating from DINOv2
        xs = tuple(torch.cat((self.cls_token[i].expand(xs[i].shape[0], -1, -1), xs[i]), dim=1) for i in range(8))

        if self.register_tokens is not None:
            xs = tuple(torch.cat(
                (
                    xs[i][:, :1],
                    self.register_tokens[i].expand(xs[i].shape[0], -1, -1),
                    xs[i][:, 1:],
                ),
                dim=1,
            ) for i in range(8))

        return convert_8tuple_to_5tuple(xs)

    def forward_features_list(self, x_list, masks_list):
        xs = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]

        for blk in self.blocks[:self.depth//2]:
            xs = blk(xs)

        if self.invariant is not None:
            # Inv-early
            x = [self.invariant(xi) for xi in xs]
            x = [self.invariant_proj(xi) for xi in x]
        else:
            # Hybrid
            x = [torch.cat(convert_5tuple_to_8tuple(x), dim=-1) for x in xs]
            # Proj if needed
            x = [self.proj(xi) for xi in x]

        for blk in self.blocks[self.depth//2:]:
            x = blk(x) 

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        xs = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks[:self.depth//2]:
            xs = blk(xs)

        if self.invariant is not None:
            # Inv-early
            x = self.invariant(xs)
            x = self.invariant_proj(x)
        else:
            # Hybrid
            xs = convert_5tuple_to_8tuple(xs)
            x = torch.cat(xs, dim=-1)
            x = self.proj(x)

        for blk in self.blocks[self.depth//2:]:
            x = blk(x) 

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        xs = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        
        assert all(i > self.depth//2 for i in blocks_to_take), f"All block indices must be > half depth, got {blocks_to_take}"

        i = 0 # Ugly tracker to kee ptrack of the block index
        for blk in self.blocks[:self.depth//2]:
            xs = blk(xs)
            i += 1
        
        if self.invariant is not None:
            # Inv-early
            x = self.invariant(xs)
            x = self.invariant_proj(x)
        else:
            xs = convert_5tuple_to_8tuple(xs)
            x = torch.cat(xs, dim=-1)
            x = self.proj(x)
        
        for blk in self.blocks[self.depth//2:]:
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
            i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            raise NotImplementedError("Chunked blocks not supported yet")
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
        
    @torch.jit.ignore
    def no_weight_decay(self):
        base_names = [
            'pos_embed.0', 'pos_embed.1', 'pos_embed.2', 'pos_embed.3', 'pos_embed.4', 'pos_embed.5', 
            'cls_token.0',
            'invariant.references', 'invariant.rotation_action', 'invariant.reflection_action'
        ]
        # Adding so it works for compile also
        no_weight_decay_params = set(base_names + [f'_orig_mod.{name}' for name in base_names])
        print('Ignoring weight decay for:', no_weight_decay_params)
        return no_weight_decay_params


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def hybrid_vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = HybridDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def hybrid_vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = HybridDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
    

def hybrid_vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = HybridDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

# model = hybrid_vit_large_wide(block_chunks=0).cuda()
# print('Parameters:', sum(p.numel() for p in model.parameters()))
# with torch.no_grad():
#     imgs = torch.randn(2, 3, 224, 224).cuda()
#     model(imgs)

# from d8_components.d8_invarization import PowerSpectrumInvariant

# model = hybrid_vit_base(block_chunks=0, invariant=PowerSpectrumInvariant).cuda()
# imgs = torch.randn(2, 3, 224, 224).cuda()
# model(imgs)