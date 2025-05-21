# Code written for the paper "Stronger ViTs With Octic Equivariance" (https://arxiv.org/abs/TBD)
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .model import OcticVisionTransformer
import torch.nn as nn
import torch
from typing import Sequence, Tuple, Union, Callable
from .d8_utils import SQRT2_OVER_2, convert_8tuple_to_5tuple, isotypic_dim_interpolation, interpolate_spatial_tuple, convert_5tuple_to_8tuple
from timm.layers import trunc_normal_
from dinov2.layers import MemEffAttention, NestedTensorBlock as Block
from .d8_layers import NestedTensorBlockD8 as BlockD8
from functools import partial
from timm.models import register_model


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

class OcticDinoVisionTransformer(OcticVisionTransformer):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth:int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        octic_block_layers: Callable = BlockD8,
        standard_block_layers: Callable = partial(Block, attn_class=MemEffAttention),
        invariant: bool = False,
        **kwargs,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            drop_path_rate (float): stochastic depth rate
            octic_block_layers (nn.Module): transformer block class for octic equivariant layers
            standard_block_layers (nn.Module): transformer block class for standard layers
            invariant (bool): use invariantization or not
        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            octic_block_layers=octic_block_layers,
            standard_block_layers=standard_block_layers,
            drop_path_rate=drop_path_rate,
            invariant=invariant,
            qkv_bias=True, ffn_bias=True, proj_bias=True,
        )
        self.depth = depth
        
        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim // 8), requires_grad=(i == 0))for i in range(8)])
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.empty(img_size//patch_size//2, img_size//patch_size//2, embed_dim//8)) for _ in range(6)])
        assert num_register_tokens >= 0
        self.register_tokens = (
           nn.ParameterList([nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim // 8), requires_grad=(i == 0))for i in range(8)]) if num_register_tokens else None
        )

        assert depth % 2 == 0, "depth should be even!"

        self.chunked_blocks = False
        self.mask_token = nn.ParameterList([nn.Parameter(torch.zeros(1, embed_dim // 8), requires_grad=(i == 0))for i in range(8)])
        self.head = nn.Identity()


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
        pos_embed = interpolate_spatial_tuple(xs, pos_embed, h, w, self.patch_embed.patch_size)
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

        if self.invariant:
            # Inv-early
            x = [self.invariantization(xi) for xi in xs]
            x = [self.invariant_proj(xi) for xi in x]
        else:
            # Hybrid
            x = [torch.cat(convert_5tuple_to_8tuple(x), dim=-1) for x in xs]

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

        if self.invariant:
            # Inv-early
            x = self.invariantization(xs)
            x = self.invariant_proj(x)
        else:
            # Hybrid
            xs = convert_5tuple_to_8tuple(xs)
            x = torch.cat(xs, dim=-1)

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
        
        if self.invariant:
            # Inv-early
            x = self.invariantization(xs)
            x = self.invariant_proj(x)
        else:
            xs = convert_5tuple_to_8tuple(xs)
            x = torch.cat(xs, dim=-1)
        
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
                out.reshape(B, w // self.patch_embed.patch_size, h // self.patch_embed.patch_size, -1).permute(0, 3, 1, 2).contiguous()
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
        base_names = ['pos_embed.0', 'pos_embed.1', 'pos_embed.2', 'pos_embed.3', 'pos_embed.4', 'pos_embed.5', 'cls_token.0',]
        no_weight_decay_params = set(base_names + [f'_orig_mod.{name}' for name in base_names])
        print('Ignoring weight decay for:', no_weight_decay_params)
        return no_weight_decay_params

@register_model
def hybrid_dinov2_vit_large_patch16(patch_size=16, num_register_tokens=0, **kwargs):
    model = OcticDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        standard_block_layers=partial(Block, attn_class=MemEffAttention, init_values=1.0e-05),
        octic_block_layers=partial(BlockD8, init_values=1.0e-05),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

@register_model
def hybrid_dinov2_vit_huge_patch16(patch_size=16, num_register_tokens=0, **kwargs):
    model = OcticDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        standard_block_layers=partial(Block, attn_class=MemEffAttention, init_values=1.0e-05),
        octic_block_layers=partial(BlockD8, init_values=1.0e-05),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

@register_model
def d8_inv_early_dinov2_vit_large_patch16(patch_size=16, num_register_tokens=0, **kwargs):
    model = OcticDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        invariant=True,
        standard_block_layers=partial(Block, attn_class=MemEffAttention, init_values=1.0e-05),
        octic_block_layers=partial(BlockD8, init_values=1.0e-05),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

@register_model
def d8_inv_early_dinov2_vit_huge_patch16(patch_size=16, num_register_tokens=0, **kwargs):
    model = OcticDinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        invariant=True,
        standard_block_layers=partial(Block, attn_class=MemEffAttention, init_values=1.0e-05),
        octic_block_layers=partial(BlockD8, init_values=1.0e-05),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model