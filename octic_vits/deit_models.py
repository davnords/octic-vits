from .model import OcticVisionTransformer
from timm.models import register_model
import torch
from deit.vit import Layer_scale_init_Block
from .d8_layers import Layer_scale_init_BlockD8

@register_model
def hybrid_deit_large_patch16(img_size=224, **kwargs):
    model = OcticVisionTransformer(
        img_size = img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        standard_block_layers=Layer_scale_init_Block,
        octic_block_layers=Layer_scale_init_BlockD8,
        **kwargs)
    return model


@register_model
def hybrid_deit_huge_patch14(img_size=224, **kwargs):
    model = OcticVisionTransformer(
        img_size = img_size,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        standard_block_layers=Layer_scale_init_Block,
        octic_block_layers=Layer_scale_init_BlockD8,
        **kwargs)
    return model

@register_model
def d8_inv_early_deit_huge_patch14(img_size=224, **kwargs):
    model = OcticVisionTransformer(
        img_size = img_size,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        invariant=True,
        standard_block_layers=Layer_scale_init_Block,
        octic_block_layers=Layer_scale_init_BlockD8,
        **kwargs)
    return model

@register_model
def d8_inv_early_deit_large_patch16(img_size=224, **kwargs):
    model = OcticVisionTransformer(
        img_size = img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        invariant=True,
        standard_block_layers=Layer_scale_init_Block,
        octic_block_layers=Layer_scale_init_BlockD8,
        **kwargs)
    return model
