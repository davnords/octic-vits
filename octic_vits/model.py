import torch.nn as nn
import torch
# Here we should create a truly generic model where D_8, I_8 and H_8 are all supported easily
# Maybe we will make one for dinov2 and one for deit iii (or maybe we can make one generic that just uses slightly different blocks)

from octic_vits.d8_layers import (
    LayerNormD8,
    PatchEmbedD8,
    TritonGeluD8,
    AttentionD8,
    MlpD8,
    Layer_scale_init_BlockD8,
    BlockD8
)

from octic_vits.d8_utils import SQRT2_OVER_2
from timm.layers import trunc_normal_
from octic_vits.d8_utils import convert_8tuple_to_5tuple, isotypic_dim_interpolation, interpolate_spatial_tuple, convert_5tuple_to_8tuple
import torch.nn.functional as F
from .d8_invariantization import PowerSpectrumInvariant
# from .standard_vit_components import Block, Attention, Mlp
from timm.models.vision_transformer import Block
from functools import partial

class OcticVisionTransformer(nn.Module):
    """ Octic Vision Transformer, modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        octic_block_layers=BlockD8,
        standard_block_layers=Block,
        Patch_layer=PatchEmbedD8,
        init_scale=1e-4,
        num_register_tokens=0,
        global_pool=False,
        invariant=False,
        octic_equi_break_layer=None, # None for breaking in the middle. -1 for breaking at the end
        **kwargs):
        super().__init__()  
        assert embed_dim % 8 == 0, "embed_dim must be divisible by 8"

        self.dropout_rate = drop_rate
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        if octic_equi_break_layer is None:
            assert depth % 2 == 0, "depth must be even"
            octic_equi_break_layer = depth // 2
        else:
            assert octic_equi_break_layer >= 0, "octic_equi_break_layer must be non-negative"
            assert octic_equi_break_layer < depth, "octic_equi_break_layer must be less than depth"
        self.octic_equi_break_layer = octic_equi_break_layer
        self.invariant = invariant
        self.num_register_tokens = num_register_tokens

        if self.invariant:
            self.invariantization = PowerSpectrumInvariant(embed_dim)
            self.invariant_proj = torch.nn.Linear(self.invariantization.output_dim, embed_dim)

        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if not global_pool:
            # Class token if not using global pooling
            self.cls_token = nn.ParameterList(
                [ # One dim irreps
                    nn.Parameter(torch.zeros(1, 1, embed_dim // 8), requires_grad=(i == 0)) for i in range(4)
                ] + 
                [ # Two dim irreps
                    nn.Parameter(torch.zeros(1, 1, 2, embed_dim // 4), requires_grad=False)
                ]
            )
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.empty(img_size//patch_size//2, img_size//patch_size//2, embed_dim//8)) for _ in range(6)])

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            octic_block_layers(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=LayerNormD8,
                act_layer=TritonGeluD8,
                init_values=init_scale,
            ) if i < self.octic_equi_break_layer else 
            standard_block_layers(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                init_values=init_scale,
            )
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        self.head = (
            nn.Linear(embed_dim, num_classes) 
            if num_classes > 0 else nn.Identity()
        )

        std = 8*.02 # Changed initialization from baseline that uses 0.02
        for p in self.pos_embed:
            trunc_normal_(p, std=SQRT2_OVER_2*std)
        if not global_pool:
            for p in self.cls_token:
                if p.requires_grad:
                    trunc_normal_(p, std=std)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_features(self, x):
        B, C_in, H, W = x.shape
        xs = self.patch_embed(x)

        pos_embed = convert_8tuple_to_5tuple(isotypic_dim_interpolation(self.pos_embed, dim=0))
        pos_embed = interpolate_spatial_tuple(xs, pos_embed, H, W, self.patch_embed.patch_size)
        xs = tuple(x+v.flatten(0,1) for x,v in zip(xs, pos_embed))

        if not self.global_pool:    
            # Append class token to the input
            cls_token = tuple(self.cls_token[i].expand(B, *self.cls_token[i].shape[1:]) for i in range(5))
            xs = tuple( torch.cat((cls_token[i], xs[i]), dim=1) for i in range(5))
                
        for blk in self.blocks[:self.octic_equi_break_layer]:
            xs = blk(xs)
        
        if self.invariant: 
            x = self.invariantization(xs)
            x = self.invariant_proj(x)
        else:
            x = torch.cat(convert_5tuple_to_8tuple(xs), dim=-1)
    
        for blk in self.blocks[self.octic_equi_break_layer:]:
            x = blk(x) 
        x = self.norm(x)

        if self.global_pool:
            # Global average pooling
            x = x.mean(dim=1) # for no cls_token
        else:
            # Pluck out the class token for classification
            x = x[:, 0] 
                
        return x

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(
                x,
                p=float(self.dropout_rate),
                training=self.training,
            )
        x = self.head(x)
        
        return x
    
    @torch.jit.ignore
    def no_weight_decay(self):
        base_names = ['pos_embed.0', 'pos_embed.1', 'pos_embed.2', 'pos_embed.3', 'pos_embed.4', 'pos_embed.5', 'cls_token.0',]
        no_weight_decay_params = set(base_names + [f'_orig_mod.{name}' for name in base_names])
        print('Ignoring weight decay for:', no_weight_decay_params)
        return no_weight_decay_params
