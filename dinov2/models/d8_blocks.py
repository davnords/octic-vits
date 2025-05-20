# BELOW IS ALL FOR EFFICIENT ATTENTION -> Waiting to implement it equivariantly

import torch
import os
import warnings
from typing import Callable, List, Any, Tuple, Dict, Union
from torch import nn, Tensor
from d8_components.d8_layers_5tuple import (
    LayerNormD8v2,
    AttentionD8,
    DropPathD8,
    MlpD8,
    LayerScaleD8,
    TritonGeluD8Five,
)
from typing import Literal

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat
        from xformers.ops import memory_efficient_attention, unbind
        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
        print("Nice, xFormers is ready to rumble!")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")

# --- Everything below is implemented for 5tuple because we believe it is the future ---- 

class MemEffAttentionD8(AttentionD8):
    # Based on DINOv2: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py#L72
    def forward(self, xs: Tuple[Tensor], attn_bias=None) -> Tuple[Tensor]:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(xs)

        B, N, C = xs[0].shape

        qkvs = self.qkv(xs)
        H = self.num_heads
        qkv = torch.cat(tuple(
            qkv_irrep.reshape(
                B, N, 3, H, C // H
            )
            for qkv_irrep in qkvs[:4]
        ) + (
            qkvs[4].reshape(
                B, N, 2, 3, H, 2 * C // H
            ).permute(0, 1, 3, 4, 2, 5).flatten(start_dim=-2),
        ), dim=-1).permute(2, 0, 1, 3, 4) # Different ordering than for F.scaled_dot_product_attention

        q, k, v = qkv[0], qkv[1], qkv[2]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        x_1d, x_2d = x.chunk(2, dim=-1)
        xs = tuple(
            x_irrep.transpose(1, 2).reshape(B, N, C)
            for x_irrep in x_1d.chunk(4, dim=-1)
        ) + (
            x_2d.reshape(B, H, N, 2, 2*C//H).permute(0, 2, 3, 1, 4).reshape(B, N, 2, 2*C),
        )

        xs = self.proj(xs)
        xs = self.proj_drop(xs)
        return xs


class Block(nn.Module):
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

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            xs = drop_add_residual_stochastic_depth(
                xs,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            xs = drop_add_residual_stochastic_depth(
                xs,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            # x = x + self.drop_path1(attn_residual_func(x))
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

# Not sure this is faster with all the tuple unpacking
# But this is part of DINOv2 architecture for speed, so we include it
def drop_add_residual_stochastic_depth(
    xs: Tuple[Tensor],
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tuple[Tensor]:
    
    # 1) extract subset using permutation
    b, n, d = xs[0].shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=xs[0].device))[:sample_subset_size]
    xs_subset = tuple(x[brange] for x in xs)

    # 2) apply residual_func to get residual
    residual = residual_func(xs_subset)

    xs_flat = tuple(x.flatten(1) for x in xs)
    residual = tuple(r.flatten(1) for r in residual)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    xs_plus_residual = tuple(torch.index_add(x_flat, 0, brange, r.to(dtype=xs[0].dtype), alpha=residual_scale_factor) for x_flat, r in zip(xs_flat, residual))
    return tuple(x_plus_residual.view_as(x) for x_plus_residual, x in zip(xs_plus_residual, xs))


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    print(f"x shape: {x.shape}, brange shape: {brange.shape}, residual shape: {residual.shape}")
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}

from typing import Sequence
def index_select_cat_torch(
    sources: Sequence[torch.Tensor], indices: Sequence[torch.Tensor]
) -> torch.Tensor:
    return torch.cat([s[i.long()].flatten() for s, i in zip(sources, indices)], dim=0)

def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [xs[0].shape[0] for xs in x_list]
    all_shapes = tuple((b, xs[0].shape[1]) for b, xs in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, xs in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(xs[0].shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias
    if branges is not None:
        # cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
        print(f"branges: {branges}")        
        cat_tensors = tuple(index_select_cat_torch([xs[i].flatten(1) for xs in x_list], branges).view(1, -1, x_list[0][i].shape[2:]) for i in range(5))
        print(f"cat_tensors: {cat_tensors[0].shape}")
        # cat_tensors = tuple(index_select_cat([xs[i].flatten(1) for xs in x_list], branges).view([1, -1, *x_list[0][i].shape[2:]]) for i in range(5))
    else:
        tensors_bs1s = tuple(tuple(xs[i].reshape([1, -1, *xs[i].shape[2:]]) for xs in x_list) for i in range(5))
        cat_tensors = tuple(torch.cat(tensors_bs1, dim=1) for tensors_bs1 in tensors_bs1s)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tuple[Tensor]],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vectors=None,
) -> Tuple[Tensor]:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(xs[0], sample_drop_ratio=sample_drop_ratio) for xs in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, xs_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = tuple(attn_bias.split(x) for x in residual_func(xs_cat, attn_bias=attn_bias))  # type: ignore
    # For some reason i decided it was easier to have a tuple of 5 lists instead of vice-versa, so now we have to get it
    # back to a list of tuples of 5 lol....
    residual_list = list(zip(*residual_list)) # reversing

    return residual_list

    outputs = []
    if scaling_vectors is None: 
        scaling_vectors = [None] * 5 # One for each tensor in the 5-tuple
    for xs, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(
            tuple(
                add_residual(x, brange, r, residual_scale_factor, scaling_vector).view_as(x) for x, r, scaling_vector in zip(xs, residual, scaling_vectors) 
            )
        )
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tuple[Tensor]]) -> List[Tuple[Tensor]]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttentionD8)

        if self.training and self.sample_drop_ratio > 0.0:
            warnings.warn("Stochastic depth currently does not pass equivariance tests, so don't use it in training before it is fixed")
            def attn_residual_func(xs: Tuple[Tensor], attn_bias=None) -> Tuple[Tensor]:
                return self.attn(self.norm1(xs), attn_bias=attn_bias)

            def ffn_residual_func(xs: Tuple[Tensor], attn_bias=None) -> Tuple[Tensor]:
                return self.mlp(self.norm2(xs))

            assert not isinstance(self.ls1, LayerScaleD8), 'LayerScaleD8 is not supported with drop path (yet), just a skill issue'
            # NOTE: This part of the code execution requires Ampere+ compute capabilities for Triton support for xformers

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vectors=[self.ls1.alpha_A1, self.ls1.alpha_A2, self.ls1.alpha_B1, self.ls1.alpha_B2, self.ls1.alpha_E] if isinstance(self.ls1, LayerScaleD8) else None,
            )
            return x_list
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vectors=[self.ls1.alpha_A1, self.ls1.alpha_A2, self.ls1.alpha_B1, self.ls1.alpha_B2, self.ls1.alpha_E] if isinstance(self.ls1, LayerScaleD8) else None,
            )
            return x_list
        else:

            def attn_residual_func(xs: Tuple[Tensor], attn_bias=None) -> Tuple[Tensor]:
                return self.ls1(self.attn(self.norm1(xs), attn_bias=attn_bias))

            def ffn_residual_func(xs: Tuple[Tensor], attn_bias=None) -> Tuple[Tensor]:
                return self.ls2(self.mlp(self.norm2(xs)))

            attn_bias, xs = get_attn_bias_and_cat(x_list)
            residual = attn_residual_func(xs, attn_bias=attn_bias)
            xs = tuple(x + r for x, r in zip(xs, residual))
            residual = ffn_residual_func(xs)
            xs = tuple(x + r for x, r in zip(xs, residual))

            out = tuple(attn_bias.split(x) for x in xs)
            # For some reason i decided it was easier to have a tuple of 5 lists instead of vice-versa, so now we have to get it
            # back to a list of tuples of 5 lol....
            out = list(zip(*out))
            return out

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# C = 512
# B = 4
# xs_list = [tuple(torch.randn(B, 196, C//8, device=device) for _ in range(4)) + (torch.randn(B, 196, 2, C//4, device=device),) for _ in range(2)]

# net = NestedTensorBlock(C, num_heads=8, attn_class=MemEffAttentionD8, drop_path=0.3).cuda()
# with torch.inference_mode():
#     out = net(xs_list)
#     out = net(out)