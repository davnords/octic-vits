import torch
import torch.nn.functional as F
import math
from itertools import repeat
import collections.abc
from typing import Optional, Literal, Tuple, Union, Callable, List
import timm
from torch import nn, Tensor

from .d8_gelu import TritonGeluD8

from .d8_utils import (
    SQRT2,
    SQRT2_OVER_2,
    SQRT2_OVER_4,
    regular_to_isotypic_D8,
    isotypic_to_regular_D8,
)

# From timm
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

###
# D8 is the group of mirroring and 90 degree rotations.
# D8 features are decomposed into 5 different irreps called A1, A2, B1, B2 and E.
# Detailed rep theory of D8 can be found at 
# https://groupprops.subwiki.org/wiki/Linear_representation_theory_of_dihedral_group:D8
# 
# A1: 1-dim irrep, invariant under all g in D8.
# A2: 1-dim irrep, invariant under rotation, -1 under mirroring.
# B1: 1-dim irrep, -1 under rotation, invariant under mirroring.
# B2: 1-dim irrep, -1 under rotation, -1 under mirroring.
#  E: 2-dim irrep, {{0, -1}, {1, 0}} under rotation, {{-1, 0}, {0, 1}} under mirroring.
#
# The regular representation of D8 is an 8-dimensional representation
# consisting of permutation matrices. In index permutation notation we have:
# Rotation given by  [0, 1, 2, 3, 4, 5, 6, 7] -> [3, 0, 2, 1, 5, 6, 7, 4].
# Mirroring given by [0, 1, 2, 3, 4, 5, 6, 7] -> [4, 5, 6, 7, 0, 1, 2, 3].
#
# The regular representation of D8 decomposes into one copy each of A1, A2, B1, B2
# and two copies of E (totalling 8 dimensions).
# The two copies of E can be thought of as a 2x2 matrix, with each column
# transforming according to the matrices given earlier.
# Schur's lemma says that linear layers from the regular representation to itself
# map each irrep to itself, giving one parameter for each of the one-dimensional
# representations and four parameters for a 2x2 matrix mapping the two copies of E
# to two copies of E.
#
# We keep features in a 8-tuple xs that is indexed as
# xs[0]: A1-features, typically of shape [B, N, C]
#   where B is batch size, N is the number of patches
#   and C is 1/8 of the feature embedding dimension.
# xs[1]: A2-features, typically of shape [B, N, C]
# xs[2]: B1-features, typically of shape [B, N, C]
# xs[3]: B2-features, typically of shape [B, N, C]
# We think of the E features as a 2x2 matrix with each column being
# an E feature.
# A first E feature is given by
# xs[4]: Upper left  E-features, typically of shape [B, N, C]
# and
# xs[5]: Lower left  E-features, typically of shape [B, N, C].
# A second E feature is given by
# xs[6]: Upper right E-features, typically of shape [B, N, C]
# and
# xs[7]: Lower right E-features, typically of shape [B, N, C].
#
###

class DropoutD8(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.dropout = nn.Dropout(p=p, inplace=inplace)

    def forward(self, xs):
        return (
            self.dropout(xs[0]),
            self.dropout(xs[1]),
            self.dropout(xs[2]),
            self.dropout(xs[3]),
            self.dropout(xs[4]),
        )

class GeluD8(nn.Module):
    def forward(self, xs):
        return regular_to_isotypic_D8(
            [F.gelu(x) for x in isotypic_to_regular_D8(xs)]
        )

class LinearD8(nn.Module):
    def __init__(self, input_channels, output_channels, bias=True):
        super().__init__()
        if input_channels % 8 != 0 or output_channels % 8 != 0:
            raise ValueError()
        # TODO: proper initialization of these weights according to
        # kaiming init or whatever.
        # x_A1, x_A2, x_B1, x_B2: [B, N, C//8]
        # x_2d, [B, N, 2, C//4]
        self.bias = bias
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.lin_A1 = nn.Linear(input_channels//8, output_channels//8, bias=bias)
        self.lin_A2 = nn.Linear(input_channels//8, output_channels//8, bias=False)
        self.lin_B1 = nn.Linear(input_channels//8, output_channels//8, bias=False)
        self.lin_B2 = nn.Linear(input_channels//8, output_channels//8, bias=False)
        # QUESTION: Do you need a SQRT2_OVER_2 or some factor here?
        self.lin_E = nn.Linear(input_channels//4, output_channels//4, bias=False)

    def forward(self, x_batched):
        assert len(x_batched) == 5, "Input should be a 5-tuple"
        x_A1, x_A2, x_B1, x_B2, x_2d = x_batched
        return self.lin_A1(x_A1), self.lin_A2(x_A2), self.lin_B1(x_B1), self.lin_B2(x_B2), self.lin_E(x_2d)

    def extra_repr(self) -> str:
        return f"in_features={self.input_channels}, out_features={self.output_channels}, bias={self.bias is not None}"

class AffineD8(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        if dim % 8 != 0:
            raise ValueError()
        self.alpha_A1 = nn.Parameter(torch.ones(dim//8))
        self.alpha_A2 = nn.Parameter(torch.ones(dim//8))
        self.alpha_B1 = nn.Parameter(torch.ones(dim//8))
        self.alpha_B2 = nn.Parameter(torch.ones(dim//8))
        
        self.alpha_E = nn.Parameter(torch.ones(dim//4))
        self.beta = None
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim//8))

    def forward(self, xs):
        if self.beta is not None:
            y_A1 = self.alpha_A1 * xs[0] + self.beta
        else:
            y_A1 = self.alpha_A1 * xs[0]
        return (
            y_A1,
            self.alpha_A2 * xs[1],
            self.alpha_B1 * xs[2],
            self.alpha_B2 * xs[3],
            self.alpha_E * xs[4]
        )


class LayerNormD8v2(nn.Module):
    def __init__(self, channels, eps=1e-05, elementwise_affine=True, bias=True):
        super().__init__()
        self.scaling = AffineD8(channels, bias=bias) if elementwise_affine else nn.Identity()
        self.eps = eps
    def forward(self, xs):
        std = SQRT2_OVER_4 * torch.sqrt(
            xs[0].var(dim=-1, unbiased=False, keepdim=True)
            + xs[1].var(dim=-1, unbiased=False, keepdim=True)
            + xs[2].var(dim=-1, unbiased=False, keepdim=True)
            + xs[3].var(dim=-1, unbiased=False, keepdim=True)

            # TODO: Verify this is correct :D
            + torch.mean(xs[4].var(dim=-1, unbiased=False, keepdim=True), dim=-2)
            + self.eps
        )
        xs = (
            (xs[0] - xs[0].mean(dim=-1, keepdim=True)) / std,
            (xs[1] - xs[1].mean(dim=-1, keepdim=True)) / std,
            (xs[2] - xs[2].mean(dim=-1, keepdim=True)) / std,
            (xs[3] - xs[3].mean(dim=-1, keepdim=True)) / std,
            
            # TODO: Verify this is correct :D
            (xs[4] - xs[4].mean(dim=-1, keepdim=True)) / std.unsqueeze(-1),
        )
        return self.scaling(xs)


class LayerScaleD8(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
    ) -> None:
        super().__init__()
        if dim % 8 != 0:
            raise ValueError()
        self.alpha_A1 = nn.Parameter(init_values*torch.ones(dim//8))
        self.alpha_A2 = nn.Parameter(init_values*torch.ones(dim//8))
        self.alpha_B1 = nn.Parameter(init_values*torch.ones(dim//8))
        self.alpha_B2 = nn.Parameter(init_values*torch.ones(dim//8))

        self.alpha_E = nn.Parameter(init_values*torch.ones(dim//4))

    def forward(self, xs):
        return (
            self.alpha_A1 * xs[0],
            self.alpha_A2 * xs[1],
            self.alpha_B1 * xs[2],
            self.alpha_B2 * xs[3],
            self.alpha_E * xs[4],
        )


class MlpD8(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=TritonGeluD8,
        norm_layer=None,
        bias=True,
        drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = LinearD8

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = DropoutD8(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = DropoutD8(drop_probs[1])

    def forward(self, xs):
        xs = self.fc1(xs)
        xs = self.act(xs)
        xs = self.drop1(xs)
        xs = self.norm(xs)
        xs = self.fc2(xs)
        xs = self.drop2(xs)
        return xs

def drop_path_d8(xs,
               drop_prob: float = 0.,
               training: bool = False,
               scale_by_keep: bool = True):
    """ Modified from timm """
    if drop_prob == 0. or not training:
        return xs
    x0 = xs[0]
    keep_prob = 1 - drop_prob
    shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x0.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    
    return (
        x0 * random_tensor,
        xs[1] * random_tensor,
        xs[2] * random_tensor,
        xs[3] * random_tensor,

        # TODO: Verify this is correct :D, this one looks OK I think
        xs[4] * random_tensor.unsqueeze(-1),
    )

class DropPathD8(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, xs):
        return drop_path_d8(xs, self.drop_prob, self.training, self.scale_by_keep)

class LiftIrrepD8Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bias,
                 irrep="A1",
                ):
        super().__init__()
        if irrep not in ["A1", "A2", "B1", "B2", "E"]:
            raise ValueError("Invalid irrep.")
        if bias and not (irrep == "A1"):
            raise ValueError("Bias only ok for A1-irrep.")
        
        kernel_size = to_2tuple(kernel_size)
        if kernel_size[0] != kernel_size[1]:
            raise NotImplementedError("Non-square kernels not implemented")
        if kernel_size[0] % 2 != 0 or kernel_size[1] % 2 != 0:
            raise NotImplementedError("Odd kernel sizes not yet implemented")
        if (kernel_size[0] == 2 or kernel_size[1] == 2) and irrep in ["A2", "B1"]:
            raise ValueError(f"No {irrep} irrep in filter kernels of size 2.")
        self.kernel_size = kernel_size
        self.stride = stride
        self.irrep = irrep
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        # TODO: could remove over-parameterization and do lookup of parameters instead of 
        # averaging mirrored filters in forward.
        # Currently all except E-irrep are overparametrized here.
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0] // 2, kernel_size[1] // 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def expand_weight(self):
        if self.irrep == "E":
            w = 0.5 * self.weight
            w2 = torch.cat([w, w.flip(-2)], dim=-2)
            return torch.cat([w2, -w2.flip(-1)], dim=-1)
        w = SQRT2_OVER_4 * self.weight
        w_rot = w.rot90(k=1, dims=(-2, -1))
        w_rot2 = w.rot90(k=2, dims=(-2, -1))
        w_rot3 = w.rot90(k=3, dims=(-2, -1))
        if self.irrep == "A1":
            weight = torch.cat(
                [
                    torch.cat([w, w_rot], dim=-2),
                    torch.cat([w_rot3, w_rot2], dim=-2),
                ],
                dim=-1,
            )
            return (weight + weight.flip(-1))
        elif self.irrep == "A2":
            weight = torch.cat(
                [
                    torch.cat([w, w_rot], dim=-2),
                    torch.cat([w_rot3, w_rot2], dim=-2),
                ],
                dim=-1,
            )
            return (weight - weight.flip(-1))
        elif self.irrep == "B1":
            weight = torch.cat(
                [
                    torch.cat([w, -w_rot], dim=-2),
                    torch.cat([-w_rot3, w_rot2], dim=-2),
                ],
                dim=-1,
            )
            return (weight + weight.flip(-1))
        elif self.irrep == "B2":
            weight = torch.cat(
                [
                    torch.cat([w, -w_rot], dim=-2),
                    torch.cat([-w_rot3, w_rot2], dim=-2),
                ],
                dim=-1,
            )
            return (weight - weight.flip(-1))

    def forward(self, x):
        weight = self.expand_weight()
        if self.irrep == "E":
            return (
                F.conv2d(x, weight, self.bias, stride=self.stride),
                F.conv2d(x, weight.rot90(k=1, dims=(-2, -1)), self.bias, stride=self.stride)
            )
        return F.conv2d(x, weight, self.bias, stride=self.stride)

class LiftD8(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bias,
                ):
        super().__init__()
        if out_channels % 8 != 0:
            raise ValueError()
        outs = out_channels // 8
        self.conv_A1 = LiftIrrepD8Conv2d(in_channels, outs, kernel_size, stride, bias=bias, irrep="A1")
        self.conv_A2 = LiftIrrepD8Conv2d(in_channels, outs, kernel_size, stride, bias=False, irrep="A2")
        self.conv_B1 = LiftIrrepD8Conv2d(in_channels, outs, kernel_size, stride, bias=False, irrep="B1")
        self.conv_B2 = LiftIrrepD8Conv2d(in_channels, outs, kernel_size, stride, bias=False, irrep="B2")
        self.conv_E_left = LiftIrrepD8Conv2d(in_channels, outs, kernel_size, stride, bias=False, irrep="E")
        self.conv_E_right = LiftIrrepD8Conv2d(in_channels, outs, kernel_size, stride, bias=False, irrep="E")

    def forward(self, img):
        return (
            self.conv_A1(img),
            self.conv_A2(img),
            self.conv_B1(img),
            self.conv_B2(img),
            *self.conv_E_left(img),
            *self.conv_E_right(img)
        )

class PatchEmbedD8(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True,
                 bias=True,
                 strict_img_size=True,
                ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if embed_dim % 8 != 0:
            raise ValueError()

        self.flatten = flatten
        self.strict_img_size = strict_img_size

        self.lift8 = LiftD8(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                timm.layers.trace_utils._assert(
                    H == self.img_size[0],
                    f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                )
                timm.layers.trace_utils._assert(
                    W == self.img_size[1],
                    f"Input width ({W}) doesn't match model ({self.img_size[1]})."
                )
            else:
                patch_W, patch_H = self.patch_size
                assert H % (patch_H*2) == 0, f"Input image height {H} is not an even multiple of patch height {patch_H}"
                assert W % (patch_W*2) == 0, f"Input image width {W} is not an even multiple of patch width: {patch_W}"

        xs = self.lift8(x)
        if self.flatten:
            xs = tuple(
                x.flatten(2).transpose(1, 2)  # BCHW -> BNC
                for x in xs
            )
        xs = (
            xs[0],
            xs[1],
            xs[2],
            xs[3],
            torch.cat((
                torch.stack(xs[4:6], dim=-2),
                torch.stack(xs[6:], dim=-2),
            ), dim=-1)
        )
        xs = self.norm(xs)
        return xs
    
    def _init_weights(self):
        for w in [
            self.lift8.conv_A1.weight.data,
            self.lift8.conv_A2.weight.data,
            self.lift8.conv_B1.weight.data,
            self.lift8.conv_B2.weight.data,
            self.lift8.conv_E_left.weight.data,
            self.lift8.conv_E_right.weight.data,
        ]:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))  # TODO: check this initialization

class IsotypicToPatchD8(nn.Module):
    def __init__(
        self, dim, patch_side, out_channels=3, bias=True, reshape_to_image=False,
    ):
        super().__init__()
        if patch_side % 2 != 0:
            raise NotImplementedError("Odd patch side not implemented.")
        self.dim = dim
        self.patch_side = patch_side
        self.out_channels = out_channels
        self.reshape_to_image = reshape_to_image

        # TODO: the below is an overparametrization,
        # we could use some indexing logic instead of
        # averaging over mirroring in forward.
        # But that is tedious due to triangular shape that defines
        # output of A1, A2, B1, B2 types,
        # and the slight increase in parameters should not matter
        # too much here.
        self.lin8 = LinearD8(dim, 2*(patch_side**2 * out_channels), bias=bias)

    def forward(self, xs):
        B, L, C = xs[0].shape
        xs = tuple(
            0.25 * x_irrep.reshape(
                B, L, self.patch_side//2, self.patch_side//2, self.out_channels)
            for x_irrep in self.lin8(xs)
        )

        out_A1 = torch.cat((
            torch.cat((xs[0],
                       xs[0].rot90(k=1, dims=(2, 3))), dim=2),
            torch.cat((xs[0].rot90(k=3, dims=(2, 3)),
                       xs[0].rot90(k=2, dims=(2, 3))), dim=2)
        ), dim=3)
        out = (out_A1 + out_A1.flip(3))

        out_A2 = torch.cat((
            torch.cat((xs[1],
                       xs[1].rot90(k=1, dims=(2, 3))), dim=2),
            torch.cat((xs[1].rot90(k=3, dims=(2, 3)),
                       xs[1].rot90(k=2, dims=(2, 3))), dim=2)
        ), dim=3)
        out = out + (out_A2 - out_A2.flip(3))

        out_B1 = torch.cat((
            torch.cat((xs[2],
                       -xs[2].rot90(k=1, dims=(2, 3))), dim=2),
            torch.cat((-xs[2].rot90(k=3, dims=(2, 3)),
                       xs[2].rot90(k=2, dims=(2, 3))), dim=2)
        ), dim=3)
        out = out + (out_B1 + out_B1.flip(3))

        out_B2 = torch.cat((
            torch.cat((xs[3],
                       -xs[3].rot90(k=1, dims=(2, 3))), dim=2),
            torch.cat((-xs[3].rot90(k=3, dims=(2, 3)),
                       xs[3].rot90(k=2, dims=(2, 3))), dim=2)
        ), dim=3)
        out = out + (out_B2 - out_B2.flip(3))

        # x_E1 = (xs[4] + xs[6])  # xs[6] contains pos enc directly through res blocks
        x_E1 = SQRT2 * xs[4]
        out_E1 = torch.cat((
            torch.cat((x_E1,
                       x_E1.flip(2)), dim=2),
            torch.cat((-x_E1.flip(3),
                       -x_E1.rot90(k=2, dims=(2, 3))), dim=2),
        ), dim=3)
        out = out + out_E1

        # x_E2 = (xs[5] + xs[7])  # xs[7] contains pos enc directly through res blocks
        x_E2 = SQRT2 * xs[5]
        out_E2 = torch.cat((
            torch.cat((x_E2,
                       x_E2.flip(2)), dim=2),
            torch.cat((-x_E2.flip(3),
                       -x_E2.rot90(k=2, dims=(2, 3))), dim=2),
        ), dim=3).rot90(k=1, dims=(2, 3))
        out = out + out_E2

        if self.reshape_to_image:
            H = W = int(math.sqrt(L))
            out = out.reshape(
                B, H, W, self.patch_side, self.patch_side, self.out_channels)
            out = out.permute(0, 5, 1, 3, 2, 4).reshape(  # B, C_out, H, p, W, p
                B, self.out_channels, H*self.patch_side, W*self.patch_side)
        else:
            out = out.reshape(B, L, self.patch_side**2 * self.out_channels)
        return out

class AttentionD8(nn.Module):
    # modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 rope=None,
                 qk_scale=None,
                ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert (dim//num_heads)%8 == 0, "dim should be divisible by 8" 

        if rope is not None:
            raise NotImplementedError("RoPE not implemented")
        if (dim // num_heads) % 8 != 0:
            raise ValueError()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = LinearD8(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearD8(dim, dim, bias=proj_bias)
        self.proj_drop = DropoutD8(proj_drop)
        self.rope = rope

        self.att = F.scaled_dot_product_attention

    def forward(self, xs):
        B, N, C = xs[0].shape

        qkvs = self.qkv(xs)
        # https://github.com/huggingface/pytorch-image-models/blob/cb4cea561a3f39bcd6a3105c72d7e0b2b928bf44/timm/models/vision_transformer.py#L92

        # There are two easy ways to do attention, one is doing it like the 8tuple implementation, the other is like the 2tuple implementation
        # There is probably an even cleaner way to do it for this 5tuple implementation
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
        ), dim=-1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        x = self.att(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

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
    

# Block used by DeiT III

class Layer_scale_init_BlockD8(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=TritonGeluD8,
                 norm_layer=LayerNormD8v2,
                 Attention_block=AttentionD8,
                 Mlp_block=MlpD8,
                 init_values=1e-4,
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPathD8(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp_block(in_features=dim,
                             hidden_features=mlp_hidden_dim,
                             act_layer=act_layer,
                             drop=drop)
        self.gamma_1 = AffineD8(dim, bias=False)
        for _, param in self.gamma_1.named_parameters(): param.data = init_values*torch.ones_like(param.data)
        self.gamma_2 = AffineD8(dim, bias=False)
        for _, param in self.gamma_2.named_parameters(): param.data = init_values*torch.ones_like(param.data)

    def forward(self, xs):
        outs = self.drop_path(self.gamma_1(self.attn(self.norm1(xs))))
        xs = tuple(x + o for x, o in zip(xs, outs))
        outs = self.drop_path(self.gamma_2(self.mlp(self.norm2(xs))))
        return tuple(x + o for x, o in zip(xs, outs))



# Block used by DINOv2

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
        act_layer: Callable[..., nn.Module] = TritonGeluD8,
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

# Note, the nested tensor block is not optimized for speed like the original block in DINOv2 (using xformers)
# This is quite trivial to implement, but we chose to keep it simple for now
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