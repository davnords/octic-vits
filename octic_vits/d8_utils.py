import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from typing import Tuple

SQRT2 = math.sqrt(2)
SQRT2_OVER_2 = 0.5 * SQRT2
SQRT2_OVER_4 = 0.5 * SQRT2_OVER_2

group_elements = ("e", "r", "rr", "rrr", "m", "mr", "mrr", "mrrr")
irreps = ("A1", "A2", "B1", "B2", "E11", "E21", "E21", "E22")
mult_table = [  # triples of g1, g2, g1g2
    ["r", "r", "rr"],
    ["r", "rr", "rrr"],
    ["r", "rrr", "e"],
    ["r", "m", "mrrr"],
    ["r", "mr", "m"],
    ["r", "mrr", "mr"],
    ["r", "mrrr", "mrr"],

    ["m", "r", "mr"],
    ["m", "rr", "mrr"],
    ["m", "rrr", "mrrr"],
    ["m", "m", "e"],
    ["m", "mr", "r"],
    ["m", "mrr", "rr"],
    ["m", "mrrr", "rrr"],

    ["rr", "r", "rrr"],
    ["rr", "rr", "e"],
    ["rr", "rrr", "r"],
    ["rr", "m", "mrr"],
    ["rr", "mr", "mrrr"],
    ["rr", "mrr", "m"],
    ["rr", "mrrr", "mr"],

    ["rrr", "r", "e"],
    ["rrr", "rr", "r"],
    ["rrr", "rrr", "rr"],
    ["rrr", "m", "mr"],
    ["rrr", "mr", "mrr"],
    ["rrr", "mrr", "mrrr"],
    ["rrr", "mrrr", "m"],

    ["mr", "r", "mrr"],
    ["mr", "rr", "mrrr"],
    ["mr", "rrr", "m"],
    ["mr", "m", "rrr"],
    ["mr", "mr", "e"],
    ["mr", "mrr", "r"],
    ["mr", "mrrr", "rr"],

    ["mrr", "r", "mrrr"],
    ["mrr", "rr", "m"],
    ["mrr", "rrr", "mr"],
    ["mrr", "m", "rr"],
    ["mrr", "mr", "rrr"],
    ["mrr", "mrr", "e"],
    ["mrr", "mrrr", "r"],

    ["mrrr", "r", "m"],
    ["mrrr", "rr", "mr"],
    ["mrrr", "rrr", "mrr"],
    ["mrrr", "m", "r"],
    ["mrrr", "mr", "rr"],
    ["mrrr", "mrr", "rrr"],
    ["mrrr", "mrrr", "e"],
]

def image_space_group_action(group_element, img):
    if group_element == "e":
        return img
    elif group_element == "r":  # rotation
        return img.rot90(dims=(-2, -1))
    elif group_element == "rr":
        return img.rot90(k=2, dims=(-2, -1))
    elif group_element == "rrr":
        return img.rot90(k=3, dims=(-2, -1))
    elif group_element == "m":  # mirroring/reflection
        return img.flip(-1)
    elif group_element == "mr":
        return img.rot90(dims=(-2, -1)).flip(-1)
    elif group_element == "mrr":
        return img.rot90(k=2, dims=(-2, -1)).flip(-1)
    elif group_element == "mrrr":
        return img.rot90(k=3, dims=(-2, -1)).flip(-1)
    else:
        raise ValueError("Invalid group element")

def regular_group_action(group_element, xs):
    if group_element == "e":
        return xs
    elif group_element == "r":  # rotation
        return (
            xs[1],
            xs[2],
            xs[3],
            xs[0],
            xs[7],
            xs[4],
            xs[5],
            xs[6]
        )
    elif group_element == "rr":
        return (
            xs[2],
            xs[3],
            xs[0],
            xs[1],
            xs[6],
            xs[7],
            xs[4],
            xs[5]
        )
    elif group_element == "rrr":
        return (
            xs[3],
            xs[0],
            xs[1],
            xs[2],
            xs[5],
            xs[6],
            xs[7],
            xs[4]
        )
    elif group_element == "m":  # mirroring/reflection
        return (
            xs[4],
            xs[5],
            xs[6],
            xs[7],
            xs[0],
            xs[1],
            xs[2],
            xs[3]
        )
    elif group_element == "mr":
        return (
            xs[7],
            xs[4],
            xs[5],
            xs[6],
            xs[1],
            xs[2],
            xs[3],
            xs[0]
        )
    elif group_element == "mrr":
        return (
            xs[6],
            xs[7],
            xs[4],
            xs[5],
            xs[2],
            xs[3],
            xs[0],
            xs[1]
        )
    elif group_element == "mrrr":
        return (
            xs[5],
            xs[6],
            xs[7],
            xs[4],
            xs[3],
            xs[0],
            xs[1],
            xs[2]
        )
    else:
        raise ValueError("Invalid group element")

def isotypic_group_action(group_element, xs):
    if group_element == "e":
        return xs
    elif group_element == "r":  # rotation
        return (
            xs[0],
            xs[1],
            -xs[2],
            -xs[3],
            -xs[5],
            xs[4],
            -xs[7],
            xs[6]
        )
    elif group_element == "rr":
        return (
            xs[0],
            xs[1],
            xs[2],
            xs[3],
            -xs[4],
            -xs[5],
            -xs[6],
            -xs[7]
        )
    elif group_element == "rrr":
        return (
            xs[0],
            xs[1],
            -xs[2],
            -xs[3],
            xs[5],
            -xs[4],
            xs[7],
            -xs[6]
        )
    elif group_element == "m":  # mirroring/reflection
        return (
            xs[0],
            -xs[1],
            xs[2],
            -xs[3],
            -xs[4],
            xs[5],
            -xs[6],
            xs[7]
        )
    elif group_element == "mr":
        return (
            xs[0],
            -xs[1],
            -xs[2],
            xs[3],
            xs[5],
            xs[4],
            xs[7],
            xs[6]
        )
    elif group_element == "mrr":
        return (
            xs[0],
            -xs[1],
            xs[2],
            -xs[3],
            xs[4],
            -xs[5],
            xs[6],
            -xs[7]
        )
    elif group_element == "mrrr":
        return (
            xs[0],
            -xs[1],
            -xs[2],
            xs[3],
            -xs[5],
            -xs[4],
            -xs[7],
            -xs[6]
        )
    else:
        raise ValueError("Invalid group element")

def spatial_and_isotypic_group_action(group_element, xs):
    B, L, C = xs[0].shape
    H = W = int(math.sqrt(L))
    return isotypic_group_action(
        group_element,
        tuple(
            image_space_group_action(
                group_element,
                x.transpose(1, 2).reshape(B, C, H, W)
            ).flatten(2).transpose(1, 2)
            for x in xs
        ),
    )

def isotypic_to_regular_D8(xs):
    # TODO: potentially FFT should not be hard-coded?
    a = xs[0] + xs[1]
    b = xs[0] - xs[1]
    c = xs[2] + xs[3]
    d = xs[2] - xs[3]
    e = xs[4] + xs[5]
    f = xs[4] - xs[5]
    g = xs[6] + xs[7]
    h = xs[6] - xs[7]
    apc = a + c
    amc = a - c
    bpd = b + d
    bmd = b - d
    eph = e + h
    emh = e - h
    fpg = f + g
    fmg = f - g
    return (
        SQRT2_OVER_4 * (apc + eph),
        SQRT2_OVER_4 * (amc + fmg),
        SQRT2_OVER_4 * (apc - eph),
        SQRT2_OVER_4 * (amc - fmg),
        SQRT2_OVER_4 * (bpd - fpg),
        SQRT2_OVER_4 * (bmd - emh),
        SQRT2_OVER_4 * (bpd + fpg),
        SQRT2_OVER_4 * (bmd + emh)
    )

def isotypic_to_regular_D8_no_FFT(xs):
    return (
      SQRT2_OVER_4 * (xs[0] + xs[1] + xs[2] + xs[3] + xs[4] + xs[5] + xs[6] - xs[7]),
      SQRT2_OVER_4 * (xs[0] + xs[1] - xs[2] - xs[3] + xs[4] - xs[5] - xs[6] - xs[7]),
      SQRT2_OVER_4 * (xs[0] + xs[1] + xs[2] + xs[3] - xs[4] - xs[5] - xs[6] + xs[7]),
      SQRT2_OVER_4 * (xs[0] + xs[1] - xs[2] - xs[3] - xs[4] + xs[5] + xs[6] + xs[7]),
      SQRT2_OVER_4 * (xs[0] - xs[1] + xs[2] - xs[3] - xs[4] + xs[5] - xs[6] - xs[7]),
      SQRT2_OVER_4 * (xs[0] - xs[1] - xs[2] + xs[3] - xs[4] - xs[5] + xs[6] - xs[7]),
      SQRT2_OVER_4 * (xs[0] - xs[1] + xs[2] - xs[3] + xs[4] - xs[5] + xs[6] + xs[7]),
      SQRT2_OVER_4 * (xs[0] - xs[1] - xs[2] + xs[3] + xs[4] + xs[5] - xs[6] + xs[7])
    )

def regular_to_isotypic_D8(xs):
    # TODO: potentially FFT should not be hard-coded
    a = xs[0] + xs[1]
    b = xs[0] - xs[1]
    c = xs[2] + xs[3]
    d = xs[2] - xs[3]
    e = xs[4] + xs[5]
    f = xs[4] - xs[5]
    g = xs[6] + xs[7]
    h = xs[6] - xs[7]
    apc = a + c
    cma = c - a
    bpd = b + d
    bmd = b - d
    epg = e + g
    gme = g - e
    fph = f + h
    fmh = f - h
    return (
        SQRT2_OVER_4 * (apc + epg),
        SQRT2_OVER_4 * (apc - epg),
        SQRT2_OVER_4 * (bpd + fph),
        SQRT2_OVER_4 * (bpd - fph),
        SQRT2_OVER_4 * (gme - cma),
        SQRT2_OVER_4 * (bmd + fmh),
        SQRT2_OVER_4 * (bmd - fmh),
        SQRT2_OVER_4 * (gme + cma)
    )

def regular_to_isotypic_D8_no_FFT(xs):
    return (
      SQRT2_OVER_4 * ( xs[0] + xs[1] + xs[2] + xs[3] + xs[4] + xs[5] + xs[6] + xs[7]),
      SQRT2_OVER_4 * ( xs[0] + xs[1] + xs[2] + xs[3] - xs[4] - xs[5] - xs[6] - xs[7]),
      SQRT2_OVER_4 * ( xs[0] - xs[1] + xs[2] - xs[3] + xs[4] - xs[5] + xs[6] - xs[7]),
      SQRT2_OVER_4 * ( xs[0] - xs[1] + xs[2] - xs[3] - xs[4] + xs[5] - xs[6] + xs[7]),
      SQRT2_OVER_4 * ( xs[0] + xs[1] - xs[2] - xs[3] - xs[4] - xs[5] + xs[6] + xs[7]),
      SQRT2_OVER_4 * ( xs[0] - xs[1] - xs[2] + xs[3] + xs[4] - xs[5] - xs[6] + xs[7]),
      SQRT2_OVER_4 * ( xs[0] - xs[1] - xs[2] + xs[3] - xs[4] + xs[5] + xs[6] - xs[7]),
      SQRT2_OVER_4 * (-xs[0] - xs[1] + xs[2] + xs[3] - xs[4] - xs[5] + xs[6] + xs[7])
    )

def convert_8tuple_to_5tuple(xs):
    return (
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        torch.cat((
            torch.stack(xs[4:6], dim=-2),
            torch.stack(xs[6:], dim=-2),
        ), dim=-1)
    )

def convert_5tuple_to_8tuple(xs):
    stacked = xs[4]  # shape: [..., 2, D]
    first_half, second_half = torch.split(stacked, stacked.shape[-1] // 2, dim=-1)  # shape [..., 2, D/2] each
    xs4, xs5 = first_half.unbind(dim=-2)   # shape [..., D/2] each
    xs6, xs7 = second_half.unbind(dim=-2)  # shape [..., D/2] each

    return (
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs4,
        xs5,
        xs6,
        xs7
    )


def isotypic_dim_interpolation(xs: Tuple[torch.Tensor], dim:int=0):
        """
        Description:
        Takes a tuple of weights and folds out the symmetries, thus extending the spatial dimension and distributing over the feature channels.
        Returns a tuple of 8 features that now in total cover 4 times the spatial area because of symmetric unfolding.

        Input: 
        xs: Tuple of 6 tensors of shape [..., H//2, W//2, C//8] where H the number of patches in the height direction, W the number of patches in the width direction, and C the number of channels.
        dim: The dimension of the spatial dimension to interpolate over. Default is 2. But for example if you have a batch dimension you can set dim=1.

        Output:
        Tuple of 8 tensors of shape [..., H, W, C//8] where H the number of patches in the height direction, W the number of patches in the width direction, and C the number of channels.
        """
        out_A1 = torch.cat((
            torch.cat((xs[0],
                       xs[0].rot90(k=1, dims=(dim, dim+1))), dim=dim),
            torch.cat((xs[0].rot90(k=3, dims=(dim, dim+1)),
                       xs[0].rot90(k=2, dims=(dim, dim+1))), dim=dim)
        ), dim=dim+1)
        out_A1 = (out_A1 + out_A1.flip(dim+1))

        out_A2 = torch.cat((
            torch.cat((xs[1],           # dims=(dim, dim+1))), dim=dim),
                       xs[1].rot90(k=1, dims=(dim, dim+1))), dim=dim),
            torch.cat((xs[1].rot90(k=3, dims=(dim, dim+1)),
                       xs[1].rot90(k=2, dims=(dim, dim+1))), dim=dim)
        ), dim=dim+1)
        out_A2 = (out_A2 - out_A2.flip(dim+1))

        out_B1 = torch.cat((
            torch.cat((xs[2],
                       -xs[2].rot90(k=1, dims=(dim, dim+1))), dim=dim),
            torch.cat((-xs[2].rot90(k=3, dims=(dim, dim+1)),
                       xs[2].rot90(k=2, dims=(dim, dim+1))), dim=dim)
        ), dim=dim+1)
        out_B1 = (out_B1 + out_B1.flip(dim+1))

        out_B2 = torch.cat((
            torch.cat((xs[3],
                       -xs[3].rot90(k=1, dims=(dim, dim+1))), dim=dim),
            torch.cat((-xs[3].rot90(k=3, dims=(dim, dim+1)),
                       xs[3].rot90(k=2, dims=(dim, dim+1))), dim=dim)
        ), dim=dim+1)
        out_B2 = (out_B2 - out_B2.flip(dim+1))

        out_E_left = torch.cat((
            xs[4], xs[4].flip(dim)
        ), dim=dim)
        out_E_left = torch.cat((
            out_E_left, -out_E_left.flip(dim+1)
        ), dim=dim+1)

        out_E_right = torch.cat((
            xs[5], xs[5].flip(dim)
        ), dim=dim)
        out_E_right = torch.cat((
            out_E_right, -out_E_right.flip(dim+1)
        ), dim=dim+1)

        return (
            out_A1, out_A2, out_B1, out_B2,
            out_E_left, out_E_left.rot90(dims=(dim, dim+1)),
            out_E_right, out_E_right.rot90(dims=(dim, dim+1))
        )

def interpolate_spatial_tuple(xs: tuple[torch.Tensor], interpolant: tuple[torch.Tensor], h:int, w:int, patch_size:int):
        """
        Input:
        - xs: Token embeddings
        - interpolant: Tuple to be interpolated, for example pos_enc, expecting tensors of shape like (H, W, C)
        - h: height
        - w: width
        - patch_size: patch size
        """
        # Inspired by Dinov2: https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179
        previous_dtype = xs[0].dtype
        npatch = xs[0].shape[1]
        
        # Check the first interpolant's shape to determine dimensions
        N = interpolant[0].shape[0]**2
        
        # Early return if no interpolation needed
        if npatch == N and w == h:
            return interpolant
        
        # Calculate the dimensions for interpolation
        dim = xs[0].shape[-1]
        h0 = h // patch_size
        w0 = w // patch_size
        M = int(math.sqrt(N))  # Number of patches in each dimension
        assert N == M * M
        
        # Stack all position embeddings into a single tensor for batched processing
        # Shape becomes [num_pos_embeds, M//2, M//2, dim]
        stacked_embeds = torch.stack([embed.float() for embed in interpolant], dim=0)
        
        # Reshape for interpolation: [num_pos_embeds, dim, M//2, M//2]
        stacked_embeds = stacked_embeds.reshape(len(interpolant), M, M, dim).permute(0, 3, 1, 2)
        
        # Batched interpolation
        interpolated = nn.functional.interpolate(
            stacked_embeds,
            size=(h0, w0),
            mode="bicubic",
            antialias=False,
        )
        
        # Reshape back: [num_pos_embeds, h0, w0, dim]
        interpolated = interpolated.permute(0, 2, 3, 1)
        
        # Convert back to list and restore original dtype
        return [interpolated[i].to(previous_dtype) for i in range(len(interpolant))]

