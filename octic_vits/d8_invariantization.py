# Code written for the paper "Stronger ViTs With Octic Equivariance" (https://arxiv.org/abs/TBD)
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .d8_utils import convert_5tuple_to_8tuple
from typing import Optional

class Invariant(nn.Module):
    """
    Base class for invariants.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.output_dim = dim

def invariant_head_factory(invariant: Invariant, C:int, num_classes:int, norm=False):
    """
    Factory function used in fully octic networks to project invariant features to logits."""    
    return nn.Sequential(
        nn.LayerNorm(invariant.output_dim, eps=1e-6) if norm else nn.Identity(),
        nn.Linear(invariant.output_dim, C),
        nn.GELU(),
        nn.Linear(C, num_classes) if num_classes > 0 else nn.Identity(),
    )
class NonInvariant(Invariant):
    def __init__(self, C:int):
        super().__init__(C)

    def forward(self, xtuple):
        xs = convert_5tuple_to_8tuple(xtuple)
        
        # Att lägga till denna faktorn leder till galna gradienter + nan
        # -------------------------
        xs = tuple(torch.abs(x) for x in xs)
        # -------------------------

        x = torch.cat(xs, dim=-1)
        return x
class LinearInvariant(Invariant):
    def __init__(self, C:int):
        super().__init__(C//8)

    def forward(self, xtuple):
        return torch.abs(xtuple[0])
class PowerSpectrumInvariant(Invariant):
    def __init__(self, C:int):
        super().__init__(6*C//8)
    def forward(self, xtuple):
        xtuple = tuple(x for x in xtuple)

        return torch.cat(
            (
                xtuple[0],
                xtuple[1].abs(),
                xtuple[2].abs(),
                xtuple[3].abs(),
                xtuple[4].norm(dim=-2, keepdim=False),
            ),
            dim=-1,
        )

class PolynomialInvariant(Invariant):
    """Basis for polynomial invariants."""
    def __init__(self, C:int):
        super().__init__(32*C//8)

    def forward(self, xtuple):
        x0, x1, x2, x3, x_2d = xtuple
        x46, x57 = x_2d.unbind(dim=-2)
        x4, x6 = x46.chunk(2, dim=-1)
        x5, x7 = x57.chunk(2, dim=-1)
        return torch.cat(
            (
                x0,
                x6**2 + x7**2,
                x4*x6 + x5*x7,
                x4**2 + x5**2,
                x3**2,
                x2**2,
                x1**2,
                x3*x6*x7,
                x3*x5*x6 + x3*x4*x7,
                x3*x4*x5,
                x2*x6**2 - x2*x7**2,
                x2*x4*x6 - x2*x5*x7,
                x2*x4**2 - x2*x5**2,
                x1*x5*x6 - x1*x4*x7,
                x1*x2*x3,
                x6**4 + x7**4,
                x4*x6**3 + x5*x7**3,
                x4**2*x6**2 + x5**2*x7**2,
                x4**3*x6 + x5**3*x7,
                x4**4 + x5**4,
                x2*x3*x5*x6 - x2*x3*x4*x7,
                x1*x3*x6**2 - x1*x3*x7**2,
                x1*x3*x4*x6 - x1*x3*x5*x7,
                x1*x3*x4**2 - x1*x3*x5**2,
                x1*x2*x6*x7,
                x1*x2*x5*x6 + x1*x2*x4*x7,
                x1*x2*x4*x5,
                x1*x6**3*x7 - x1*x6*x7**3,
                x1*x5*x6**3 - x1*x4*x7**3,
                x1*x4*x5*x6**2 - x1*x4*x5*x7**2,
                x1*x4**2*x5*x6 - x1*x4*x5**2*x7,
                x1*x4**3*x5 - x1*x4*x5**3,
            ),
            dim=-1,
        )
    
class ThirdOrderInvariant(Invariant):
    def __init__(self, C:int):
        super().__init__(15*C//8)
    def forward(self, xtuple):
        x0, x1, x2, x3, x_2d = xtuple
        x46, x57 = x_2d.unbind(dim=-2)
        x4, x6 = x46.chunk(2, dim=-1)
        x5, x7 = x57.chunk(2, dim=-1)
        return torch.cat(
            (
                x0**3,
                x0*(x6**2 + x7**2),
                x0*(x4*x6 + x5*x7),
                x0*(x4**2 + x5**2),
                x0*x3**2,
                x0*x2**2,
                x0*x1**2,
                x3*x6*x7,
                x3*x5*x6 + x3*x4*x7,
                x3*x4*x5,
                x2*x6**2 - x2*x7**2,
                x2*x4*x6 - x2*x5*x7,
                x2*x4**2 - x2*x5**2,
                x1*x5*x6 - x1*x4*x7,
                x1*x2*x3,
            ),
            dim=-1,
        )
class MaxFilteringInvariant(Invariant):
    def __init__(
        self,
        input_channels: int,
        num_references: Optional[int] = None,
        learnable_references: bool = True,
        global_avg: bool = False,
    ):
        if num_references is None:
            num_references = input_channels * 2
        super().__init__(num_references)
        self.references = nn.Parameter(
            F.normalize(
                torch.randn(num_references, input_channels//8, 8),
                dim=(1, 2),
            ),
            requires_grad=learnable_references,
        )
        self.rotation_action = nn.Parameter(
            torch.tensor([
                [1., 0, 0, 0, 0, 0, 0, 0],
                [0., 1, 0, 0, 0, 0, 0, 0],
                [0., 0, -1, 0, 0, 0, 0, 0],
                [0., 0, 0, -1, 0, 0, 0, 0],
                [0., 0, 0, 0, 0, -1, 0, 0],
                [0., 0, 0, 0, 1, 0, 0, 0],
                [0., 0, 0, 0, 0, 0, 0, -1],
                [0., 0, 0, 0, 0, 0, 1, 0],
            ]),
            requires_grad=False,
        )
        self.reflection_action = nn.Parameter(
            torch.diag_embed(torch.tensor(
                [1., -1, 1, -1, -1, 1, -1, 1]
            )),
            requires_grad=False,
        )
        self.global_avg = global_avg
    def expand_references_d8(self):
        r = self.rotation_action
        m = self.reflection_action
        stack = torch.stack(
            (
                self.references.transpose(-2, -1),
                torch.einsum("ij,dcj->dic", r, self.references),
                torch.einsum("ij,dcj->dic", r @ r, self.references),
                torch.einsum("ij,dcj->dic", r @ r @ r, self.references),
                torch.einsum("ij,dcj->dic", m, self.references),
                torch.einsum("ij,dcj->dic", m @ r, self.references),
                torch.einsum("ij,dcj->dic", m @ r @ r, self.references),
                torch.einsum("ij,dcj->dic", m @ r @ r @ r, self.references),
            ),
            dim=0,
        )
        return stack.flatten(start_dim=-2)
    def forward(self, xtuple):
        x0, x1, x2, x3, x_2d = xtuple
        x46, x57 = x_2d.unbind(dim=-2)
        x4, x6 = x46.chunk(2, dim=-1)
        x5, x7 = x57.chunk(2, dim=-1)
        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), dim=-1)

        expanded_ref = self.expand_references_d8()
        if self.global_avg:
            # Global avg. of tokens coming in here
            products = torch.einsum("kdc,bc->bkd", expanded_ref, x)
        else:
            products = torch.einsum("kdc,bnc->bnkd", expanded_ref, x)
        return torch.max(products, dim=-2, keepdim=False).values

class CanonizationInvariant(Invariant):
    def __init__(self, dim, learnable_reference=True, global_avg=False):
        super().__init__(dim)
        self.reference = nn.Parameter(
            F.normalize(torch.randn(dim), dim=0),
            requires_grad=learnable_reference,
        )
        self.rotation_action = nn.Parameter(
            torch.tensor([
                [1., 0, 0, 0, 0, 0, 0, 0],
                [0., 1, 0, 0, 0, 0, 0, 0],
                [0., 0, -1, 0, 0, 0, 0, 0],
                [0., 0, 0, -1, 0, 0, 0, 0],
                [0., 0, 0, 0, 0, -1, 0, 0],
                [0., 0, 0, 0, 1, 0, 0, 0],
                [0., 0, 0, 0, 0, 0, 0, -1],
                [0., 0, 0, 0, 0, 0, 1, 0],
            ]),
            requires_grad=False,
        )
        self.reflection_action = nn.Parameter(
            torch.diag_embed(torch.tensor(
                [1., -1, 1, -1, -1, 1, -1, 1]
            )),
            requires_grad=False,
        )
        self.global_avg = global_avg
    def expand_x_d8(self, x):
        r = self.rotation_action
        m = self.reflection_action
        stack = torch.stack(
            (
                x.transpose(-2, -1),
                torch.einsum("ij,...cj->...ic", r, x),
                torch.einsum("ij,...cj->...ic", r @ r, x),
                torch.einsum("ij,...cj->...ic", r @ r @ r, x),
                torch.einsum("ij,...cj->...ic", m, x),
                torch.einsum("ij,...cj->...ic", m @ r, x),
                torch.einsum("ij,...cj->...ic", m @ r @ r, x),
                torch.einsum("ij,...cj->...ic", m @ r @ r @ r, x),
            ),
            dim=-3,
        )
        return stack.flatten(start_dim=-2)
    def forward(self, xtuple):
        x0, x1, x2, x3, x_2d = xtuple
        x46, x57 = x_2d.unbind(dim=-2)
        x4, x6 = x46.chunk(2, dim=-1)
        x5, x7 = x57.chunk(2, dim=-1)
        x = torch.stack((x0, x1, x2, x3, x4, x5, x6, x7), dim=-1)
        
        # Global avg. of tokens coming in here
        if self.global_avg:
            x = x.unsqueeze(1)

        expanded_x = self.expand_x_d8(x)

        products = torch.einsum("c,bnkc->bnk", self.reference, expanded_x)
        max_idx = torch.max(products, dim=-1, keepdim=True).indices
        # lol, sorry for below code / G
        out = torch.gather(
            expanded_x,
            2,
            max_idx.unsqueeze(-1).expand(-1, -1, -1, self.reference.shape[0]),
        ).squeeze(-2)
        
        if self.global_avg:
            out = out.squeeze(1) # Adding last squeeze to remove token dim (because it is 1 after global avg pool)
        return out