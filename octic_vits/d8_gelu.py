# Code written for the paper "Stronger ViTs With Octic Equivariance" (https://arxiv.org/abs/TBD)
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl
import helion.language as hl
import helion

@triton.jit
def tl_gelu(x, SQRT2_OVER_2: tl.constexpr=0.7071067812):
    x_fp32 = x.to(tl.float32)  # Cast to fp32 to avoid mixed precision error (tl.math.erf is not supported in fp16)
    cdf = 0.5 * (1.0 + tl.math.erf(SQRT2_OVER_2 * x_fp32))  
    return x * cdf.to(x.dtype)

@triton.jit
def tl_gelu_grad(x,
                 SQRT2_OVER_2: tl.constexpr=0.7071067812,
                 ONE_OVER_SQRT2PI: tl.constexpr=0.3989422804):
    
    x_fp32 = x.to(tl.float32) # Cast to fp32 to avoid mixed precision error (tl.math.erf is not supported in fp16)
    cdf = 0.5 * (1 + tl.math.erf(SQRT2_OVER_2 * x_fp32))
    cdf_grad = ONE_OVER_SQRT2PI * tl.exp(-0.5 * x_fp32 * x_fp32)
    cdf_grad = cdf_grad.to(x.dtype)  # Cast back to original dtype
    cdf = cdf.to(x.dtype)  # Cast back to original dtype
    return (cdf_grad * x + cdf)

@triton.jit
def tl_isotypic_to_regular(
    x0, x1, x2, x3, x4, x5, x6, x7,
    SQRT2_OVER_4: tl.constexpr=0.3535533906,
):
    a = x0 + x1
    b = x0 - x1
    c = x2 + x3
    d = x2 - x3
    e = x4 + x5
    f = x4 - x5
    g = x6 + x7
    h = x6 - x7
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

@triton.jit
def tl_regular_to_isotypic(
    x0, x1, x2, x3, x4, x5, x6, x7,
    SQRT2_OVER_4: tl.constexpr=0.3535533906,
):
    a = x0 + x1
    b = x0 - x1
    c = x2 + x3
    d = x2 - x3
    e = x4 + x5
    f = x4 - x5
    g = x6 + x7
    h = x6 - x7
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

@triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
            triton.Config({'BLOCK_SIZE': 1024}),
            triton.Config({'BLOCK_SIZE': 2048}),
            triton.Config({'BLOCK_SIZE': 4096}),
        ],
        key=['B', 'N', 'C'],
)
@triton.jit
def d8_gelu_fwd_kernel(
    # Pointer to tensor
    x_A1_ptr,  # Input tensor [B, N, C]
    x_A2_ptr,  # [B, N, C]
    x_B1_ptr,  # [B, N, C]
    x_B2_ptr,  # [B, N, C]
    x_2d_ptr,  # [B, N, 2, 2*C]
    y_A1_ptr,  # Output tensor
    y_A2_ptr,  
    y_B1_ptr,  
    y_B2_ptr,  
    y_2d_ptr,
    # Matrix dimensions
    B,  # Batch size
    N,  # Sequence length
    C,  # Channel dimension size
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Use a 1D grid for simplicity
    pid = tl.program_id(0)
    
    # Total number of elements in C dimension across all B,N
    total_elements = B * N * C
    
    # Calculate the starting position for this thread block
    start_idx = pid * BLOCK_SIZE
    
    # Process BLOCK_SIZE elements at once
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices
    mask = offsets < total_elements
    
    # For each valid offset, calculate the corresponding b, n, c indices
    # Each element in C dimension corresponds to a pair in the 3rd dimension
    b = offsets // (N * C)
    n = (offsets // C) % N
    c = offsets % C
    
    # Only process indices that are in bounds
    valid = mask & (b < B) & (n < N) & (c < C)
    
    # Calculate the actual tensor indices for valid elements
    indices = b * (N * C) + n * C + c  # index for [b, n, c]
    indices_0 = b * (N * 4 * C) + n * (4 * C) + c  # index for [b, n, 0, c] in x_2d
    indices_1 = indices_0 + C  # Offset to get to [b, n, 0, C+c]
    indices_2 = indices_1 + C  # Offset to get to [b, n, 1, c]
    indices_3 = indices_2 + C  # Offset to get to [b, n, 1, C+c]
    
    # Load elements for valid indices
    x_1d_0 = tl.load(x_A1_ptr + indices, mask=valid)
    x_1d_1 = tl.load(x_A2_ptr + indices, mask=valid)
    x_1d_2 = tl.load(x_B1_ptr + indices, mask=valid)
    x_1d_3 = tl.load(x_B2_ptr + indices, mask=valid)
    x_2d_0 = tl.load(x_2d_ptr + indices_0, mask=valid)
    x_2d_1 = tl.load(x_2d_ptr + indices_1, mask=valid)
    x_2d_2 = tl.load(x_2d_ptr + indices_2, mask=valid)
    x_2d_3 = tl.load(x_2d_ptr + indices_3, mask=valid)
    
    # inv FFT
    (
        x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_2, x_2d_1, x_2d_3
    ) = tl_isotypic_to_regular(
        x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_2, x_2d_1, x_2d_3
    )

    # gelu
    x_1d_0 = tl_gelu(x_1d_0)
    x_1d_1 = tl_gelu(x_1d_1)
    x_1d_2 = tl_gelu(x_1d_2)
    x_1d_3 = tl_gelu(x_1d_3)
    x_2d_0 = tl_gelu(x_2d_0)
    x_2d_1 = tl_gelu(x_2d_1)
    x_2d_2 = tl_gelu(x_2d_2)
    x_2d_3 = tl_gelu(x_2d_3)

    # FFT
    (
        x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_2, x_2d_1, x_2d_3
    ) = tl_regular_to_isotypic(
        x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_2, x_2d_1, x_2d_3
    )
    
    # Store results
    tl.store(y_A1_ptr + indices, x_1d_0, mask=valid)
    tl.store(y_A2_ptr + indices, x_1d_1, mask=valid)
    tl.store(y_B1_ptr + indices, x_1d_2, mask=valid)
    tl.store(y_B2_ptr + indices, x_1d_3, mask=valid)
    tl.store(y_2d_ptr + indices_0, x_2d_0, mask=valid)
    tl.store(y_2d_ptr + indices_1, x_2d_1, mask=valid)
    tl.store(y_2d_ptr + indices_2, x_2d_2, mask=valid)
    tl.store(y_2d_ptr + indices_3, x_2d_3, mask=valid)

@triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
            triton.Config({'BLOCK_SIZE': 1024}),
            triton.Config({'BLOCK_SIZE': 2048}),
            triton.Config({'BLOCK_SIZE': 4096}),
        ],
        key=['B', 'N', 'C'],
)
@triton.jit
def d8_gelu_bwd_kernel(
    g_A1_ptr,  # gradient tensor [B, N, C]
    g_A2_ptr,  # [B, N, C]
    g_B1_ptr,  # [B, N, C]
    g_B2_ptr,  # [B, N, C]
    g_2d_ptr,  # [B, N, 2, 2*C]
    x_A1_ptr,  # Input tensor [B, N, C]
    x_A2_ptr,  # [B, N, C]
    x_B1_ptr,  # [B, N, C]
    x_B2_ptr,  # [B, N, C]
    x_2d_ptr,  # [B, N, 2, 2*C]
    g_in_A1_ptr,  # gradient of input tensor [B, N, C]
    g_in_A2_ptr,  # [B, N, C]
    g_in_B1_ptr,  # [B, N, C]
    g_in_B2_ptr,  # [B, N, C]
    g_in_2d_ptr,  # [B, N, 2, 2*C]
    # Matrix dimensions
    B,  # Batch size
    N,  # Sequence length
    C,  # Channel dimension size
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Use a 1D grid for simplicity
    pid = tl.program_id(0)
    
    # Total number of elements in C dimension across all B,N
    total_elements = B * N * C
    
    # Calculate the starting position for this thread block
    start_idx = pid * BLOCK_SIZE
    
    # Process BLOCK_SIZE elements at once
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid indices
    mask = offsets < total_elements
    
    # For each valid offset, calculate the corresponding b, n, c indices
    # Each element in C dimension corresponds to a pair in the 3rd dimension
    b = offsets // (N * C)
    n = (offsets // C) % N
    c = offsets % C
    
    # Only process indices that are in bounds
    valid = mask & (b < B) & (n < N) & (c < C)
    
    # Calculate the actual tensor indices for valid elements
    indices = b * (N * C) + n * C + c  # index for [b, n, c]
    indices_0 = b * (N * 4 * C) + n * (4 * C) + c  # index for [b, n, 0, c] in x_2d
    indices_1 = indices_0 + C  # Offset to get to [b, n, 0, C+c]
    indices_2 = indices_1 + C  # Offset to get to [b, n, 1, c]
    indices_3 = indices_2 + C  # Offset to get to [b, n, 1, C+c]
    
    # Load elements for valid indices
    x_1d_0 = tl.load(x_A1_ptr + indices, mask=valid)
    x_1d_1 = tl.load(x_A2_ptr + indices, mask=valid)
    x_1d_2 = tl.load(x_B1_ptr + indices, mask=valid)
    x_1d_3 = tl.load(x_B2_ptr + indices, mask=valid)
    x_2d_0 = tl.load(x_2d_ptr + indices_0, mask=valid)
    x_2d_1 = tl.load(x_2d_ptr + indices_1, mask=valid)
    x_2d_2 = tl.load(x_2d_ptr + indices_2, mask=valid)
    x_2d_3 = tl.load(x_2d_ptr + indices_3, mask=valid)
    
    g_1d_0 = tl.load(g_A1_ptr + indices, mask=valid)
    g_1d_1 = tl.load(g_A2_ptr + indices, mask=valid)
    g_1d_2 = tl.load(g_B1_ptr + indices, mask=valid)
    g_1d_3 = tl.load(g_B2_ptr + indices, mask=valid)
    g_2d_0 = tl.load(g_2d_ptr + indices_0, mask=valid)
    g_2d_1 = tl.load(g_2d_ptr + indices_1, mask=valid)
    g_2d_2 = tl.load(g_2d_ptr + indices_2, mask=valid)
    g_2d_3 = tl.load(g_2d_ptr + indices_3, mask=valid)

    # inv FFT
    (
        x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_2, x_2d_1, x_2d_3
    ) = tl_isotypic_to_regular(
        x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_2, x_2d_1, x_2d_3
    )
    # inv FFT (FFT transposed)
    (
        g_1d_0, g_1d_1, g_1d_2, g_1d_3, g_2d_0, g_2d_2, g_2d_1, g_2d_3
    ) = tl_isotypic_to_regular(
        g_1d_0, g_1d_1, g_1d_2, g_1d_3, g_2d_0, g_2d_2, g_2d_1, g_2d_3
    )

    # gelu grad
    x_1d_0 = tl_gelu_grad(x_1d_0)
    x_1d_1 = tl_gelu_grad(x_1d_1)
    x_1d_2 = tl_gelu_grad(x_1d_2)
    x_1d_3 = tl_gelu_grad(x_1d_3)
    x_2d_0 = tl_gelu_grad(x_2d_0)
    x_2d_1 = tl_gelu_grad(x_2d_1)
    x_2d_2 = tl_gelu_grad(x_2d_2)
    x_2d_3 = tl_gelu_grad(x_2d_3)

    # multiply according to chain rule
    g_1d_0 = x_1d_0 * g_1d_0
    g_1d_1 = x_1d_1 * g_1d_1
    g_1d_2 = x_1d_2 * g_1d_2
    g_1d_3 = x_1d_3 * g_1d_3
    g_2d_0 = x_2d_0 * g_2d_0
    g_2d_1 = x_2d_1 * g_2d_1
    g_2d_2 = x_2d_2 * g_2d_2
    g_2d_3 = x_2d_3 * g_2d_3

    # FFT (inv FFT transposed)
    (
        g_1d_0, g_1d_1, g_1d_2, g_1d_3, g_2d_0, g_2d_2, g_2d_1, g_2d_3
    ) = tl_regular_to_isotypic(
        g_1d_0, g_1d_1, g_1d_2, g_1d_3, g_2d_0, g_2d_2, g_2d_1, g_2d_3
    )

    # Store gradient results
    tl.store(g_in_A1_ptr + indices, g_1d_0, mask=valid)
    tl.store(g_in_A2_ptr + indices, g_1d_1, mask=valid)
    tl.store(g_in_B1_ptr + indices, g_1d_2, mask=valid)
    tl.store(g_in_B2_ptr + indices, g_1d_3, mask=valid)
    tl.store(g_in_2d_ptr + indices_0, g_2d_0, mask=valid)
    tl.store(g_in_2d_ptr + indices_1, g_2d_1, mask=valid)
    tl.store(g_in_2d_ptr + indices_2, g_2d_2, mask=valid)
    tl.store(g_in_2d_ptr + indices_3, g_2d_3, mask=valid)

def d8_gelu_fwd(x_A1, x_A2, x_B1, x_B2, x_2d):
    """
    Args:
        x_A1: Input tensor of shape [B, N, C]
        x_A2: Input tensor of shape [B, N, C]
        x_B1: Input tensor of shape [B, N, C]
        x_B2: Input tensor of shape [B, N, C]
        x_2d: Input tensor of shape [B, N, 2, 2*C]
        BLOCK_SIZE: Size of blocks. Effective number of elements loaded is 8 times this.

    Returns:
        The input tensors after d8-equivariant gelu, by iFFT->gelu->FFT.
    """
    assert len(x_A1.shape) == 3, "Input must be a 3D tensor [B, N, C]"
    assert len(x_2d.shape) == 4, "Input must be a 4D tensor [B, N, 2, 2*C]"
    assert x_2d.shape[2] == 2, "3rd dim must be size 2"
    assert x_A1.shape[0] == x_2d.shape[0], "Inputs must have compatible size"
    assert x_A1.shape[1] == x_2d.shape[1], "Inputs must have compatible size"
    assert 2*x_A1.shape[2] == x_2d.shape[3], "Inputs must have compatible size"
    for x, y in zip(x_A1.shape, x_A2.shape):
        assert x == y, "Inputs must have compatible size"
    for x, y in zip(x_A1.shape, x_B1.shape):
        assert x == y, "Inputs must have compatible size"
    for x, y in zip(x_A1.shape, x_B2.shape):
        assert x == y, "Inputs must have compatible size"
    assert 2*x_A1.shape[2] == x_2d.shape[3], "Inputs must have compatible size"
    assert x_A1.is_contiguous(), "Input must be contiguous"
    assert x_A2.is_contiguous(), "Input must be contiguous"
    assert x_B1.is_contiguous(), "Input must be contiguous"
    assert x_B2.is_contiguous(), "Input must be contiguous"
    assert x_2d.is_contiguous(), "Input must be contiguous"
    
    B, N, C = x_A1.shape
    y_A1 = torch.empty_like(x_A1)
    y_A2 = torch.empty_like(x_A2)
    y_B1 = torch.empty_like(x_B1)
    y_B2 = torch.empty_like(x_B2)
    y_2d = torch.empty_like(x_2d)
    
    # Total number of C elements across all B,N dimensions
    total_elements = B * N * C
    
    # Number of blocks needed
    grid = lambda META: (triton.cdiv(total_elements, META['BLOCK_SIZE']), ) 
    
    # Launch the kernel with a 1D grid for simplicity
    d8_gelu_fwd_kernel[grid](
        x_A1, x_A2, x_B1, x_B2, x_2d, y_A1, y_A2, y_B1, y_B2, y_2d, B, N, C
    )
    
    return y_A1, y_A2, y_B1, y_B2, y_2d

def d8_gelu_bwd(g_A1, g_A2, g_B1, g_B2, g_2d, x_A1, x_A2, x_B1, x_B2, x_2d):
    """
    
    Args:
        g_A1: gradient tensor of shape [B, N, C]
        g_A2: gradient tensor of shape [B, N, C]
        g_B1: gradient tensor of shape [B, N, C]
        g_B2: gradient tensor of shape [B, N, C]
        g_2d: gradient tensor of shape [B, N, 2, 2*C]
        x_A1: Original input tensor of shape [B, N, C]
        x_A2: Original input tensor of shape [B, N, C]
        x_B1: Original input tensor of shape [B, N, C]
        x_B2: Original input tensor of shape [B, N, C]
        x_2d: Original input tensor of shape [B, N, 2, 2*C]
        
    Returns:
        The gradiant tensors before d8-equivariant gelu, by iFFT->gelu->FFT.
    """
    assert len(x_A1.shape) == 3, "Input must be a 3D tensor [B, N, C]"
    assert len(x_2d.shape) == 4, "Input must be a 4D tensor [B, N, 2, 2*C]"
    assert x_2d.shape[2] == 2, "3rd dim must be size 2"
    assert x_A1.shape[0] == x_2d.shape[0], "Inputs must have compatible size"
    assert x_A1.shape[1] == x_2d.shape[1], "Inputs must have compatible size"
    assert 2*x_A1.shape[2] == x_2d.shape[3], "Inputs must have compatible size"
    for x, y in zip(x_A1.shape, x_A2.shape):
        assert x == y, "Inputs must have compatible size"
    for x, y in zip(x_A1.shape, x_B1.shape):
        assert x == y, "Inputs must have compatible size"
    for x, y in zip(x_A1.shape, x_B2.shape):
        assert x == y, "Inputs must have compatible size"
    assert 2*x_A1.shape[2] == x_2d.shape[3], "Inputs must have compatible size"
    assert x_A1.is_contiguous(), "Input must be contiguous"
    assert x_A2.is_contiguous(), "Input must be contiguous"
    assert x_B1.is_contiguous(), "Input must be contiguous"
    assert x_B2.is_contiguous(), "Input must be contiguous"
    assert x_2d.is_contiguous(), "Input must be contiguous"
    for g, x in zip(g_A1.shape, x_A1.shape):
        assert g == x, "Inputs must have compatible size"
    for g, x in zip(g_A2.shape, x_A2.shape):
        assert g == x, "Inputs must have compatible size"
    for g, x in zip(g_B1.shape, x_B1.shape):
        assert g == x, "Inputs must have compatible size"
    for g, x in zip(g_B2.shape, x_B2.shape):
        assert g == x, "Inputs must have compatible size"
    for g, x in zip(g_2d.shape, x_2d.shape):
        assert g == x, "Inputs must have compatible size"
    
    B, N, C = x_A1.shape

    g_in_A1 = torch.empty_like(g_A1)
    g_in_A2 = torch.empty_like(g_A2)
    g_in_B1 = torch.empty_like(g_B1)
    g_in_B2 = torch.empty_like(g_B2)
    g_in_2d = torch.empty_like(g_2d)
    
    # Total number of C elements across all B,N dimensions
    total_elements = B * N * C
    
    # Number of blocks needed
    grid = lambda META: (triton.cdiv(total_elements, META['BLOCK_SIZE']), ) 
    
    # Launch the kernel with a 1D grid for simplicity
    d8_gelu_bwd_kernel[grid](
        g_A1, g_A2, g_B1, g_B2, g_2d,
        x_A1, x_A2, x_B1, x_B2, x_2d,
        g_in_A1, g_in_A2, g_in_B1, g_in_B2, g_in_2d,
        B, N, C
    )
    
    return g_in_A1, g_in_A2, g_in_B1, g_in_B2, g_in_2d

class TritonGeluD8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_A1, x_A2, x_B1, x_B2, x_2d):
        x_A1, x_A2, x_B1, x_B2, x_2d = (
            x_A1.contiguous(), x_A2.contiguous(),
            x_B1.contiguous(), x_B2.contiguous(),
            x_2d.contiguous()
        )
        ctx.save_for_backward(x_A1, x_A2, x_B1, x_B2, x_2d)
        return d8_gelu_fwd(x_A1, x_A2, x_B1, x_B2, x_2d)
    @staticmethod
    def backward(ctx, g_A1, g_A2, g_B1, g_B2, g_2d):
        x_A1, x_A2, x_B1, x_B2, x_2d = ctx.saved_tensors
        return d8_gelu_bwd(g_A1.contiguous(),
                           g_A2.contiguous(),
                           g_B1.contiguous(),
                           g_B2.contiguous(),
                           g_2d.contiguous(),
                           x_A1.contiguous(),
                           x_A2.contiguous(),
                           x_B1.contiguous(),
                           x_B2.contiguous(),
                           x_2d.contiguous())

class TritonGeluD8(torch.nn.Module):
    def forward(self, xs):
        return TritonGeluD8Function.apply(xs[0], xs[1], xs[2], xs[3], xs[4])

################################################################################
# Helion implementation
################################################################################

def gelu(x):
    SQRT_OVER_2 = 0.7071067812
    cdf = 0.5 * (1.0 + torch.erf(SQRT_OVER_2 * x))
    return x * cdf

def gelu_grad(x):
    SQRT_OVER_2 = 0.7071067812
    ONE_OVER_SQRT2PI = 0.3989422804
    x_fp32 = x.to(torch.float32)
    cdf = 0.5 * (1 + torch.erf(SQRT_OVER_2 * x_fp32))
    cdf_grad = ONE_OVER_SQRT2PI * torch.exp(-0.5 * x_fp32 * x_fp32)
    cdf_grad = cdf_grad.to(x.dtype)
    cdf = cdf.to(x.dtype)
    return cdf_grad * x + cdf

def isotypic_to_regular(x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_1, x_2d_2, x_2d_3):
    SQRT2_OVER_4 = 0.3535533906
    a = x_1d_0 + x_1d_1
    b = x_1d_0 - x_1d_1
    c = x_1d_2 + x_1d_3
    d = x_1d_2 - x_1d_3
    e = x_2d_0 + x_2d_1
    f = x_2d_0 - x_2d_1
    g = x_2d_2 + x_2d_3
    h = x_2d_2 - x_2d_3
    apc = a + c
    amc = a - c
    bpd = b + d
    bmd = b - d
    eph = e + h
    emh = e - h
    fpg = f + g
    fmg = f - g
    
    x_1d_0 = SQRT2_OVER_4 * (apc + eph)
    x_1d_1 = SQRT2_OVER_4 * (amc + fmg)
    x_1d_2 = SQRT2_OVER_4 * (apc - eph)
    x_1d_3 = SQRT2_OVER_4 * (amc - fmg)
    x_2d_0 = SQRT2_OVER_4 * (bpd - fpg)
    x_2d_1 = SQRT2_OVER_4 * (bmd - emh)
    x_2d_2 = SQRT2_OVER_4 * (bpd + fpg)
    x_2d_3 = SQRT2_OVER_4 * (bmd + emh)
    
    return x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_1, x_2d_2, x_2d_3

def regular_to_isotypic(x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_1, x_2d_2, x_2d_3):
    SQRT2_OVER_4 = 0.3535533906
    a = x_1d_0 + x_1d_1
    b = x_1d_0 - x_1d_1
    c = x_1d_2 + x_1d_3
    d = x_1d_2 - x_1d_3
    e = x_2d_0 + x_2d_1
    f = x_2d_0 - x_2d_1
    g = x_2d_2 + x_2d_3
    h = x_2d_2 - x_2d_3
    apc = a + c
    cma = c - a
    bpd = b + d
    bmd = b - d
    epg = e + g
    gme = g - e
    fph = f + h
    fmh = f - h
    
    x_1d_0 = SQRT2_OVER_4 * (apc + epg)
    x_1d_1 = SQRT2_OVER_4 * (apc - epg)
    x_1d_2 = SQRT2_OVER_4 * (bpd + fph)
    x_1d_3 = SQRT2_OVER_4 * (bpd - fph)
    x_2d_0 = SQRT2_OVER_4 * (gme - cma)
    x_2d_1 = SQRT2_OVER_4 * (bmd + fmh)
    x_2d_2 = SQRT2_OVER_4 * (bmd - fmh)
    x_2d_3 = SQRT2_OVER_4 * (gme + cma)
    
    return x_1d_0, x_1d_1, x_1d_2, x_1d_3, x_2d_0, x_2d_1, x_2d_2, x_2d_3

@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper], config=helion.Config(block_sizes=[1, 4, 32], loop_orders=[[2, 0, 1]], flatten_loops=[False], num_warps=4, num_stages=7, indexing='pointer'))
def helion_d8_gelu_fwd(
    x_A1, x_A2, x_B1, x_B2, x_2d,
):
    # TODO: Some asserts are curently not getting compiled

    # assert len(x_A1.shape) == 3, "Input must be a 3D tensor [B, N, C]"
    # assert len(x_2d.shape) == 4, "Input must be a 4D tensor [B, N, 2, 2*C]"
    # assert x_2d.shape[2] == 2, "3rd dim must be size 2"
    # assert x_A1.shape[0] == x_2d.shape[0], "Inputs must have compatible size"
    # assert x_A1.shape[1] == x_2d.shape[1], "Inputs must have compatible size"
    # assert 2*x_A1.shape[2] == x_2d.shape[3], "Inputs must have compatible size"
    # for x, y in zip(x_A1.shape, x_A2.shape):
    #     assert x == y, "Inputs must have compatible size"
    # for x, y in zip(x_A1.shape, x_B1.shape):
    #     assert x == y, "Inputs must have compatible size"
    # for x, y in zip(x_A1.shape, x_B2.shape):
    #     assert x == y, "Inputs must have compatible size"
    # assert 2*x_A1.shape[2] == x_2d.shape[3], "Inputs must have compatible size"
    # assert x_A1.is_contiguous(), "Input must be contiguous"
    # assert x_A2.is_contiguous(), "Input must be contiguous"
    # assert x_B1.is_contiguous(), "Input must be contiguous"
    # assert x_B2.is_contiguous(), "Input must be contiguous"
    # assert x_2d.is_contiguous(), "Input must be contiguous"

    B, N, C = x_A1.shape
    y_A1 = torch.empty_like(x_A1)
    y_A2 = torch.empty_like(x_A2)
    y_B1 = torch.empty_like(x_B1)
    y_B2 = torch.empty_like(x_B2)
    y_2d = torch.empty_like(x_2d)
    
    x_2d_0 = x_2d[:,:,0,:C]
    x_2d_1 = x_2d[:,:,0,C:]
    x_2d_2 = x_2d[:,:,1,:C]
    x_2d_3 = x_2d[:,:,1,C:]
    
    y_2d_0 = torch.empty_like(x_2d_0)
    y_2d_1 = torch.empty_like(x_2d_1)
    y_2d_2 = torch.empty_like(x_2d_2)
    y_2d_3 = torch.empty_like(x_2d_3)

    # Total number of C elements across all B,N dimensions    
    for tile_b, tile_n, tile_c in hl.tile([B, N, C]):
        x_1d_0_chunk = x_A1[tile_b, tile_n, tile_c]
        x_1d_1_chunk = x_A2[tile_b, tile_n, tile_c]
        x_1d_2_chunk = x_B1[tile_b, tile_n, tile_c]
        x_1d_3_chunk = x_B2[tile_b, tile_n, tile_c]
        x_2d_0_chunk = x_2d_0[tile_b, tile_n, tile_c]
        x_2d_1_chunk = x_2d_1[tile_b, tile_n, tile_c]
        x_2d_2_chunk = x_2d_2[tile_b, tile_n, tile_c]
        x_2d_3_chunk = x_2d_3[tile_b, tile_n, tile_c]
        
        # isotypic to regular

        x_1d_0_chunk, x_1d_1_chunk, x_1d_2_chunk, x_1d_3_chunk, x_2d_0_chunk, x_2d_1_chunk, x_2d_2_chunk, x_2d_3_chunk = isotypic_to_regular(x_1d_0_chunk, x_1d_1_chunk, x_1d_2_chunk, x_1d_3_chunk, x_2d_0_chunk, x_2d_1_chunk, x_2d_2_chunk, x_2d_3_chunk)

        # gelu
        x_1d_0_chunk = gelu(x_1d_0_chunk)
        x_1d_1_chunk = gelu(x_1d_1_chunk)
        x_1d_2_chunk = gelu(x_1d_2_chunk)
        x_1d_3_chunk = gelu(x_1d_3_chunk)
        x_2d_0_chunk = gelu(x_2d_0_chunk)
        x_2d_1_chunk = gelu(x_2d_1_chunk)
        x_2d_2_chunk = gelu(x_2d_2_chunk)
        x_2d_3_chunk = gelu(x_2d_3_chunk)
    
        # regular to isotypic
        x_1d_0_chunk, x_1d_1_chunk, x_1d_2_chunk, x_1d_3_chunk, x_2d_0_chunk, x_2d_1_chunk, x_2d_2_chunk, x_2d_3_chunk = regular_to_isotypic(x_1d_0_chunk, x_1d_1_chunk, x_1d_2_chunk, x_1d_3_chunk, x_2d_0_chunk, x_2d_1_chunk, x_2d_2_chunk, x_2d_3_chunk)
        
        y_A1[tile_b, tile_n, tile_c] = x_1d_0_chunk
        y_A2[tile_b, tile_n, tile_c] = x_1d_1_chunk
        y_B1[tile_b, tile_n, tile_c] = x_1d_2_chunk
        y_B2[tile_b, tile_n, tile_c] = x_1d_3_chunk
        y_2d_0[tile_b, tile_n, tile_c] = x_2d_0_chunk
        y_2d_1[tile_b, tile_n, tile_c] = x_2d_1_chunk
        y_2d_2[tile_b, tile_n, tile_c] = x_2d_2_chunk
        y_2d_3[tile_b, tile_n, tile_c] = x_2d_3_chunk

    y_2d[:, :, 0, :C] = y_2d_0
    y_2d[:, :, 0, C:] = y_2d_1
    y_2d[:, :, 1, :C] = y_2d_2
    y_2d[:, :, 1, C:] = y_2d_3
    return y_A1, y_A2, y_B1, y_B2, y_2d

## Note that this has been tuned for a RTX A5500
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper], config=helion.Config(block_sizes=[8, 8, 4], loop_orders=[[1, 0, 2]], flatten_loops=[True], num_warps=1, num_stages=6, indexing='block_ptr'))
def helion_d8_gelu_bwd(g_A1, g_A2, g_B1, g_B2, g_2d, x_A1, x_A2, x_B1, x_B2, x_2d):
    # TODO: Some asserts are curently not getting compiled

    # assert len(x_A1.shape) == 3, "Input must be a 3D tensor [B, N, C]"
    # assert len(x_2d.shape) == 4, "Input must be a 4D tensor [B, N, 2, 2*C]"
    # assert x_2d.shape[2] == 2, "3rd dim must be size 2"
    # assert x_A1.shape[0] == x_2d.shape[0], "Inputs must have compatible size"
    # assert x_A1.shape[1] == x_2d.shape[1], "Inputs must have compatible size"
    # assert 2*x_A1.shape[2] == x_2d.shape[3], "Inputs must have compatible size"
    # for x, y in zip(x_A1.shape, x_A2.shape):
    #     assert x == y, "Inputs must have compatible size"
    # for x, y in zip(x_A1.shape, x_B1.shape):
    #     assert x == y, "Inputs must have compatible size"
    # for x, y in zip(x_A1.shape, x_B2.shape):
    #     assert x == y, "Inputs must have compatible size"
    # assert 2*x_A1.shape[2] == x_2d.shape[3], "Inputs must have compatible size"
    # assert x_A1.is_contiguous(), "Input must be contiguous"
    # assert x_A2.is_contiguous(), "Input must be contiguous"
    # assert x_B1.is_contiguous(), "Input must be contiguous"
    # assert x_B2.is_contiguous(), "Input must be contiguous"
    # assert x_2d.is_contiguous(), "Input must be contiguous"
    # for g, x in zip(g_A1.shape, x_A1.shape):
    #     assert g == x, "Inputs must have compatible size"
    # for g, x in zip(g_A2.shape, x_A2.shape):
    #     assert g == x, "Inputs must have compatible size"
    # for g, x in zip(g_B1.shape, x_B1.shape):
    #     assert g == x, "Inputs must have compatible size"
    # for g, x in zip(g_B2.shape, x_B2.shape):
    #     assert g == x, "Inputs must have compatible size"
    # for g, x in zip(g_2d.shape, x_2d.shape):
    #     assert g == x, "Inputs must have compatible size"
    

    B, N, C = x_A1.shape

    g_in_A1 = torch.empty_like(g_A1)
    g_in_A2 = torch.empty_like(g_A2)
    g_in_B1 = torch.empty_like(g_B1)
    g_in_B2 = torch.empty_like(g_B2)
    g_in_2d = torch.empty_like(g_2d)
    
    x_2d_0 = x_2d[:, :, 0, :C]
    x_2d_1 = x_2d[:, :, 0, C:]
    x_2d_2 = x_2d[:, :, 1, :C]
    x_2d_3 = x_2d[:, :, 1, C:]
    
    g_2d_0 = g_2d[:, :, 0, :C]
    g_2d_1 = g_2d[:, :, 0, C:]
    g_2d_2 = g_2d[:, :, 1, :C]
    g_2d_3 = g_2d[:, :, 1, C:]
    
    g_in_2d_0 = g_in_2d[:, :, 0, :C]  
    g_in_2d_1 = g_in_2d[:, :, 0, C:]
    g_in_2d_2 = g_in_2d[:, :, 1, :C]
    g_in_2d_3 = g_in_2d[:, :, 1, C:]
    
    for tile_b, tile_n, tile_c in hl.tile([B, N, C]):
        
        x_1d_0_chunk = x_A1[tile_b, tile_n, tile_c]
        x_1d_1_chunk = x_A2[tile_b, tile_n, tile_c]
        x_1d_2_chunk = x_B1[tile_b, tile_n, tile_c]
        x_1d_3_chunk = x_B2[tile_b, tile_n, tile_c]
        x_2d_0_chunk = x_2d_0[tile_b, tile_n, tile_c]
        x_2d_1_chunk = x_2d_1[tile_b, tile_n, tile_c]
        x_2d_2_chunk = x_2d_2[tile_b, tile_n, tile_c]
        x_2d_3_chunk = x_2d_3[tile_b, tile_n, tile_c]
    
        g_1d_0_chunk = g_A1[tile_b, tile_n, tile_c]
        g_1d_1_chunk = g_A2[tile_b, tile_n, tile_c]
        g_1d_2_chunk = g_B1[tile_b, tile_n, tile_c]
        g_1d_3_chunk = g_B2[tile_b, tile_n, tile_c]
        g_2d_0_chunk = g_2d_0[tile_b, tile_n, tile_c]
        g_2d_1_chunk = g_2d_1[tile_b, tile_n, tile_c]
        g_2d_2_chunk = g_2d_2[tile_b, tile_n, tile_c]
        g_2d_3_chunk = g_2d_3[tile_b, tile_n, tile_c]
        

        # inv FFT
        x_1d_0_chunk, x_1d_1_chunk, x_1d_2_chunk, x_1d_3_chunk, x_2d_0_chunk, x_2d_1_chunk, x_2d_2_chunk, x_2d_3_chunk = isotypic_to_regular(x_1d_0_chunk, x_1d_1_chunk, x_1d_2_chunk, x_1d_3_chunk, x_2d_0_chunk, x_2d_1_chunk, x_2d_2_chunk, x_2d_3_chunk)
        
        # inv FFT (FFT tranposed)
        g_1d_0_chunk, g_1d_1_chunk, g_1d_2_chunk, g_1d_3_chunk, g_2d_0_chunk, g_2d_1_chunk, g_2d_2_chunk, g_2d_3_chunk = regular_to_isotypic(g_1d_0_chunk, g_1d_1_chunk, g_1d_2_chunk, g_1d_3_chunk, g_2d_0_chunk, g_2d_1_chunk, g_2d_2_chunk, g_2d_3_chunk)
        
        
        # gelu grid
        
        x_1d_0_chunk = gelu_grad(x_1d_0_chunk)
        x_1d_1_chunk = gelu_grad(x_1d_1_chunk)
        x_1d_2_chunk = gelu_grad(x_1d_2_chunk)
        x_1d_3_chunk = gelu_grad(x_1d_3_chunk)
        x_2d_0_chunk = gelu_grad(x_2d_0_chunk)
        x_2d_1_chunk = gelu_grad(x_2d_1_chunk)
        x_2d_2_chunk = gelu_grad(x_2d_2_chunk)
        x_2d_3_chunk = gelu_grad(x_2d_3_chunk)
        
        # multiply according to chain rule
        g_1d_0_chunk = g_1d_0_chunk * x_1d_0_chunk
        g_1d_1_chunk = g_1d_1_chunk * x_1d_1_chunk
        g_1d_2_chunk = g_1d_2_chunk * x_1d_2_chunk
        g_1d_3_chunk = g_1d_3_chunk * x_1d_3_chunk
        g_2d_0_chunk = g_2d_0_chunk * x_2d_0_chunk
        g_2d_1_chunk = g_2d_1_chunk * x_2d_1_chunk
        
        # FFT (inv FFT tranposed)
        g_1d_0_chunk, g_1d_1_chunk, g_1d_2_chunk, g_1d_3_chunk, g_2d_0_chunk, g_2d_1_chunk, g_2d_2_chunk, g_2d_3_chunk = regular_to_isotypic(g_1d_0_chunk, g_1d_1_chunk, g_1d_2_chunk, g_1d_3_chunk, g_2d_0_chunk, g_2d_1_chunk, g_2d_2_chunk, g_2d_3_chunk)
        
        g_in_A1[tile_b, tile_n, tile_c] = g_1d_0_chunk
        g_in_A2[tile_b, tile_n, tile_c] = g_1d_1_chunk
        g_in_B1[tile_b, tile_n, tile_c] = g_1d_2_chunk
        g_in_B2[tile_b, tile_n, tile_c] = g_1d_3_chunk
        g_in_2d_0[tile_b, tile_n, tile_c] = g_2d_0_chunk
        g_in_2d_1[tile_b, tile_n, tile_c] = g_2d_1_chunk
        g_in_2d_2[tile_b, tile_n, tile_c] = g_2d_2_chunk
        g_in_2d_3[tile_b, tile_n, tile_c] = g_2d_3_chunk
        
    g_in_2d[:, :, 0, :C] = g_in_2d_0
    g_in_2d[:, :, 0, C:] = g_in_2d_1
    g_in_2d[:, :, 1, :C] = g_in_2d_2
    g_in_2d[:, :, 1, C:] = g_in_2d_3

    return g_in_A1, g_in_A2, g_in_B1, g_in_B2, g_in_2d

class HelionGeluD8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_A1, x_A2, x_B1, x_B2, x_2d):
        x_A1, x_A2, x_B1, x_B2, x_2d = (
            x_A1.contiguous(), x_A2.contiguous(),
            x_B1.contiguous(), x_B2.contiguous(),
            x_2d.contiguous()
        )
        ctx.save_for_backward(x_A1, x_A2, x_B1, x_B2, x_2d)
        return helion_d8_gelu_fwd(x_A1, x_A2, x_B1, x_B2, x_2d)
    @staticmethod
    def backward(ctx, g_A1, g_A2, g_B1, g_B2, g_2d):
        x_A1, x_A2, x_B1, x_B2, x_2d = ctx.saved_tensors
        return helion_d8_gelu_bwd(g_A1.contiguous(),
                           g_A2.contiguous(),
                           g_B1.contiguous(),
                           g_B2.contiguous(),
                           g_2d.contiguous(),
                           x_A1.contiguous(),
                           x_A2.contiguous(),
                           x_B1.contiguous(),
                           x_B2.contiguous(),
                           x_2d.contiguous())

class HelionGeluD8(torch.nn.Module):
    def forward(self, xs):
        return HelionGeluD8Function.apply(xs[0], xs[1], xs[2], xs[3], xs[4])



def test_fwd():
    device = "cuda"
    # Create a sample tensor
    B, N, C = 8, 196, 64
    x = torch.randn(2*B*N*4*C, device=device)
    x_1d, x_2d = torch.chunk(x, 2)
    x_1d = x_1d.reshape(B, N, 4, C)
    x_A1, x_A2, x_B1, x_B2 = x_1d[:, :, 0], x_1d[:, :, 1], x_1d[:, :, 2], x_1d[:, :, 3]
    x_2d = x_2d.reshape(B, N, 2, 2*C)
    # print("Input tensors:")
    # print(x_1d)
    # print(x_2d)
    
    # Make a copy for the reference implementation
    x_ref_A1 = x_A1.clone()
    x_ref_A2 = x_A2.clone()
    x_ref_B1 = x_B1.clone()
    x_ref_B2 = x_B2.clone()
    x_ref_2d = x_2d.clone()
    
    # Apply the triton implementation
    x_A1, x_A2, x_B1, x_B2, x_2d = d8_gelu_fwd(x_A1.contiguous(), x_A2.contiguous(), x_B1.contiguous(), x_B2.contiguous(), x_2d.contiguous())
    # print("Output tensors:")
    # print(x_1d)
    # print(x_2d)

    
    # Verify the results with a PyTorch implementation
    from octic_vits.d8_layers import (
        GeluD8,
    )
    gelu_d8 = GeluD8()

    @torch.compile
    def reference_implementation(x_A1, x_A2, x_B1, x_B2, x_2d, return_split=True):
        C = x_A1.shape[-1]
        xs_D8 = gelu_d8([
            x_A1,
            x_A2,
            x_B1,
            x_B2,
            x_2d[..., 0, :C],
            x_2d[..., 1, :C],
            x_2d[..., 0, C:],
            x_2d[..., 1, C:]
        ])
        if return_split:
            return (
                xs_D8[0],
                xs_D8[1],
                xs_D8[2],
                xs_D8[3],
                torch.cat((
                    torch.stack(xs_D8[4:6], dim=-2),
                    torch.stack(xs_D8[6:], dim=-2),
                ), dim=-1)
            )
        return xs_D8
    
    x_ref_A1, x_ref_A2, x_ref_B1, x_ref_B2, x_ref_2d = reference_implementation(
        x_ref_A1, x_ref_A2, x_ref_B1, x_ref_B2, x_ref_2d)
    # print("Reference result:")
    # print(x_ref_1d)
    # print(x_ref_2d)
    
    # Check if the results match
    print("Results for A1 feats match:",
          torch.allclose(x_ref_A1, x_A1))
    print("Results for A2 feats match:",
          torch.allclose(x_ref_A2, x_A2))
    print("Results for B1 feats match:",
          torch.allclose(x_ref_B1, x_B1))
    print("Results for B2 feats match:",
          torch.allclose(x_ref_B2, x_B2))
    print("Results for 2d feats match:",
          torch.allclose(x_ref_2d, x_2d))
    
    # Performance comparison
    import time
    
    # Larger tensor for benchmarking
    B, N, C = 16, 196, 32
    
    # Benchmark Triton implementation
    x_triton_1d = torch.randn(B, N, 4, C, device='cuda')
    x_triton_2d = torch.randn(B, N, 2, 2*C, device='cuda')

    x_d8 = (
        x_triton_1d[:, :, 0].clone(),
        x_triton_1d[:, :, 1].clone(),
        x_triton_1d[:, :, 2].clone(),
        x_triton_1d[:, :, 3].clone(),
        x_triton_2d[:, :, 0, :C].clone(),
        x_triton_2d[:, :, 1, :C].clone(),
        x_triton_2d[:, :, 0, C:].clone(),
        x_triton_2d[:, :, 1, C:].clone()
    )

    x_triton_A1 = x_d8[0].clone()
    x_triton_A2 = x_d8[1].clone()
    x_triton_B1 = x_d8[2].clone()
    x_triton_B2 = x_d8[3].clone()

    compile_gelu_d8 = torch.compile(GeluD8())
    
    with torch.inference_mode():
        # Warmup
        x_triton_copy_A1 = x_triton_A1.clone()
        x_triton_copy_A2 = x_triton_A2.clone()
        x_triton_copy_B1 = x_triton_B1.clone()
        x_triton_copy_B2 = x_triton_B2.clone()
        x_triton_copy_2d = x_triton_2d.clone()
        for _ in range(10):
            _ = d8_gelu_fwd(x_triton_copy_A1, x_triton_copy_A2, x_triton_copy_B1, x_triton_copy_B2, x_triton_copy_2d)
        
        torch.cuda.synchronize()
        start = time.time()
        x_triton_copy_A1 = x_triton_A1.clone()
        x_triton_copy_A2 = x_triton_A2.clone()
        x_triton_copy_B1 = x_triton_B1.clone()
        x_triton_copy_B2 = x_triton_B2.clone()
        x_triton_copy_2d = x_triton_2d.clone()
        for _ in range(100):
            _ = d8_gelu_fwd(x_triton_copy_A1, x_triton_copy_A2, x_triton_copy_B1, x_triton_copy_B2, x_triton_copy_2d)
        torch.cuda.synchronize()
        triton_time = time.time() - start
        
        # Warmup
        x_triton_copy_A1 = x_triton_A1.clone()
        x_triton_copy_A2 = x_triton_A2.clone()
        x_triton_copy_B1 = x_triton_B1.clone()
        x_triton_copy_B2 = x_triton_B2.clone()
        x_triton_copy_2d = x_triton_2d.clone()
        for _ in range(10):
            _ = reference_implementation(x_triton_copy_A1,
                                         x_triton_copy_A2,
                                         x_triton_copy_B1,
                                         x_triton_copy_B2,
                                         x_triton_copy_2d,
                                         return_split=True)
        
        # Benchmark PyTorch implementation
        torch.cuda.synchronize()
        start = time.time()
        x_triton_copy_A1 = x_triton_A1.clone()
        x_triton_copy_A2 = x_triton_A2.clone()
        x_triton_copy_B1 = x_triton_B1.clone()
        x_triton_copy_B2 = x_triton_B2.clone()
        x_triton_copy_2d = x_triton_2d.clone()
        for _ in range(100):
            _ = reference_implementation(x_triton_copy_A1,
                                         x_triton_copy_A2,
                                         x_triton_copy_B1,
                                         x_triton_copy_B2,
                                         x_triton_copy_2d,
                                         return_split=True)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start

        # Warmup
        x_d8_copy = [x.clone() for x in x_d8]
        for _ in range(10):
            _ = compile_gelu_d8(x_d8_copy)
        
        # Benchmark original PyTorch implementation
        torch.cuda.synchronize()
        start = time.time()
        x_d8_copy = [x.clone() for x in x_d8]
        for _ in range(100):
            _ = compile_gelu_d8(x_d8_copy)
        torch.cuda.synchronize()
        original_d8_gelu_time = time.time() - start
        
    print(f"Triton implementation: {triton_time:.6f} seconds")
    print(f"PyTorch implementation: {pytorch_time:.6f} seconds")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")
    print("===")
    print(f"GeluD8 previous implementation: {original_d8_gelu_time:.6f} seconds")
    print(f"Speedup: {original_d8_gelu_time / triton_time:.2f}x")

def test_bwd():
    device = "cuda"
    # Create a sample tensor
    B, N, C = 16, 196, 32
    xs = [
        torch.randn(B, N, C, requires_grad=True, device=device)
        for _ in range(8)
    ]
    x_1d = torch.stack(xs[:4], dim=2).clone().detach().requires_grad_(True)
    x_2d = torch.cat(
        [torch.stack(xs[4:6], dim=2), torch.stack(xs[6:], dim=2)],
        dim=-1,
    ).clone().detach().requires_grad_(True)

    triton_gelu_d8 = TritonGeluD8()
    y_A1, y_A2, y_B1, y_B2, y_2d = triton_gelu_d8([x_1d[:, :, 0], x_1d[:, :, 1], x_1d[:, :, 2], x_1d[:, :, 3], x_2d])

    cat_all = torch.cat([
        y_A1.unsqueeze(2),
        y_A2.unsqueeze(2),
        y_B1.unsqueeze(2), 
        y_B2.unsqueeze(2),
        y_2d.reshape([B, N, 2, 2, C]).transpose(2, 3).reshape([B, N, 4, C])
    ], dim=2)
    g = torch.randn_like(cat_all).detach()
    g_copy = g.clone()

    cat_all.backward(g, retain_graph=True)
    grad_cat_all = torch.cat([
        x_1d.grad,
        x_2d.grad.reshape([B, N, 2, 2, C]).transpose(2, 3).reshape([B, N, 4, C])
    ], dim=2)

    from octic_vits.d8_layers import (
        GeluD8,
    )
    gelu_d8 = GeluD8()
    ys = gelu_d8(xs)
    stack_all = torch.stack(ys, dim=2)
    stack_all.backward(g_copy, retain_graph=True)
    grad_stack_all = torch.stack([x.grad for x in xs], dim=2)

    # print(cat_all)
    # print(stack_all)

    # print(grad_cat_all)
    # print(grad_stack_all)

    print("Gelu output all close:",
          torch.allclose(cat_all, stack_all, rtol=1e-5, atol=1e-5))
    print("Gelu gradient all close:",
          torch.allclose(grad_cat_all, grad_stack_all, rtol=1e-5, atol=1e-5))

    from triton.testing import do_bench
    
    print(f"PyTorch Fwd Time: {do_bench(lambda: gelu_d8(xs))} ms")
    print(f"PyTorch Bwd Time: {do_bench(lambda: stack_all.backward(g_copy, retain_graph=True))} ms")

    print(f"Triton Fwd Time: {do_bench(lambda: triton_gelu_d8([x_1d[:, :, 0], x_1d[:, :, 1], x_1d[:, :, 2], x_1d[:, :, 3], x_2d]))} ms")
    print(f"Triton Bwd Time: {do_bench(lambda: cat_all.backward(g, retain_graph=True))} ms")

    
    ### Helion ###
    
    helion_gelu_d8 = HelionGeluD8()
    y_A1_helion, y_A2_helion, y_B1_helion, y_B2_helion, y_2d_helion = helion_gelu_d8([x_1d[:, :, 0], x_1d[:, :, 1], x_1d[:, :, 2], x_1d[:, :, 3], x_2d])

    cat_all_helion = torch.cat([
        y_A1_helion.unsqueeze(2),
        y_A2_helion.unsqueeze(2),
        y_B1_helion.unsqueeze(2), 
        y_B2_helion.unsqueeze(2),
        y_2d_helion.reshape([B, N, 2, 2, C]).transpose(2, 3).reshape([B, N, 4, C])
    ], dim=2)
    
    g_helion = torch.randn_like(cat_all_helion).detach()

    cat_all_helion.backward(g_helion, retain_graph=True)
    grad_cat_all_helion = torch.cat([
        x_1d.grad,
        x_2d.grad.reshape([B, N, 2, 2, C]).transpose(2, 3).reshape([B, N, 4, C])
    ], dim=2)
    
    print("Helion Gelu output all close:",
          torch.allclose(cat_all_helion, cat_all, rtol=1e-5, atol=1e-5))
    print("Helion Gelu gradient all close:",
          torch.allclose(grad_cat_all_helion, grad_stack_all, rtol=1e-5, atol=1e-5))
    
    print(f"Helion Fwd Time: {do_bench(lambda: helion_gelu_d8([x_1d[:, :, 0], x_1d[:, :, 1], x_1d[:, :, 2], x_1d[:, :, 3], x_2d]))} ms")
    print(f"Helion Bwd Time: {do_bench(lambda: cat_all_helion.backward(g_helion, retain_graph=True))} ms")
    
    
if __name__ == "__main__":
    # test_fwd()
    test_bwd()