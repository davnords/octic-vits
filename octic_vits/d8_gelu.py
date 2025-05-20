import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# TODO: better optimization of BLOCK_SIZE in kernels
# TODO: seemingly, torch.compile does not tune BLOCK_SIZE currently (?)
# at least they are not output in the logs.

# TODO: better handling of constants
# SQRT2_OVER_2 = math.sqrt(2) / 2
# SQRT2_OVER_4 = math.sqrt(2) / 4
# ONE_OVER_SQRT2PI = 1 / math.sqrt(2*math.pi)

@triton.jit
def tl_gelu(x, SQRT2_OVER_2: tl.constexpr=0.7071067812):

    # Implementation supporting mixed precision
    # tl.math.erf is not supported in fp16
    x_fp32 = x.to(tl.float32)  
    cdf = 0.5 * (1.0 + tl.math.erf(SQRT2_OVER_2 * x_fp32))  
    return x * cdf.to(x.dtype)

@triton.jit
def tl_gelu_grad(x,
                 SQRT2_OVER_2: tl.constexpr=0.7071067812,
                 ONE_OVER_SQRT2PI: tl.constexpr=0.3989422804):
    
    # Fp16 fix 
    x_fp32 = x.to(tl.float32)  
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

    cat_all.backward(g)
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
    stack_all.backward(g_copy)
    grad_stack_all = torch.stack([x.grad for x in xs], dim=2)

    # print(cat_all)
    # print(stack_all)

    # print(grad_cat_all)
    # print(grad_stack_all)

    print("Gelu output all close:",
          torch.allclose(cat_all, stack_all, rtol=1e-5, atol=1e-5))
    print("Gelu gradient all close:",
          torch.allclose(grad_cat_all, grad_stack_all, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    # test_fwd()
    test_bwd()