import numpy as np
from fvcore.nn.jit_handles import get_shape
from fvcore.nn import FlopCountAnalysis
from typing import List, Any

def add_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Addition FLOPs: Element-wise addition (1 operation per element)."""
    return np.prod(get_shape(outputs[0]))

def mul_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Multiplication FLOPs: Element-wise multiplication (1 operation per element)."""
    return np.prod(get_shape(outputs[0]))

def sub_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Subtraction FLOPs: Element-wise subtraction (1 operation per element)."""
    return np.prod(get_shape(outputs[0]))

def neg_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Negation FLOPs: Element-wise negation (1 operation per element)."""
    return np.prod(get_shape(outputs[0]))

def flip_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Flip FLOPs: Memory operation, typically counted as zero FLOPs."""
    return 0.0  # Flipping does not perform mathematical computations

def sigmoid_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Sigmoid FLOPs: Approximated as 4 FLOPs per element (exp + add + div).
       Source https://machinethink.net/blog/how-fast-is-my-model/
    """
    return 4 * np.prod(get_shape(outputs[0]))

def mean_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Mean FLOPs: Sum + division over the reduced dimensions."""
    return np.prod(get_shape(inputs[0]))  # Sum + division

def var_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Variance FLOPs: Requires mean computation, subtraction, squaring, sum, and division."""
    input_shape = get_shape(inputs[0])
    reduction_dim = np.prod(input_shape) / np.prod(get_shape(outputs[0]))
    return 2 * reduction_dim  # Mean + (x - mean)^2 sum + division

def sqrt_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Square root FLOPs: Approximate as 4 FLOPs per element (Newton-Raphson iterations)."""
    return 4 * np.prod(get_shape(outputs[0]))

def div_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Division FLOPs: Element-wise division (1 operation per element)."""
    return np.prod(get_shape(outputs[0]))

def softmax_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Softmax FLOPs: Exponentiation + sum + division."""
    input_shape = get_shape(inputs[0])
    num_elements = np.prod(input_shape)
    return 5 * num_elements  # exp(x) + sum + division per element

def gelu_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """GELU FLOPs: Uses tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    return 8 * np.prod(get_shape(outputs[0]))  # Multiplication, addition, tanh, exponentiation

def pow_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Power FLOPs: Uses exponentiation (log + exp), approximated as 5 FLOPs per element."""
    return 5 * np.prod(get_shape(outputs[0]))

def hstack_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Horizontal stacking FLOPs: Memory operation, typically counted as zero FLOPs."""
    return 0.0  # Concatenation does not perform mathematical computations

def cartesian_prod_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Cartesian product FLOPs: Each element in inputs combined with every element in another input."""
    # Such a small part of the entire thing, will be inconsequential
    return 0.0

def clone_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Clone FLOPs: Memory operation, typically counted as zero FLOPs."""
    return 0.0  # Cloning does not perform mathematical computations

def rand_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Random number generation FLOPs: Typically counted as zero FLOPs."""
    return 0.0  # Random generation does not perform mathematical computations

def argsort_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Argsort FLOPs: Sort operation, approximated based on the sorting algorithm complexity."""
    input_shape = get_shape(inputs[0])
    n_elements = np.prod(input_shape)
    return n_elements * np.log(n_elements)  # Rough estimate for sorting complexity

def lt_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Less than FLOPs: Element-wise comparison (1 operation per element)."""
    return np.prod(get_shape(outputs[0]))  # One comparison per output element

def python_op_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Custom Python operation FLOPs: Define based on the specific operation's complexity."""
    # This will depend on the specific computation happening in your Python operation.
    # Placeholder; adjust according to the actual operation.
    return np.prod(get_shape(outputs[0]))  # Modify as needed based on operation

def repeat_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """Repeat FLOPs: Memory operation, typically counted as zero FLOPs."""
    return 0.0  # Repeating does not perform mathematical computations

def rotate_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """
    Rotation FLOPs: Memory access operations for tensor rotation.
    
    While rotation doesn't involve arithmetic operations on values,
    it requires accessing and moving each element once, which has
    a computational cost. We count 1 operation per element.
    """
    return np.prod(get_shape(outputs[0]))

def scaled_dot_product_attention_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    # Formulas derived from: https://jax-ml.github.io/scaling-book/transformers/
    """
    Count FLOPs for the scaled dot product attention operation.
    """
    q, k, v = inputs[0], inputs[1], inputs[2]
    assert get_shape(q) == get_shape(k)

    # Even though they are the same we do this way to align notation with https://jax-ml.github.io/scaling-book/transformers/
    # Deviating from scaling book by:
    # - Counting only forward pass (i.e. only a third of the "train FLOPs")
    # - Counting MACs instead of FLOPs, so skipping a factor of 2 (in-line with fvcore convention)
    # - Order of the shape is slightly different but names of variables are the same (e.g. N=number of query heads)
    B, N, T, H = get_shape(q)
    _, K, S, _ = get_shape(k)
    G = N // K # 1 for us because N==K
    approx_flops = 12*B*(T**2)*N*H
    approx_flops /=3 # Because we only count forward pass
    approx_flops /=2 # Becuase we count MACs and not FLOPs (strictly speaking)
    return approx_flops

def sum_flop_jit(inputs: List[Any], outputs: List[Any]) -> float:
    """
    Sum FLOPs: 
    For a tensor of shape (d1, d2, ..., dn), summing over k dimensions requires
    one addition per element except the first in each reduction. So total FLOPs 
    = number of elements in the reduced tensor - 1 per reduction operation.
    """
    input_shape = get_shape(inputs[0])
    output_shape = get_shape(outputs[0])
    input_elems = np.prod(input_shape)
    output_elems = np.prod(output_shape)

    # Each output element results from summing multiple input elements.
    # Total additions = input_elems - output_elems
    return float(input_elems - output_elems)

def jits(flops: FlopCountAnalysis):
    """ 
    Decorate FlopCounter by adding support for more ops.
    Source for inspiration: https://github.com/facebookresearch/fvcore/issues/129
    """
    flops.set_op_handle("aten::add", add_flop_jit)
    flops.set_op_handle("aten::mul", mul_flop_jit)
    flops.set_op_handle("aten::sub", sub_flop_jit)
    flops.set_op_handle("aten::neg", neg_flop_jit)
    flops.set_op_handle("aten::flip", flip_flop_jit)
    flops.set_op_handle("aten::sigmoid", sigmoid_flop_jit)
    flops.set_op_handle("aten::mean", mean_flop_jit)
    flops.set_op_handle("aten::var", var_flop_jit)
    flops.set_op_handle("aten::sqrt", sqrt_flop_jit)
    flops.set_op_handle("aten::div", div_flop_jit)
    flops.set_op_handle("aten::softmax", softmax_flop_jit)
    flops.set_op_handle("aten::gelu", gelu_flop_jit)
    flops.set_op_handle("aten::pow", pow_flop_jit)
    flops.set_op_handle("aten::hstack", hstack_flop_jit)

    flops.set_op_handle("aten::cartesian_prod", cartesian_prod_flop_jit)
    flops.set_op_handle("aten::clone", clone_flop_jit)
    flops.set_op_handle("aten::rand", rand_flop_jit)
    flops.set_op_handle("aten::argsort", argsort_flop_jit)
    flops.set_op_handle("aten::lt", lt_flop_jit)
    flops.set_op_handle("prim::PythonOp.cuRoPE2D_func", python_op_flop_jit)
    flops.set_op_handle("aten::scaled_dot_product_attention", scaled_dot_product_attention_flop_jit)
    flops.set_op_handle("aten::repeat", repeat_flop_jit)
    flops.set_op_handle("aten::rot90", rotate_flop_jit)
    flops.set_op_handle("aten::sum", sum_flop_jit)

    # Dubbelkolla ett linjärt lager (flops)
    # Kolla relationen mellan flops i ett linjärt lager vs. attention (hur mycket av nätets flops är attention vs. linjär)
    # Kolla varför loss hög vid initialisering
    # Öka storlek -> bencha throughput med max-autotune