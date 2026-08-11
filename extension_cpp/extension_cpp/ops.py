import torch
from torch import Tensor

__all__ = ["mymuladd", "myadd_out", "mygemm", "mygemm_small", "flash_attn"]


def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context)


@torch.library.register_fake("extension_cpp::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)


def mygemm(a: Tensor, b: Tensor) -> Tensor:
    """Computes C = A @ B^T with a CuTe HGEMM kernel (TN layout).

    A is (M, K) and B is (N, K), both float16/CUDA; returns C of shape (M, N).
    Auto-dispatches between large-tile (128×256) and small-tile (64×128)
    kernels based on grid occupancy.
    """
    return torch.ops.extension_cpp.mygemm.default(a, b)


def mygemm_small(a: Tensor, b: Tensor) -> Tensor:
    """Small-tile variant (BM=64, BN=128) — always uses the small tile kernel."""
    return torch.ops.extension_cpp.mygemm_small.default(a, b)


@torch.library.register_fake("extension_cpp::mygemm")
def _(a, b):
    torch._check(a.dim() == 2)
    torch._check(b.dim() == 2)
    torch._check(a.shape[1] == b.shape[1])
    torch._check(a.dtype == torch.float16)
    torch._check(b.dtype == torch.float16)
    torch._check(a.device == b.device)
    return a.new_empty((a.shape[0], b.shape[0]))


@torch.library.register_fake("extension_cpp::mygemm_small")
def _(a, b):
    torch._check(a.dim() == 2)
    torch._check(b.dim() == 2)
    torch._check(a.shape[1] == b.shape[1])
    torch._check(a.dtype == torch.float16)
    torch._check(b.dtype == torch.float16)
    torch._check(a.device == b.device)
    return a.new_empty((a.shape[0], b.shape[0]))


def flash_attn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Flash Attention with a CuTe kernel.

    Q, K, V are 4D tensors of shape (B, H, N, D) in float16/CUDA.
    Returns O of the same shape as Q.
    Auto-dispatches across head dims 16/32/64/128/256.
    """
    return torch.ops.extension_cpp.flash_attn.default(q, k, v)


@torch.library.register_fake("extension_cpp::flash_attn")
def _(q, k, v):
    torch._check(q.dim() == 4)
    torch._check(k.dim() == 4)
    torch._check(v.dim() == 4)
    torch._check(q.shape[0] == k.shape[0] == v.shape[0])  # B
    torch._check(q.shape[1] == k.shape[1] == v.shape[1])  # H
    torch._check(q.shape[3] == k.shape[3] == v.shape[3])  # D
    torch._check(q.dtype == torch.float16 or q.dtype == torch.bfloat16)
    torch._check(k.dtype == q.dtype)
    torch._check(v.dtype == q.dtype)
    torch._check(q.device == k.device == v.device)
    return q.new_empty(q.shape)
