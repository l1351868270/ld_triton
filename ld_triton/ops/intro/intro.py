

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)


def copy_scalar(input, output):
    grid = (1, )
    copy_scalar_kernel[grid](input, output)


def test_copy_scalar():
    input = torch.tensor([42.0], device="cuda")
    output = torch.empty_like(input)
    copy_scalar(input, output)
    torch.testing.assert_close(input, output, atol=0.0, rtol=0)


@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)


def memcpy(input, output, XBLOCK,):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    memcpy_kernel[grid](
        input, output, xnumel, XBLOCK,
    )


@pytest.mark.parametrize("XBLOCK", [64])
@pytest.mark.parametrize("xnumel", [40, 500])
def test_memcpy(XBLOCK, xnumel):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


@triton.autotune(
    configs=[triton.Config({"XBLOCK": 2 ** i}, num_warps=1) for i in range(8, 14)],
    key=["xnumel"],
)
@gluon.jit
def memcpy_autotune_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK)


def memcpy_autotune(input, output):
    xnumel = input.numel()

    def grid(META):
        return (triton.cdiv(xnumel, META["XBLOCK"]), )

    memcpy_autotune_kernel[grid](input, output, xnumel)


if __name__ == "__main__":
    torch.manual_seed(0)
    xnumel = 2 << 30

    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    fn = lambda: memcpy_autotune(input, output)
    ms = triton.testing.do_bench(fn)
    gbytes = 2 * xnumel * input.element_size() >> 30
    print("Benchmarking memcpy")
    print("===================")
    print(f"gbytes:      {gbytes}")
    print(f"Time:        {ms:.2f} ms")
    print(f"Throughput: {gbytes / (ms * 1e-3):.2f} GB/s")

"""
H20:
gbytes:      16
Time:        50.29 ms
Throughput: 318.18 GB/s
"""
