
import pytest
from functools import partial
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    idxs = gl.arange(0, XBLOCK, layout)

    offs = start + idxs
    in_ptrs = in_ptr + offs
    out_ptrs = out_ptr + offs
    mask = offs < xnumel

    val = gl.load(in_ptrs, mask=mask)
    gl.store(out_ptrs, val, mask=mask)


def memcpy_1d(input, output, XBLOCK, layout, num_warps):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)


@pytest.mark.parametrize("XBLOCK", [64])
@pytest.mark.parametrize("xnumel", [40, 500])
@pytest.mark.parametrize("num_warps", [4])
def test_memcpy_1d(XBLOCK, xnumel, num_warps):
    # python -m pytest ld_triton/ops/intro/layouts.py
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    layout = gl.BlockedLayout([1], [32], [num_warps], [0])
    memcpy_1d(input, output, XBLOCK, layout, num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


def bench_memcpy_impl(input, output, impl):
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(input, ms)


def bench_memcpy(impl):
    torch.manual_seed(0)
    xnumel = 2 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    return bench_memcpy_impl(input, output, impl)

if __name__ == "__main__":
    # python ld_triton/ops/intro/layouts.py
    print("R vs. Throughput")
    print("================")
    XBLOCK = 2048
    num_warps = 4
    kernel = partial(memcpy_1d, XBLOCK=XBLOCK, num_warps=num_warps)
    compiled_kernels = []
    for i in range(0, 5):
        R = 2**i
        layout = gl.BlockedLayout([R], [32], [num_warps], [0])
        impl = partial(kernel, layout=layout)
        compiled_kernel, throughput = bench_memcpy(impl)
        compiled_kernels.append((R, compiled_kernel))
        print(f"R={R:<3} {throughput:.3f} TB/s")
    print()
