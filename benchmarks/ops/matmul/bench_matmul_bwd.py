import torch
import triton

from ld_triton.ops.matmul.naive_matmul import naive_matmul
from ld_triton.ops.matmul.triton_matmul import triton_matmul


configs = []


configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider", 
        line_vals=['cublas', "naive", "triton"], 
        line_names=['cuBLAS', "Naive","Triton"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance-fp16",
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16, requires_grad=True)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16, requires_grad=True)
    dc = torch.randn((M, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        c = torch.matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: c.backward(dc, retain_graph=True), quantiles=quantiles)
    if provider == 'naive':
        c = naive_matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: c.backward(dc, retain_graph=True), quantiles=quantiles)
    if provider == 'triton':
        c = triton_matmul(a, b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: c.backward(dc, retain_graph=True), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)