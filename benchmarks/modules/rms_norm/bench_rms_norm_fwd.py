
import torch
import triton

from ld_triton.modules.rms_norm.llama3_rms_norm import Llama3RMSNorm
from ld_triton.modules.rms_norm.naive_rms_norm import NaiveRMSNorm
from ld_triton.modules.rms_norm.triton_rms_norm import TritonRMSNorm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['llama3', "naive", "triton"], 
        line_names=['Llama3', "Naive","Triton"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel='GB/s',
        plot_name='rms-norm-fwd',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'fwd'},
    ))
def bench_rms_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            triton_m = TritonRMSNorm(N, weight=weight, eps=eps)
            return triton_m(x)

        if provider == "naive":
            naive_m = NaiveRMSNorm(N, weight=weight, eps=eps)
            return naive_m(x)

        if provider == "llama3":
            llama3_m = Llama3RMSNorm(N, weight=weight, eps=eps)
            return llama3_m(x)


    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


bench_rms_norm.run(show_plots=True, print_data=True)