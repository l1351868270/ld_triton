import torch
import triton


def make_rotation_matrix(d: int, seed: int = 0) -> torch.tensor:
  G = torch.randn(d, d)
  Q, R = torch.linalg.qr(G)
  diag_sign = torch.sign(torch.diag(R))
  Q = Q * diag_sign
  return Q


@torch.compile
def make_rotation_matrix_compile(d: int, seed: int = 0) -> torch.tensor:
  G = torch.randn(d, d)
  Q, R = torch.linalg.qr(G)
  diag_sign = torch.sign(torch.diag(R))
  Q = Q * diag_sign
  return Q


if __name__ == "__main__": 
    torch.set_default_device('cuda:0')
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["dim"],
            x_vals=[2 ** i for i in range(2, 12)],
            line_arg="provider", 
            line_vals=["naive", "compile"], 
            line_names=["Naive", "Compile"],
            styles=[("green", "-"), ("red", "-")],
            ylabel="ms",
            plot_name="rotation",
            args={},
    ))


    @triton.testing.perf_report(configs)
    def benchmark(dim, provider):
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'naive':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: make_rotation_matrix(dim), quantiles=quantiles)
        if provider == 'compile':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: make_rotation_matrix_compile(dim), quantiles=quantiles)
        return ms, max_ms, min_ms


    benchmark.run(show_plots=True, print_data=True)

# black200, torch.compile性能急剧下降
#       dim  Naive (ms)  Compile (ms)
# 0     4.0    0.076864      0.063360
# 1     8.0    0.085024      0.074624
# 2    16.0    0.099200      0.088960
# 3    32.0    0.142144      0.147696
# 4    64.0    0.285856      0.405536
# 5   128.0    0.746352      1.200480
# 6   256.0    0.869488     20.415999
# 7   512.0    2.075616     31.295135
# 8  1024.0    4.897712    123.820061
# 9  2048.0   11.854320    348.667938
  
