
import torch

from ld_triton.ops.rms_norm.triton_rms_norm import triton_rms_norm


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, dim, weight: torch.tensor=None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = torch.nn.Parameter(torch.ones(dim).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.weight, self.eps, False)

    def extra_repr(self) -> str:
        return 'dim={dim}, eps={eps}'.format(**self.__dict__)