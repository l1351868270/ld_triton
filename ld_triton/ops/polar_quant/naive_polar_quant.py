import torch


def make_rotation_matrix(d: int, seed: int = 0) -> torch.tensor:
  gen = torch.Generator().manual_seed(seed)
  G = torch.randn(d, d, generator=gen)
  Q, R = torch.linalg.qr(G)
  diag_sign = torch.sign(torch.diag(R))
  Q = Q * diag_sign
  return Q
  
  
