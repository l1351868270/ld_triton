
import torch


def naive_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    dtype = G.dtype
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        C = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(dtype)


if __name__ == "__main__":
    M, N = 4, 8
    device, dtype = "cuda", torch.float32
    input = torch.randn(M, N)
    output = naive_newtonschulz5(input, steps=5, eps=1e-7)
    print(f"output: {output}")
