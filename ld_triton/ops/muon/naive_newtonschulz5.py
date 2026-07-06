
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
        # c * A @ A 的结合顺序对精度影响很大
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(dtype)


def naive_newtonschulz5_1(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    dtype = G.dtype
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        # c * (A @ A) 的结合顺序对精度影响很大
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(dtype)


def naive_batched_newtonschulz5(G, steps, eps=1e-7):
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    dtype = G.dtype
    X = G.bfloat16()
    X /= (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.to(dtype)


def naive_batched_newtonschulz5_1(G, steps=5, eps=1e-7):
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    dtype = G.dtype
    X = G.bfloat16()
    X /= (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if X.size(-2) > X.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    if X.size(-2) > X.size(-1):
        X = X.mT
    return X.to(dtype)


def naive_batched_newtonschulz5_2(G, steps=5, eps=1e-7):
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    dtype = G.dtype
    X = G.bfloat16()
    X /= (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if X.size(-2) > X.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
    if X.size(-2) > X.size(-1):
        X = X.mT
    return X.to(dtype)


if __name__ == "__main__":
    # 分别用@, torch.bmm, torch.baddbmm实现Newton-Schulz迭代，观察steps=5时的误差差别很大，是对精度敏感的
    B, M, N = 2, 4, 8
    device, dtype = "cuda", torch.float32
    input = torch.randn(B, M, N)
    output_list = []
    for b in range(B):
        o= naive_newtonschulz5(input[b].clone(), steps=5, eps=1e-7)
        output_list.append(o)
    output_0 = torch.stack(output_list, dim=0)

    output_list_1 = []
    for b in range(B):
        o= naive_newtonschulz5_1(input[b].clone(), steps=5, eps=1e-7)
        output_list_1.append(o)
    output_1 = torch.stack(output_list_1, dim=0)

    batched_output = naive_batched_newtonschulz5(input.clone(), steps=5, eps=1e-7)
    assert torch.allclose(output_0, batched_output, atol=1e-02, rtol=1e-02)
    
    batched_output_1 = naive_batched_newtonschulz5_1(input.clone(), steps=5, eps=1e-7)
    assert torch.allclose(output_1, batched_output_1, atol=1e-02, rtol=1e-02)

    batched_output_2 = naive_batched_newtonschulz5_2(input.clone(), steps=5, eps=1e-7)
    assert torch.allclose(batched_output_1, batched_output_2, atol=1e-02, rtol=1e-02)

