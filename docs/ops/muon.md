# Newton-Schulz 迭代
```
# Pytorch code
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

两个优化点：
1. 相同shape的矩阵使用batch newtonschulz迭代
2. 对称矩阵乘法可以只算上三角或者下三角


# 参考

[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)

https://github.com/KellerJordan/Muon/blob/master/muon.py

https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

