# forward
<p>
$X = e^{input}$
</p>

<p>
$Y = e^{input_1}$
</p>

<p>
$Q_{ij} = \text{softmax}(X)_{ij} = \frac{e^{x_{ij}}}{\sum_{k=1}^{N}e^{ik}}$
</p>

<p>
$P_{ij} = \text{softmax}(Y)_{ij} = \frac{e^{y_{ij}}}{\sum_{k=1}^{N}e^{ik}}$
</p>

<p>
$\text{reverseKL}_{i} = \sum_{j=1}^{N}Q_{ij} * (\text{log}Q_{ij} - \text{log}P_{ij})$
</p>

# backward
## 链式法则
<p>
$\frac{\partial loss}{\partial reverseKL}$ 已知
</p>

## backward Q
### 求导
<p>
  $\frac{\partial x(\log(x)-\log(y))}{\partial x} = log(x) - log(y) + 1 $
</p>

### 链式法则
元素形式
<p>
$\frac{\partial loss}{\partial Q_{ij}} = \sum_{p} \frac{\partial loss}{\partial reverseKL_{p}} * \frac{\partial reverseKL_{p}}{\partial Q_{ij}}$
</p>

<p>
$= \frac{\partial loss}{\partial reverseKL_{i}} * \frac{\partial reverseKL_{i}}{\partial Q_{ij}}$
</p>

<p>
$= dout_{i} * (Q_{ij} - P_{ij} + 1)$
</p>
矩阵形式
<p>
$\frac{\partial loss}{\partial Q} = dout * (Q - P + 1)$
</p>

## backward x
参见[softmax](https://github.com/l1351868270/ld_triton/blob/main/docs/ops/softmax.md)

<p>
$\frac{\partial loss}{\partial X} = softmax(X) * \left(dout - sum \left( softmax(x) * dout, dim=-1, keepdim=True \right) \right)$
</p>

## backward input
元素形式
<p>
$\frac{\partial loss}{\partial input_{ij}} = \sum_{p} \sum_{q}\frac{\partial loss}{\partial X_{pq}} * \frac{\partial X_{pq}}{\partial input_{ij}}$
</p>

<p>
$\frac{\partial loss}{\partial input_{ij}} = dout_{ij} * e^{input_{ij}}$
</p>
矩阵形式
<p>
$\frac{\partial loss}{\partial input} = dout * e^{input}$
</p>

