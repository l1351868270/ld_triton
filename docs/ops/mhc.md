
# Sinkhorn-Knopp
## forward

论文里面的公式是：是先col在row但是实现都是先row再col，本文采用先row再col

论文：

for t in range(iter): $M^{(t)} = \tau_{r}(\tau_{c}(M^{(t-1)}))$

实现：

for t in range(iter): $M^{(t)} = \tau_{c}(\tau_{r}(M^{(t-1)}))$

元素级别描述：

<p>
$\tau_{r}(M^{(t-1)}_{ij}) = \frac{M^{(t-1)}_{ij}}{\sum_{k=0}^{n_{hc}}M^{(t-1)}_{ik}}$
</p>

<p>
  $\hat{M}^{(t-1)}_{ij} = \tau_{r}(M^{(t-1)}_{ij})$
</p>

<p>
$\tau_{c}(\hat{M}^{(t)}_{ij}) = \frac{\hat{M}^{(t)}_{ij}}{\sum_{k=0}^{n_{hc}}\hat{M}^{(t)}_{kj}}$
</p>

## 求导

对 $\tau_{c}$ 求导

<p>
$\frac{\partial \tau_{c}(\hat{M}^{(t)}_{ij})}{\partial \hat{M}^{(t)}_{pq}} = \partial \frac{\hat{M}^{(t)}_{ij}}{\sum_{k=0}^{n_{hc}}\hat{M}^{(t)}_{kj}} / \partial \hat{M}^{(t)}_{ij} $
</p>

$q \neq j$

<p>
$\frac{\partial \tau_{c}(\hat{M}^{(t)}_{ij})}{\partial \hat{M}^{(t)}_{pq}} = 0$
</p>

<p>
$q = j, p = i$
</p>

<p>
$\frac{\partial \tau_{c}(\hat{M}^{(t)}_{ij})}{\partial \hat{M}^{(t)}_{pq}} = \frac{\partial \tau_{c}(\hat{M}^{(t)}_{ij})}{\partial \hat{M}^{(t)}_{ij}}$
</p>

<p>
$=\frac{(\sum_{k=0}^{n_{hc}}\hat{M}^{(t)}_{kj}) - \hat{M}^{(t)}_{ij}}{(\sum_{k=0}^{n_{hc}}\hat{M}^{(t)}_{kj})^{2}}$
</p>

<p>
$q = j, p \neq i$
</p>

<p>
$\frac{\partial \tau_{c}(\hat{M}^{(t)}_{ij})}{\partial \hat{M}^{(t)}_{pj}} = \frac{\partial \tau_{c}(\hat{M}^{(t)}_{ij})}{\partial \hat{M}^{(t)}_{pj}}$
</p>

<p>
$=\frac{ - \hat{M}^{(t)}_{pj}}{(\sum_{k=0}^{n_{hc}}\hat{M}^{(t)}_{kj})^{2}}$
</p>

对 $\tau_{r}$ 求导

<p>
$\frac{\partial \tau_{r}(M^{(t-1)}_{ij})}{\partial M^{(t-1)}_{pq}} = \partial \frac{M^{(t-1)}_{ij}}{\sum_{k=0}^{n_{hc}}M^{(t-1)}_{ik}} / \partial M^{(t-1)}_{pq}$
</p>

$p \neq i$

<p>
$\frac{\partial \tau_{r}(\hat{M}^{(t)}_{ij})}{\partial \hat{M}^{(t)}_{pq}} = 0$
</p>

