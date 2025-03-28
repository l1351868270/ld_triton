# forward
$x \in R^{MN}$

<p>
$max(x)_{i} = max({x_{i,0},...,x_{i,n-1}}) = x_{i,k}$
</p>

# backward
## 通用求导

$\frac {\partial x}{\partial x} = 1$

$\frac {\partial {x}}{\partial y} = 0$

## x求导

<p>
$\frac {\partial max(x)_{p}}{\partial x_{ij}}$
</p>

<p>
$=\frac {\partial x_{pk}}{\partial x_{ij}}$
</p>

### $p \neq i$
<p>
$\frac {\partial max(x)_{p}}{\partial x_{ij}}=0$
</p>

### $p = i, k \neq j$
<p>
$\frac {\partial max(x)_{p}}{\partial x_{ij}}=\frac {\partial x_{ik}}{\partial x_{ij}}=0$
</p>

### $p = i, k = j$
<p>
$\frac {\partial max(x)_{p}}{\partial x_{ij}}=\frac {\partial x_{ik}}{\partial x_{ij}}=1$
</p>

## 链式法则

$\frac{\partial f(max(x))}{\partial x_{ij}}$

<p>
$=\sum_{p=0}^{M-1} \frac{\partial f(max(x))}{\partial max(x)_{p}} . \frac{\partial max(x)_{p}}{\partial x_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1} df_{p} . \frac{\partial max(x)_{p}}{\partial x_{ij}}$
</p>

<p>
$=df_{i} . \frac{\partial x_{ik}}{\partial x_{ij}}$
</p>

<p>
$=df_{i} . \frac{\partial x_{ik}}{\partial x_{ij}}$
</p>

### $k \neq j$

$\frac{\partial f(max(x))}{\partial x_{ij}} = 0$

### $k = j$

$\frac{\partial f(max(x))}{\partial x_{ij}} = df_{i}$

