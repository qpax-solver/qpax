# qpax
Differentiable QP solver in [JAX](https://github.com/google/jax).

[![Paper](http://img.shields.io/badge/arXiv-2207.00669-B31B1B.svg)](https://arxiv.org/abs/2406.11749)

This package can be used for solving convex quadratic programs of the following form:

$$
\begin{align*}
\underset{x}{\text{minimize}} & \quad \frac{1}{2}x^TQx + q^Tx \\
\text{subject to} & \quad  Ax = b, \\
                  & \quad  Gx \leq h
\end{align*}
$$

where $Q \succeq 0$. This solver can be combined with JAX's `jit` and `vmap` functionality, as well as differentiated with reverse-mode `grad`. 

The QP is solved with a primal-dual interior point algorithm detailed in [cvxgen](https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf), with the solution to the linear systems computed with reduction techniques from [cvxopt](http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf). At an approximate primal-dual solution, the primal variable $x$ is differentiated with respect to the problem parameters using the implicit function theorem as shown in [optnet](https://arxiv.org/abs/1703.00443), and their pytorch-based qp solver [qpth](https://github.com/locuslab/qpth).

## Installation

To install directly from github using `pip`:

```bash
$ pip install qpax
```

Alternatively, to install from source in editable mode:

```bash
$ pip install -e .
```

## Usage

### 🚨 Float32 Warning 🚨

The solver tolerance (`solver_tol`) should be something reasonable given the available precision. With 32bit precision (the default in JAX), `solver_tol` should be greater than `1e-5`.

| Precision  | Tolerance   |
|------------|------------|
| `jnp.float32` | `solver_tol`$\in$ `[1e-5, 1e-2]` |
| `jnp.float64` | `solver_tol`$\in$ `[1e-12, 1e-2]`  |

In order to enable 64bit precision, you can do the following at startup:

```python
# again, this only works on startup!
import jax
jax.config.update("jax_enable_x64", True)
```
This is taken from the [JAX - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

### Solving a QP
We can solve QPs with qpax in a way that plays nice with JAX's `jit` and `vmap`:
```python
import qpax

# solve QP (this can be combined with jit or vmap)
x, s, z, y, converged, iters = qpax.solve_qp(Q, q, A, b, G, h, solver_tol=1e-6)
```

### Linear System Solver

By default, qpax uses a Cholesky factorization to solve the internal linear systems. You can switch to QR factorization instead, which can be more numerically stable for ill-conditioned problems:

```python
import qpax

# use QR factorization for the internal linear solves
x, s, z, y, converged, iters = qpax.solve_qp(
    Q, q, A, b, G, h,
    linear_solver=qpax.LinearSolver.QR,
)
```

Available options are `qpax.LinearSolver.CHOLESKY` (default) and `qpax.LinearSolver.QR`.
### Solving a batch of QP's 

Here let's solve a batch of nonnegative least squares problems as QPs. This outlines two bits of functionality from `qpax`, first is the ability to solve QPs without any equality constraints, and second is the ability to `vmap` over a batch of QPs. 

```python 
import numpy as np
import jax 
import jax.numpy as jnp 
from jax import jit, grad, vmap  
import qpax 
import timeit

"""
solve batched non-negative least squares (nnls) problems
 
min_x    |Fx - g|^2 
st        x >= 0 
"""

n = 5   # size of x 
m = 10  # rows in F 

# create data for N_qps random nnls problems  
N_qps = 10000 
Fs = jnp.array(np.random.randn(N_qps, m, n))
gs = jnp.array(np.random.randn(N_qps, m))

@jit
def form_qp(F, g):
  # convert the least squares to qp form 
  n = F.shape[1]
  Q = F.T @ F 
  q = -F.T @ g 
  G = -jnp.eye(n)
  h = jnp.zeros(n)
  A = jnp.zeros((0, n))
  b = jnp.zeros(0)
  return Q, q, A, b, G, h

# create the QPs in a batched fashion 
Qs, qs, As, bs, Gs, hs = vmap(form_qp, in_axes = (0, 0))(Fs, gs)

# create function for solving a batch of QPs 
batch_qp = jit(vmap(qpax.solve_qp_primal, in_axes = (0, 0, 0, 0, 0, 0)))

xs = batch_qp(Qs, qs, As, bs, Gs, hs)
```

### Differentiating a QP 

Alternatively, if we are only looking to use the primal variable `x`, we can use `solve_qp_primal` which enables automatic differentiation:

```python
import jax 
import jax.numpy as jnp 
import qpax 

def loss(Q, q, A, b, G, h):
    x = qpax.solve_qp_primal(Q, q, A, b, G, h, solver_tol=1e-4, target_kappa=1e-3) 
    x_bar = jnp.ones(len(q))
    return jnp.dot(x - x_bar, x - x_bar)
  
# gradient of loss function   
loss_grad = jax.grad(loss, argnums = (0, 1, 2, 3, 4, 5))

# compatible with jit 
loss_grad_jit = jax.jit(loss_grad)

# calculate derivatives 
derivs = loss_grad_jit(Q, q, A, b, G, h)
dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh = derivs 
```
where `target_kappa` is used to determine how much smoothing should be applied to the gradients through `solve_qp_primal`. For more detail on `target_kappa`, please refer to [the paper](https://arxiv.org/abs/2406.11749).
## Citation 
[![Paper](http://img.shields.io/badge/arXiv-2207.00669-B31B1B.svg)](https://arxiv.org/abs/2406.11749)
```
@misc{tracy2024differentiability,
    title={On the Differentiability of the Primal-Dual Interior-Point Method},
    author={Kevin Tracy and Zachary Manchester},
    year={2024},
    eprint={2406.11749},
    archivePrefix={arXiv},
    primaryClass={math.OC}
}
```
