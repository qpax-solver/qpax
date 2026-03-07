import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from misc_test_utils import finite_difference, generate_random_qp

import qpax


@jax.jit
def my_f(Q, q, A, b, G, h):
    x = qpax.solve_qp_primal(Q, q, A, b, G, h, target_kappa=1e-3)
    x_bar = jnp.ones(len(q))
    return jnp.dot(x - x_bar, x - x_bar)


def my_f_select(inputs, X, i):
    # replace the ith element of inputs with X
    new_inputs = tuple(X if index == i else value for index, value in enumerate(inputs))
    return my_f(*new_inputs)


def test_derivs():
    np.random.seed(3)
    nx = 15
    ns = 10
    # nz = ns
    ny = 3
    Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(nx, ns, ny)

    inputs = (Q, q, A, b, G, h)
    grad_my_f = jax.jit(grad(my_f, argnums=(0, 1, 2, 3, 4, 5)))
    derivs = grad_my_f(*inputs)

    input_names = ("Q", "q", "A", "b", "G", "h")
    for i in range(6):
        print("-------------input: ", input_names[i], "----------------")

        def lambda_f(_X, _i=i):
            return my_f_select(inputs, _X, _i)

        # lambda_f = lambda _X: my_f_select(inputs, _X, i)

        fd_deriv = finite_difference(lambda_f, inputs[i])

        assert fd_deriv.shape == derivs[i].shape

        print("fd_deriv_norm: ")
        print(jnp.linalg.norm(fd_deriv))
        print("error_norm: ", jnp.linalg.norm(derivs[i] - fd_deriv))

        assert jnp.linalg.norm(derivs[i] - fd_deriv) < (0.2 * jnp.linalg.norm(fd_deriv))
