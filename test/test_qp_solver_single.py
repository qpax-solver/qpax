import jax

# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# import jax.scipy as jsp
# from jax import grad
# from jax import jacobian
# from jax import jit
# from jax import vmap
import numpy as np

# from jax import custom_vjp
from misc_test_utils import check_kkt_conditions, generate_random_qp

import qpax


def test_qp_solver():
    np.random.seed(1)

    # test 1000 normal QP's
    nx = 15
    ns = 10
    # nz = ns
    ny = 3

    jit_solve_qp = jax.jit(qpax.solve_qp)
    solver_tol = 1e-2
    for first_test_iter in range(1000):
        Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(
            nx, ns, ny
        )
        x, s, z, y, converged, iters = jit_solve_qp(
            Q, q, A, b, G, h, solver_tol=solver_tol
        )
        print(
            "test iter: ", first_test_iter, "converged: ", converged, "iters: ", iters
        )
        print("x - xreal: ", jnp.linalg.norm(x - x_true))

        assert converged == 1
        assert iters <= 10  # this is much stricter than the continuation criteria

        check_kkt_conditions(Q, q, A, b, G, h, x, s, z, y, solver_tol=solver_tol)

    # test 1000 inequality-only QP's
    nx = 15
    ns = 10
    # nz = ns
    ny = 0

    for second_test_iter in range(1000):
        Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(
            nx, ns, ny
        )
        x, s, z, y, converged, iters = jit_solve_qp(
            Q, q, A, b, G, h, solver_tol=solver_tol
        )

        print(
            "test iter: ", second_test_iter, "converged: ", converged, "iters: ", iters
        )
        print("x - xreal: ", jnp.linalg.norm(x - x_true))

        assert converged == 1
        assert iters <= 10  # this is much stricter than the continuation criteria

        check_kkt_conditions(Q, q, A, b, G, h, x, s, z, y, solver_tol=solver_tol)
