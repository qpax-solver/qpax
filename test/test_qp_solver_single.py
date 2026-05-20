import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpax import solve_qp

from .misc_test_utils import check_kkt_conditions, generate_random_qp


@pytest.mark.parametrize("backend", ["e", "i"])
def test_qp_solver(backend):
    np.random.seed(1)

    nx = 15
    ns = 10
    ny = 3

    jit_solve_qp = jax.jit(solve_qp, static_argnames=("backend",))
    solver_tol = 1e-2
    for first_test_iter in range(100):
        Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(
            nx, ns, ny
        )
        x, s, z, y, converged, iters = jit_solve_qp(
            Q, q, A, b, G, h, backend=backend, solver_tol=solver_tol
        )
        print(
            "test iter: ", first_test_iter, "converged: ", converged, "iters: ", iters
        )
        print("x - xreal: ", jnp.linalg.norm(x - x_true))

        del s_true, z_true, y_true

        assert converged == 1
        assert iters <= 10

        check_kkt_conditions(Q, q, A, b, G, h, x, s, z, y, solver_tol=solver_tol)

    ny = 0

    for second_test_iter in range(100):
        Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(
            nx, ns, ny
        )
        x, s, z, y, converged, iters = jit_solve_qp(
            Q, q, A, b, G, h, backend=backend, solver_tol=solver_tol
        )

        print(
            "test iter: ", second_test_iter, "converged: ", converged, "iters: ", iters
        )
        print("x - xreal: ", jnp.linalg.norm(x - x_true))

        del s_true, z_true, y_true

        assert converged == 1
        assert iters <= 10

        check_kkt_conditions(Q, q, A, b, G, h, x, s, z, y, solver_tol=solver_tol)


@pytest.mark.parametrize(
    ("backend", "seed", "nx", "ns", "ny"),
    [
        ("e", 0, 2, 2, 1),
        ("i", 13, 4, 3, 1),
    ],
)
def test_solver_preserves_state_on_terminal_iteration(backend, seed, nx, ns, ny):
    np.random.seed(seed)

    solver_tol = 1e-2

    Q, q, A, b, G, h, *_ = generate_random_qp(nx, ns, ny)
    Q = Q.astype(jnp.float32)
    q = q.astype(jnp.float32)
    A = A.astype(jnp.float32)
    b = b.astype(jnp.float32)
    G = G.astype(jnp.float32)
    h = h.astype(jnp.float32)

    full = solve_qp(Q, q, A, b, G, h, backend=backend, solver_tol=solver_tol)
    iters = int(full[-1])

    assert int(full[-2]) == 1
    assert iters > 1

    capped = solve_qp(
        Q,
        q,
        A,
        b,
        G,
        h,
        backend=backend,
        solver_tol=solver_tol,
        max_iter=iters - 1,
    )

    for full_value, capped_value in zip(full[:4], capped[:4], strict=True):
        np.testing.assert_array_equal(
            np.asarray(full_value), np.asarray(capped_value)
        )
