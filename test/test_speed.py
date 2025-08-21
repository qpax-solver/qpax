from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
from misc_test_utils import check_kkt_conditions, generate_random_qp

import qpax


def test_solver(
    nx, ns, ny, n_test=1000, solver_tol=1e-6, elastic=False, elastic_penalty=1e3
):

    @jax.jit
    def solve_primal(Q, q, A, b, G, h):
        if elastic:
            return qpax.solve_qp_elastic_primal(Q, q, G, h, elastic_penalty, solver_tol)
        return qpax.solve_qp_primal(Q, q, A, b, G, h, solver_tol=solver_tol)

    # Initial solve for jit-compilation
    Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(nx, ns, ny)
    start_time = time.perf_counter()
    _ = solve_primal(Q, q, A, b, G, h).block_until_ready()
    jit_time = time.perf_counter() - start_time

    # Determine solve times from jitted solver
    times = []
    for _ in range(1000):
        Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(
            nx, ns, ny
        )
        start_time = time.perf_counter()
        _ = solve_primal(Q, q, A, b, G, h).block_until_ready()
        times.append(time.perf_counter() - start_time)

    mean_solve_time = np.mean(times)
    std_solve_time = np.std(times)

    print(f"Jit compilation time: {jit_time}")
    print(f"Mean: {mean_solve_time * 1e6} microseconds")
    print(f"Std: {std_solve_time * 1e6} microseconds")


def run_tests(elastic=False):
    np.random.seed(1)

    print(f"\n== Speed test: {'elastic' if elastic else 'regular'} QP solver ==\n")

    if not elastic:
        # Normal QPs
        nx = 15
        ns = 10
        ny = 3
        print("Testing inequality/equality constrained QPs")
        test_solver(nx, ns, ny, elastic=elastic)

    # Inequality-only QP's
    nx = 15
    ns = 10
    ny = 0
    print("\nTesting inequality-only constrained QPs")
    test_solver(nx, ns, ny, elastic=elastic)

    if not elastic:
        # Larger normal QPs
        nx = 45
        ns = 30
        ny = 9
        print("\nTesting larger inequality/equality constrained QPs")
        test_solver(nx, ns, ny, elastic=elastic)

    # Larger Inequality-only QP's
    nx = 45
    ns = 30
    ny = 0
    print("\nTesting larger inequality-only constrained QPs")
    test_solver(nx, ns, ny, elastic=elastic)


if __name__ == "__main__":
    run_tests(elastic=False)
    run_tests(elastic=True)
