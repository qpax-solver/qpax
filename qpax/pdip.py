"""PDIP functions for solving QP problems."""

from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


class LinearSolver(Enum):
    CHOLESKY = "cholesky"
    QR = "qr"


def _factor(M, solver: LinearSolver):
    """Factorize a matrix using the chosen method."""
    if solver is LinearSolver.QR:
        return jnp.linalg.qr(M)
    return jsp.linalg.cho_factor(M)


def _solve(factored, rhs, solver: LinearSolver):
    """Solve a linear system given a factorization."""
    if solver is LinearSolver.QR:
        Q, R = factored
        return jsp.linalg.solve_triangular(R, Q.T @ rhs)
    return jsp.linalg.cho_solve(factored, rhs)


# kept for backward compat (elastic_qp imports this)
def qr_solve(qr, rhs):
    """Solve a linear system using the QR decomposition."""
    return jsp.linalg.solve_triangular(qr[1], qr[0].T @ rhs)


class QPData(NamedTuple):
    Q: jax.Array
    q: jax.Array
    A: jax.Array
    b: jax.Array
    G: jax.Array
    h: jax.Array


class SolverParams(NamedTuple):
    tol: float = 1e-3
    max_iter: int = 30
    linear_solver: LinearSolver = LinearSolver.CHOLESKY


class QPState(NamedTuple):
    x: jax.Array
    s: jax.Array
    z: jax.Array
    y: jax.Array


def initialize(qp: QPData, solver: LinearSolver = LinearSolver.CHOLESKY) -> QPState:
    """Initialize primal and dual variables from CVXGEN/CVXOPT."""
    Q, q, A, b, G, h = qp

    H = Q + G.T @ G
    L_H = _factor(H, solver)
    F = A @ _solve(L_H, A.T, solver)
    L_F = _factor(F, solver)

    r1 = -q + G.T @ h
    y = _solve(L_F, A @ _solve(L_H, r1, solver) - b, solver)
    x = _solve(L_H, r1 - A.T @ y, solver)
    z = G @ x - h

    alpha_p = -jnp.min(-z)
    s = jnp.where(alpha_p < 0, -z, -z + (1 + alpha_p))

    alpha_d = -jnp.min(z)
    z = jnp.where(alpha_d >= 0, z + (1 + alpha_d), z)

    return QPState(x, s, z, y)


def factorize_kkt(Q, G, A, s, z, solver: LinearSolver = LinearSolver.CHOLESKY):
    """Factorize the KKT system."""
    P_inv_vec = z / s
    H = Q + G.T @ (G.T * P_inv_vec).T
    L_H = _factor(H, solver)
    F = A @ _solve(L_H, A.T, solver)
    L_F = _factor(F, solver)

    return P_inv_vec, L_H, L_F


def solve_kkt_rhs(
    G,
    A,
    s,
    z,
    P_inv_vec,
    L_H,
    L_F,
    v1,
    v2,
    v3,
    v4,
    solver: LinearSolver = LinearSolver.CHOLESKY,
):
    """Solve the KKT system for the residuals v1, v2, v3, v4."""
    r2 = v3 - v2 / z
    p1 = v1 + G.T @ (P_inv_vec * r2)

    dy = _solve(L_F, A @ _solve(L_H, p1, solver) - v4, solver)
    dx = _solve(L_H, p1 - A.T @ dy, solver)
    ds = v3 - G @ dx
    dz = (v2 - z * ds) / s

    return dx, ds, dz, dy


def ort_linesearch(x, dx):
    """max alpha <= 1 such that x + dx >= 0"""
    alphas = jnp.where(dx < 0, -x / dx, 1.0)
    return jnp.minimum(1.0, jnp.min(alphas))  # type: ignore


def centering_params(s, z, ds_a, dz_a):
    """duality gap + cc term in predictor-corrector PDIP"""
    mu = jnp.dot(s, z) / len(s)
    alpha = jnp.min(jnp.array([ort_linesearch(s, ds_a), ort_linesearch(z, dz_a)]))
    sigma = (jnp.dot(s + alpha * ds_a, z + alpha * dz_a) / jnp.dot(s, z)) ** 3
    return sigma, mu


def solve_eq_only(Q, q, A, b, solver: LinearSolver = LinearSolver.CHOLESKY):
    """Solve equality constrained QP (Boyd, Convex, pg 559)."""
    Q_f = _factor(Q, solver)

    Qinv_At = _solve(Q_f, A.T, solver)
    Qinv_g = _solve(Q_f, q, solver)

    S = A @ Qinv_At
    S_f = _factor(S, solver)
    y = _solve(S_f, -(A @ Qinv_g + b), solver)

    x = _solve(Q_f, -A.T @ y - q, solver)

    return x, y


def remove_inf_constraints(G, h):
    """Remove infinite constraints from G and h."""
    h_mask_is_inf = jnp.isinf(h)
    h2 = jnp.where(h_mask_is_inf, 0, h)
    G2 = jnp.diag(1 * ~h_mask_is_inf) @ G
    return G2, h2, h_mask_is_inf


def solve_qp(
    Q: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    G: jax.Array,
    h: jax.Array,
    solver_tol: float = 1e-3,
    max_iter: int = 30,
    linear_solver: LinearSolver = LinearSolver.CHOLESKY,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int, int]:
    """Solve a QP using a primal-dual interior point method.

    Args:
        Q: (n, n) positive definite matrix
        q: (n,) vector
        A: (m, n) equality constraint matrix
        b: (m,) equality constraint vector
        G: (p, n) inequality constraint matrix
        h: (p,) inequality constraint vector
        linear_solver: LinearSolver.CHOLESKY or LinearSolver.QR

    Returns:
        x: (n,) optimal solution
        s: (p,) inequality slack variables
        z: (p,) inequality dual variables
        y: (m,) equality dual variables
        converged: int convergence flag
        pdip_iter: int number of iterations
    """

    # make sure each matrix is 2D
    Q = jnp.atleast_2d(Q)
    A = jnp.atleast_2d(A)
    G = jnp.atleast_2d(G)

    if (len(b) == 0) and (len(h) == 0):
        x = jnp.linalg.solve(Q, -q)
        return x, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), 1, 0

    if len(h) == 0:
        x, y = solve_eq_only(Q, q, A, b, linear_solver)
        return x, jnp.zeros(0), jnp.zeros(0), y, 1, 0

    G, h, h_mask_is_inf = remove_inf_constraints(G, h)  # type: ignore
    Q = 0.5 * (Q + Q.T)

    params = SolverParams(
        tol=solver_tol,
        max_iter=max_iter,
        linear_solver=linear_solver,
    )
    qp = QPData(Q, q, A, b, G, h)
    state = initialize(qp, linear_solver)

    nonlocal_tol = params.tol
    ls = params.linear_solver

    def _step(inputs):
        qp, st, converged, pdip_iter = inputs
        Q, q, A, b, G, h = qp
        x, s, z, y = st

        r1 = Q @ x + q + A.T @ y + G.T @ z
        r2 = s * z
        r3 = G @ x + s - h
        r4 = A @ x - b

        kkt_res = jnp.concatenate((r1, r2, r3, r4))
        converged = jnp.where(
            jnp.linalg.norm(kkt_res, ord=jnp.inf) < nonlocal_tol, 1, 0
        )

        P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, z, ls)
        _, ds_a, dz_a, _ = solve_kkt_rhs(
            G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4, ls
        )

        sigma, mu = centering_params(s, z, ds_a, dz_a)
        r2 = r2 - (sigma * mu - (ds_a * dz_a))
        dx, ds, dz, dy = solve_kkt_rhs(
            G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4, ls
        )

        alpha = 0.99 * jnp.min(
            jnp.array([ort_linesearch(s, ds), ort_linesearch(z, dz)])
        )

        new_state = QPState(
            x + alpha * dx, s + alpha * ds, z + alpha * dz, y + alpha * dy
        )
        return (qp, new_state, converged, pdip_iter + 1)

    def _cond(inputs):
        _, _, converged, pdip_iter = inputs
        return jnp.logical_and(pdip_iter < params.max_iter, converged == 0)

    init = (qp, state, 0, 0)
    outputs = jax.lax.while_loop(_cond, _step, init)

    _, final_state, converged, pdip_iter = outputs
    x, s, z, y = final_state

    z = jnp.where(h_mask_is_inf, 0, z)
    s = jnp.where(h_mask_is_inf, jnp.inf, s)

    return x, s, z, y, converged, pdip_iter  # type: ignore


def solve_qp_debug(
    Q,
    q,
    A,
    b,
    G,
    h,
    solver_tol=1e-3,
    max_iter=30,
    linear_solver=LinearSolver.CHOLESKY,
):
    """Debug solving with verbose printing."""
    Q = jnp.atleast_2d(Q)
    A = jnp.atleast_2d(A)
    G = jnp.atleast_2d(G)

    if (len(b) == 0) and (len(h) == 0):
        x = jnp.linalg.solve(Q, -q)
        return x, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), 1, 0

    if len(h) == 0:
        x, y = solve_eq_only(Q, q, A, b, linear_solver)
        return x, jnp.zeros(0), jnp.zeros(0), y, 1, 0

    G, h, h_mask_is_inf = remove_inf_constraints(G, h)  # type: ignore
    Q = 0.5 * (Q + Q.T)

    params = SolverParams(
        tol=solver_tol,
        max_iter=max_iter,
        linear_solver=linear_solver,
    )
    qp = QPData(Q, q, A, b, G, h)  # type: ignore
    state = initialize(qp, linear_solver)
    ls = params.linear_solver

    def _step_debug(inputs):
        qp, st, converged, pdip_iter = inputs
        Q, q, A, b, G, h = qp
        x, s, z, y = st

        r1 = Q @ x + q + A.T @ y + G.T @ z
        r2 = s * z
        r3 = G @ x + s - h
        r4 = A @ x - b

        kkt_res = jnp.concatenate((r1, r2, r3, r4))
        converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < params.tol, 1, 0)

        P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, z, ls)
        _, ds_a, dz_a, _ = solve_kkt_rhs(
            G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4, ls
        )

        sigma, mu = centering_params(s, z, ds_a, dz_a)
        r2 = r2 - (sigma * mu - (ds_a * dz_a))
        dx, ds, dz, dy = solve_kkt_rhs(
            G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4, ls
        )

        alpha = 0.99 * jnp.min(
            jnp.array([1.0, 0.99 * ort_linesearch(s, ds), 0.99 * ort_linesearch(z, dz)])
        )

        r4_print = r4 if len(r4) > 0 else jnp.zeros(1)
        nr1 = jnp.linalg.norm(r1, ord=jnp.inf)
        nr2 = jnp.linalg.norm(r2, ord=jnp.inf)
        nr3 = jnp.linalg.norm(r3, ord=jnp.inf)
        nr4 = jnp.linalg.norm(r4_print, ord=jnp.inf)
        print(
            f"{pdip_iter:3d}   {nr1:9.2e}   {nr2:9.2e}"
            f"  {nr3:9.2e}  {nr4:9.2e}   {alpha:6.4f}"
        )

        new_state = QPState(
            x + alpha * dx, s + alpha * ds, z + alpha * dz, y + alpha * dy
        )
        return (qp, new_state, converged, pdip_iter + 1)

    def _cond(inputs):
        _, _, converged, pdip_iter = inputs
        return jnp.logical_and(pdip_iter < params.max_iter, converged == 0)

    print("iter      r1          r2         r3         r4        alpha")
    print("------------------------------------------------------")

    val = (qp, state, 0, 0)
    while _cond(val):
        val = _step_debug(val)

    _, final_state, converged, pdip_iter = val
    x, s, z, y = final_state

    z = jnp.where(h_mask_is_inf, 0, z)  # type: ignore
    s = jnp.where(h_mask_is_inf, jnp.inf, s)  # type: ignore

    return x, s, z, y, converged, pdip_iter
