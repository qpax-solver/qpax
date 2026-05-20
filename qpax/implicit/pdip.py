"""PDIP functions for solving QP problems."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from qpax._verbose import print_footer, print_header


class QPData(NamedTuple):
    Q: jax.Array
    q: jax.Array
    A: jax.Array
    b: jax.Array
    G: jax.Array
    h: jax.Array


class QPState(NamedTuple):
    x: jax.Array
    s: jax.Array
    z: jax.Array
    y: jax.Array


class LinearSolver(Enum):
    CHOLESKY = "cholesky"
    QR = "qr"


class SolverParams(NamedTuple):
    tol: float = 1e-3
    max_iter: int = 30
    linear_solver: LinearSolver = LinearSolver.CHOLESKY


# ---------------------------------- helpers --------------------------------- #


def _all_finite(*arrays) -> jax.Array:
    """Return True iff every provided array/scalar is finite."""
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(jnp.ravel(a))) for a in arrays]))


def remove_inf_constraints(G, h):
    """Remove infinite constraints from G and h."""
    h_mask_is_inf = jnp.isinf(h)
    h2 = jnp.where(h_mask_is_inf, 0, h)
    G2 = jnp.diag(1 * ~h_mask_is_inf) @ G
    return G2, h2, h_mask_is_inf


# ------------------------------ initialization ------------------------------ #


def _factor_init(M, solver: LinearSolver):
    """Factorize a matrix using the chosen method."""
    if solver is LinearSolver.QR:
        return jnp.linalg.qr(M)
    return jsp.linalg.cho_factor(M)


def _solve_init(factored, rhs, solver: LinearSolver):
    """Solve a linear system given a factorization."""
    if solver is LinearSolver.QR:
        Q, R = factored
        return jsp.linalg.solve_triangular(R, Q.T @ rhs)
    return jsp.linalg.cho_solve(factored, rhs)


def initialize(qp: QPData, solver: LinearSolver = LinearSolver.CHOLESKY) -> QPState:
    """Initialize primal and dual variables from CVXGEN/CVXOPT."""
    Q, q, A, b, G, h = qp

    GtG = jnp.matmul(G.T, G, precision=jax.lax.Precision.HIGHEST)
    H = Q + GtG
    L_H = _factor_init(H, solver)
    F = A @ _solve_init(L_H, A.T, solver)
    L_F = _factor_init(F, solver)

    r1 = -q + G.T @ h
    y = _solve_init(L_F, A @ _solve_init(L_H, r1, solver) - b, solver)
    x = _solve_init(L_H, r1 - A.T @ y, solver)
    z = G @ x - h

    alpha_p = -jnp.min(-z)
    s = jnp.where(alpha_p < 0, -z, -z + (1 + alpha_p))

    alpha_d = -jnp.min(z)
    z = jnp.where(alpha_d >= 0, z + (1 + alpha_d), z)

    return QPState(x, s, z, y)


def solve_eq_only(Q, q, A, b, solver: LinearSolver = LinearSolver.CHOLESKY):
    """Solve equality constrained QP (Boyd, Convex, pg 559)."""
    Q_f = _factor_init(Q, solver)

    Qinv_At = _solve_init(Q_f, A.T, solver)
    Qinv_g = _solve_init(Q_f, q, solver)

    S = A @ Qinv_At
    S_f = _factor_init(S, solver)
    y = _solve_init(S_f, -(A @ Qinv_g + b), solver)

    x = _solve_init(Q_f, -A.T @ y - q, solver)

    return x, y


# ------------------------------ retraction map ------------------------------ #


def retraction_map(v: jax.Array, kappa: jax.Array | float) -> jax.Array:
    """Softplus-style retraction map enforcing b(v) * b(-v) = kappa."""
    sq = jnp.sqrt(v * v + 4.0 * kappa)
    return jnp.where(v >= 0, 0.5 * (v + sq), 2.0 * kappa / (sq - v))


def derivative_retraction_map(v: jax.Array, kappa: jax.Array | float) -> jax.Array:
    """Derivative of the softplus-style retraction map with respect to v."""
    scalar = v.dtype.type
    kappa = jnp.asarray(kappa, dtype=v.dtype)
    vv = v * v
    sq = jnp.sqrt(vv + scalar(4.0) * kappa)
    return jnp.where(
        v < 0,
        scalar(2.0) * kappa / (vv + scalar(4.0) * kappa - v * sq),
        scalar(0.5) * (scalar(1.0) + v / sq),
    )


def derivative_retraction_map_kappa(
    v: jax.Array, kappa: jax.Array | float
) -> jax.Array:
    """Derivative of the softplus-style retraction map with respect to kappa."""
    return 1.0 / jnp.sqrt(v * v + 4.0 * kappa)


# -------------------------------- line search ------------------------------- #


def ort_linesearch(x, dx):
    """max alpha <= 1 such that x + dx >= 0"""
    alphas = jnp.where(dx < 0, -x / dx, 1.0)
    return jnp.minimum(1.0, jnp.min(alphas))  # type: ignore


# ---------------------------- linear system solve --------------------------- #


def factorize_implicit_kkt(Q, A, G, v, kappa):
    """Factorize the reduced 3-block implicit KKT system with LU.

    Builds and factors the symmetric-indefinite saddle matrix
        [Q - G'G   G'       A'      ]
        [G        -diag(Bn)  0      ]
        [A         0          0     ]
    where Bn = diag(b_kappa'(-v)) has entries in [0, 1]. No Bp/Bn ratio is formed.
    """
    Bp_vec = derivative_retraction_map(v, kappa)
    Bn_vec = derivative_retraction_map(-v, kappa)
    c_vec = derivative_retraction_map_kappa(v, kappa)

    nz = G.shape[0]
    ny = A.shape[0]

    GtG = jnp.matmul(G.T, G, precision=jax.lax.Precision.HIGHEST)
    J = jnp.block(
        [
            [Q - GtG, G.T, A.T],
            [G, -jnp.diag(Bn_vec), jnp.zeros((nz, ny), dtype=Q.dtype)],
            [A, jnp.zeros((ny, nz), dtype=Q.dtype), jnp.zeros((ny, ny), dtype=Q.dtype)],
        ]
    )
    L_J = jsp.linalg.lu_factor(J)

    return Bp_vec, Bn_vec, c_vec, L_J


def solve_implicit_kkt_rhs(G, Bn_vec, Bp_vec, c_vec, L_J, rt, re, ri, rz, rs, rk):
    """Solve the implicit KKT system given a pre-computed LU factorization."""
    nx = G.shape[1]
    nz = G.shape[0]

    rhs = jnp.concatenate([rt - G.T @ (ri + rz - rs), ri - rs - c_vec * rk, re])
    sol = jsp.linalg.lu_solve(L_J, -rhs)

    dx = sol[:nx]
    dv = sol[nx : nx + nz]
    dy = sol[nx + nz :]

    dz = -rz + Bp_vec * dv - c_vec * rk
    ds = -rs - Bn_vec * dv - c_vec * rk
    dk = -rk

    return dx, ds, dz, dy, dv, dk


# ---------------------------------------------------------------------------- #
#                                    solver                                    #
# ---------------------------------------------------------------------------- #


def solve_qp(
    Q: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    G: jax.Array,
    h: jax.Array,
    solver_tol: float = 1e-5,
    max_iter: int = 30,
    linear_solver: LinearSolver = LinearSolver.CHOLESKY,
    sigma: float = 0.125,
    verbose: bool = False,
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

    def _step(inputs):
        qp, st, converged, pdip_iter = inputs
        Q, q, A, b, G, h = qp
        x, s, z, y = st

        v = z - s
        kappa = jnp.maximum(jnp.dot(s, z) / len(s), 1e-14)

        rt = Q @ x + q + A.T @ y + G.T @ z
        rc = s * z
        ri = G @ x + s - h
        re = A @ x - b
        rz = z - retraction_map(v, kappa)
        rs = s - retraction_map(-v, kappa)

        kkt_res = jnp.concatenate((rt, rc, ri, re))
        converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < params.tol, 1, 0)

        Bp_vec, Bn_vec, c_vec, L_J = factorize_implicit_kkt(Q, A, G, v, kappa)

        # TEST  ------------- STEP -------------
        kappa_target = sigma * kappa
        rk = kappa - kappa_target

        dx, ds, dz, dy, dv, dk = solve_implicit_kkt_rhs(
            G, Bn_vec, Bp_vec, c_vec, L_J, rt, re, ri, rz, rs, rk
        )
        # TEST  ------------- STEP -------------

        alpha = 0.99 * jnp.min(
            jnp.array([ort_linesearch(s, ds), ort_linesearch(z, dz)])
        )

        if verbose:
            re_print = re if len(re) > 0 else jnp.zeros(1)
            print(
                f"{pdip_iter:3d}   {kappa:9.2e}   "
                f"{jnp.linalg.norm(rt, ord=jnp.inf):9.2e}   "
                f"{jnp.linalg.norm(rc, ord=jnp.inf):9.2e}  "
                f"{jnp.linalg.norm(ri, ord=jnp.inf):9.2e}  "
                f"{jnp.linalg.norm(re_print, ord=jnp.inf):9.2e}    "
                f"{alpha:6.4f}   {sigma:9.4f}"
            )

        # Under vmap, the while loop runs until the slowest lane converges.
        # Freezing converged lanes avoids post-convergence drift in f32.
        take = converged == 0
        x_new = jnp.where(take, x + alpha * dx, x)
        y_new = jnp.where(take, y + alpha * dy, y)
        v_new = jnp.where(take, v + alpha * dv, v)
        kappa_new = jnp.where(take, kappa + alpha * dk, kappa)
        z_new = jnp.where(take, retraction_map(v_new, kappa_new), z)
        s_new = jnp.where(take, retraction_map(-v_new, kappa_new), s)

        new_state = QPState(x_new, s_new, z_new, y_new)
        return (qp, new_state, converged, pdip_iter + 1)

    def _cond(inputs):
        _, _, converged, pdip_iter = inputs
        return jnp.logical_and(pdip_iter < params.max_iter, converged == 0)

    init = (qp, state, 0, 0)
    if verbose:
        print_header(
            n=Q.shape[0],
            m=A.shape[0],
            p=G.shape[0],
            tol=solver_tol,
            max_iter=max_iter,
            precision="f32" if Q.dtype == jnp.float32 else "f64",
            backend="implicit",
            sigma=sigma,
        )
        print(
            "iter     κ            rt          rc         ri         re           α          σ"  # noqa: E501
        )
        print(
            "----------------------------------------------------------------------------------------"
        )
        outputs = init
        while _cond(outputs):
            outputs = _step(outputs)
    else:
        outputs = jax.lax.while_loop(_cond, _step, init)

    _, final_state, converged, pdip_iter = outputs
    x, s, z, y = final_state

    z = jnp.where(h_mask_is_inf, 0, z)
    s = jnp.where(h_mask_is_inf, jnp.inf, s)

    if verbose:
        print_footer(converged, 0.5 * x @ Q @ x + q @ x, pdip_iter)

    return x, s, z, y, converged, pdip_iter  # type: ignore
