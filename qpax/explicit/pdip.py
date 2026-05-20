"""PDIP functions for solving QP problems."""

from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from qpax._verbose import print_footer, print_header


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
    tol: float = 1e-5
    max_iter: int = 30
    linear_solver: LinearSolver = LinearSolver.CHOLESKY


class QPState(NamedTuple):
    x: jax.Array
    s: jax.Array
    z: jax.Array
    y: jax.Array


def _all_finite(*arrays) -> jax.Array:
    """Return True iff every provided array/scalar is finite."""
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(jnp.ravel(a))) for a in arrays]))


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
    solver_tol: float = 1e-5,
    max_iter: int = 30,
    linear_solver: LinearSolver = LinearSolver.CHOLESKY,
    return_bad_step: bool = False,
    sigma: float = 0.125,  # noqa: ARG001 — accepted for API uniformity; explicit uses Mehrotra centering
    verbose: bool = False,
):
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
        outputs = (x, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), 1, 0)
        return (*outputs, False) if return_bad_step else outputs

    if len(h) == 0:
        x, y = solve_eq_only(Q, q, A, b, linear_solver)
        outputs = (x, jnp.zeros(0), jnp.zeros(0), y, 1, 0)
        return (*outputs, False) if return_bad_step else outputs

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
        qp, st, converged, pdip_iter, bad_step_seen = inputs
        Q, q, A, b, G, h = qp
        x, s, z, y = st
        s_prev = s
        z_prev = z

        # Match relax_qp's protection (pdip_relaxed.py): floor s,z so that
        # cond(H = Q + Gᵀ diag(z/s) G) stays within reach of f32 Cholesky.
        # Under vmap, the slowest lane drags every other lane through extra
        # iterations; without this floor those iterations drive s,z below
        # f32 epsilon and produce NaN in the factorization.
        floor = jnp.sqrt(jnp.finfo(s.dtype).eps)
        s = jnp.maximum(s, floor)
        z = jnp.maximum(z, floor)

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
        step_finite = _all_finite(
            P_inv_vec, ds_a, dz_a, sigma, mu, dx, ds, dz, dy, alpha
        )
        bad_step_seen = jnp.logical_or(bad_step_seen, jnp.logical_not(step_finite))

        if verbose:
            r4_print = r4 if len(r4) > 0 else jnp.zeros(1)
            print(
                f"{pdip_iter:3d}   {jnp.linalg.norm(r1, ord=jnp.inf):9.2e}"
                f"   {jnp.linalg.norm(r2, ord=jnp.inf):9.2e}"
                f"  {jnp.linalg.norm(r3, ord=jnp.inf):9.2e}"
                f"  {jnp.linalg.norm(r4_print, ord=jnp.inf):9.2e}"
                f"   {alpha:6.4f}  {mu:9.4f}   {sigma:9.4f}"
            )

        # Under vmap, jax.lax.while_loop runs every lane until the slowest
        # one converges; freezing already-converged lanes prevents f32
        # Newton-step noise from drifting the state into NaN.
        take = converged == 0
        new_state = QPState(
            jnp.where(take, x + alpha * dx, x),
            jnp.where(take, s + alpha * ds, s_prev),
            jnp.where(take, z + alpha * dz, z_prev),
            jnp.where(take, y + alpha * dy, y),
        )
        return (qp, new_state, converged, pdip_iter + 1, bad_step_seen)

    def _cond(inputs):
        _, _, converged, pdip_iter, _ = inputs
        return jnp.logical_and(pdip_iter < params.max_iter, converged == 0)

    init = (qp, state, 0, 0, jnp.bool_(False))
    if verbose:
        print_header(
            n=Q.shape[0],
            m=A.shape[0],
            p=G.shape[0],
            tol=solver_tol,
            max_iter=max_iter,
            precision="f32" if Q.dtype == jnp.float32 else "f64",
            backend="explicit",
        )
        print(
            "iter      rt          rc         ri         re        α         mu         σ"  # noqa: E501
        )
        print(
            "----------------------------------------------------------------------------------"
        )
        outputs = init
        while _cond(outputs):
            outputs = _step(outputs)
    else:
        outputs = jax.lax.while_loop(_cond, _step, init)

    _, final_state, converged, pdip_iter, bad_step_seen = outputs
    x, s, z, y = final_state

    z = jnp.where(h_mask_is_inf, 0, z)
    s = jnp.where(h_mask_is_inf, jnp.inf, s)

    if verbose:
        print_footer(converged, 0.5 * x @ Q @ x + q @ x, pdip_iter)

    outputs = (x, s, z, y, converged, pdip_iter)
    return (*outputs, bad_step_seen) if return_bad_step else outputs  # type: ignore


def solve_qp_group_flags(
    Q: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    G: jax.Array,
    h: jax.Array,
    solver_tol: float = 1e-5,
    max_iter: int = 30,
    linear_solver: LinearSolver = LinearSolver.CHOLESKY,
):
    """Diagnostic copy of solve_qp that returns per-operation first-fire iters.

    For each of the five subgroups (scaling, predictor, centering, corrector,
    linesearch), the returned int32 is the PDIP iteration at which a non-finite
    intermediate was FIRST observed in that group (0-indexed), or `max_iter + 1`
    if never. This preserves time ordering across iterations, unlike a
    cumulative bool flag.
    """
    Q = jnp.atleast_2d(Q)
    A = jnp.atleast_2d(A)
    G = jnp.atleast_2d(G)

    sentinel = jnp.int32(max_iter + 2)
    never_iters = (sentinel,) * 5

    if (len(b) == 0) and (len(h) == 0):
        x = jnp.linalg.solve(Q, -q)
        return (x, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), 1, 0, *never_iters)

    if len(h) == 0:
        x, y = solve_eq_only(Q, q, A, b, linear_solver)
        return (x, jnp.zeros(0), jnp.zeros(0), y, 1, 0, *never_iters)

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

    def _update_iter(current, bad_this_iter, pdip_iter):
        return jnp.where(
            jnp.logical_and(bad_this_iter, current == sentinel),
            pdip_iter.astype(jnp.int32),
            current,
        )

    def _step(inputs):
        (
            qp,
            st,
            converged,
            pdip_iter,
            scaling_iter,
            predictor_iter,
            centering_iter,
            corrector_iter,
            linesearch_iter,
        ) = inputs
        Q, q, A, b, G, h = qp
        x, s, z, y = st
        s_prev = s
        z_prev = z

        # Mirror solve_qp's s,z floor so this diagnostic version follows the
        # same numerical path as production. Without it the predictor's
        # `(v2 - z*ds)/s` overflows in f32 long before sigma can fire, which
        # mis-attributes the first-NaN site.
        floor = jnp.sqrt(jnp.finfo(s.dtype).eps)
        s = jnp.maximum(s, floor)
        z = jnp.maximum(z, floor)

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

        scaling_bad = jnp.logical_not(_all_finite(P_inv_vec))
        predictor_bad = jnp.logical_not(_all_finite(ds_a, dz_a))
        centering_bad = jnp.logical_not(_all_finite(sigma, mu))
        corrector_bad = jnp.logical_not(_all_finite(dx, ds, dz, dy))
        linesearch_bad = jnp.logical_not(_all_finite(alpha))

        scaling_iter = _update_iter(scaling_iter, scaling_bad, pdip_iter)
        predictor_iter = _update_iter(predictor_iter, predictor_bad, pdip_iter)
        centering_iter = _update_iter(centering_iter, centering_bad, pdip_iter)
        corrector_iter = _update_iter(corrector_iter, corrector_bad, pdip_iter)
        linesearch_iter = _update_iter(linesearch_iter, linesearch_bad, pdip_iter)

        take = converged == 0
        new_state = QPState(
            jnp.where(take, x + alpha * dx, x),
            jnp.where(take, s + alpha * ds, s_prev),
            jnp.where(take, z + alpha * dz, z_prev),
            jnp.where(take, y + alpha * dy, y),
        )
        return (
            qp,
            new_state,
            converged,
            pdip_iter + 1,
            scaling_iter,
            predictor_iter,
            centering_iter,
            corrector_iter,
            linesearch_iter,
        )

    def _cond(inputs):
        _, _, converged, pdip_iter, *_ = inputs
        return jnp.logical_and(pdip_iter < params.max_iter, converged == 0)

    init = (qp, state, 0, jnp.int32(0), *never_iters)
    outputs = jax.lax.while_loop(_cond, _step, init)

    (
        _,
        final_state,
        converged,
        pdip_iter,
        scaling_iter,
        predictor_iter,
        centering_iter,
        corrector_iter,
        linesearch_iter,
    ) = outputs
    x, s, z, y = final_state

    z = jnp.where(h_mask_is_inf, 0, z)
    s = jnp.where(h_mask_is_inf, jnp.inf, s)

    return (
        x,
        s,
        z,
        y,
        converged,
        pdip_iter,
        scaling_iter,
        predictor_iter,
        centering_iter,
        corrector_iter,
        linesearch_iter,
    )
