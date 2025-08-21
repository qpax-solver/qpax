"""PDIP functions for solving QP problems."""

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

# FACTORIZATION = "qr"
FACTORIZATION = "cholesky"

def factorized_solve(factorization, rhs):
    """Solve a linear system using QR or Cholesky decomposition."""
    if FACTORIZATION == "cholesky":
        y = jsp.linalg.solve_triangular(factorization, rhs, lower=True)
        x = jsp.linalg.solve_triangular(factorization.T, y, lower=False)
        return x
    elif FACTORIZATION == "qr":
        return jsp.linalg.solve_triangular(factorization[1], factorization[0].T @ rhs)
    else:
        raise ValueError(f"Unknown factorization: {FACTORIZATION}. Must be 'qr' or 'cholesky'")

def factorize(mat):
    if FACTORIZATION == "cholesky":
        return jnp.linalg.cholesky(mat)
    elif FACTORIZATION == "qr":
        return jnp.linalg.qr(mat)
    else:
        raise ValueError(f"Unknown factorization: {FACTORIZATION}. Must be 'qr' or 'cholesky'")


def initialize(Q, q, A, b, G, h):
    """Initialize primal and dual variables from CVXGEN/CVXOPT."""
    H = Q + G.T @ G
    L_H = factorize(H)
    F = A @ factorized_solve(L_H, A.T)
    L_F = factorize(F)

    r1 = -q + G.T @ h
    y = factorized_solve(L_F, A @ factorized_solve(L_H, r1) - b)
    x = factorized_solve(L_H, r1 - A.T @ y)
    z = G @ x - h

    alpha_p = -jnp.min(-z)

    s = jnp.where(alpha_p < 0, -z, -z + (1 + alpha_p))

    alpha_d = -jnp.min(z)

    z = jnp.where(alpha_d >= 0, z + (1 + alpha_d), z)

    return x, s, z, y


def factorize_kkt(Q, G, A, s, z):
    """Factorize the KKT system."""
    P_inv_vec = z / s
    H = Q + G.T @ (G.T * P_inv_vec).T
    L_H = factorize(H)
    F = A @ factorized_solve(L_H, A.T)
    L_F = factorize(F)

    return P_inv_vec, L_H, L_F


def solve_kkt_rhs(Q, G, A, s, z, P_inv_vec, L_H, L_F, v1, v2, v3, v4):
    """
    Solve the KKT system for the residuals v1, v2, v3, v4.

    Algorithm 1 PDIP Linear System Solver
    """
    r2 = v3 - v2 / z
    p1 = v1 + G.T @ (P_inv_vec * r2)

    dy = factorized_solve(L_F, A @ factorized_solve(L_H, p1) - v4)
    dx = factorized_solve(L_H, p1 - A.T @ dy)
    ds = v3 - G @ dx
    dz = (v2 - z * ds) / s

    return dx, ds, dz, dy


def ort_linesearch(x, dx):
    """maximum alpha <= 1 st x + alpha * dx >= 0"""

    def body(_x, _dx):
        return jnp.where(_dx < 0, -_x / _dx, jnp.inf)

    batch = jax.vmap(body, in_axes=(0, 0))
    return jnp.min(jnp.array([1.0, jnp.min(batch(x, dx))]))


def centering_params(s, z, ds_a, dz_a):
    """duality gap + cc term in predictor-corrector PDIP"""
    mu = jnp.dot(s, z) / len(s)

    alpha = jnp.min(jnp.array([ort_linesearch(s, ds_a), ort_linesearch(z, dz_a)]))

    sigma = (jnp.dot(s + alpha * ds_a, z + alpha * dz_a) / jnp.dot(s, z)) ** 3

    return sigma, mu


def pdip_pc_step(inputs):
    """One step of the predictor-corrector PDIP algorithm."""
    # unpack inputs
    Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter = inputs

    # residuals (equation 10)
    r1 = Q @ x + q + A.T @ y + G.T @ z
    r2 = s * z
    r3 = G @ x + s - h
    r4 = A @ x - b

    # check convergence
    kkt_res = jnp.concatenate((r1, r2, r3, r4))
    converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

    # affine step
    P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, z)
    _, ds_a, dz_a, _ = solve_kkt_rhs(Q, G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

    # change centering + correcting params
    sigma, mu = centering_params(s, z, ds_a, dz_a)
    r2 = r2 - (sigma * mu - (ds_a * dz_a))
    dx, ds, dz, dy = solve_kkt_rhs(Q, G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

    # linesearch and update primal & dual vars (equation 11)
    alpha = 0.99 * jnp.min(jnp.array([ort_linesearch(s, ds), ort_linesearch(z, dz)]))

    x = x + alpha * dx
    s = s + alpha * ds
    z = z + alpha * dz
    y = y + alpha * dy

    return (Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter + 1)


# 0 Q
# 1 q
# 2 A
# 3 b
# 4 G
# 5 h
# 6 x
# 7 s
# 8 z
# 9 y
# 10 solver_tol
# 11 converged
# 12 pdip_iter


def solve_eq_only(Q, q, A, b):
    """Solve equality constrained QP (Boyd, Convex, pg 559)."""
    Q_f = factorize(Q)

    Qinv_At = factorized_solve(Q_f, A.T)
    Qinv_g = factorized_solve(Q_f, q)

    S = -A @ Qinv_At
    S_f = factorize(S)
    y = factorized_solve(S_f, A @ Qinv_g + b)

    x = factorized_solve(Q_f, -A.T @ y - q)

    return x, y


# def remove_inf_constraints(G, h):
#     """Remove infinite constraints from G and h."""

#     def body(Grow, hval):
#         isinf = jnp.isinf(hval)
#         new_h_val = jnp.where(isinf, 0, hval)
#         new_G_row = jnp.where(isinf, jnp.zeros(len(Grow)), Grow)
#         return new_G_row, new_h_val

#     G, h = jax.vmap(body, in_axes=(0, 0))(G, h)


#     return G, h
def remove_inf_constraints(G, h):
    """Remove infinite constraints from G and h."""

    # mask for infinite values in h
    h_mask_is_inf = jnp.isinf(h)

    # if h is inf, we set that element to 0
    h2 = jnp.where(h_mask_is_inf, 0, h)

    # remove the rows of G corresponding to inf h values
    G2 = jnp.diag(1 * ~h_mask_is_inf) @ G

    return G2, h2, h_mask_is_inf


def solve_qp(
    Q: jax.Array, q: jax.Array, A: jax.Array, b: jax.Array, G: jax.Array, h: jax.Array, solver_tol: float = 1e-3
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, int, int]:
    """Solve a QP using a primal-dual interior point method.

    Args:
        Q: (n, n) positive definite matrix
        q: (n,) vector
        A: (m, n) equality constraint matrix
        b: (m,) equality constraint vector
        G: (p, n) inequality constraint matrix
        h: (p,) inequality constraint vector

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
        # no constraints so we can solve directly
        x = jnp.linalg.solve(Q, -q)
        return x, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), 1, 0

    if len(h) == 0:
        # only equality constraints, no need for PDIP method
        x, y = solve_eq_only(Q, q, A, b)

        return x, jnp.zeros(0), jnp.zeros(0), y, 1, 0

    # any inf's in constraints bound and we remove the constraint
    G, h, h_mask_is_inf = remove_inf_constraints(G, h)

    # symmetrize Q
    Q = 0.5 * (Q + Q.T)

    x, s, z, y = initialize(Q, q, A, b, G, h)

    # continuation criteria for normal predictor-corrector
    def pc_continuation_criteria(inputs):
        converged = inputs[11]
        pdip_iter = inputs[12]

        return jnp.logical_and(pdip_iter < 30, converged == 0)

    converged = 0
    pdip_iter = 0
    init_inputs = (Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter)

    outputs = jax.lax.while_loop(pc_continuation_criteria, pdip_pc_step, init_inputs)

    x, s, z, y = outputs[6:10]

    # set z to 0 where h was inf
    z = jnp.where(h_mask_is_inf, 0, z)

    # set s to inf where h was inf (this is not necessary)
    s = jnp.where(h_mask_is_inf, jnp.inf, s)

    converged = outputs[11]
    pdip_iter = outputs[12]

    return x, s, z, y, converged, pdip_iter


def pdip_pc_step_debug(inputs):
    # unpack inputs
    Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter = inputs

    # residuals
    r1 = Q @ x + q + A.T @ y + G.T @ z
    r2 = s * z
    r3 = G @ x + s - h
    r4 = A @ x - b

    # check convergence
    kkt_res = jnp.concatenate((r1, r2, r3, r4))
    converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

    # affine step
    P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, z)
    _, ds_a, dz_a, _ = solve_kkt_rhs(Q, G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

    # change centering + correcting params
    sigma, mu = centering_params(s, z, ds_a, dz_a)
    r2 = r2 - (sigma * mu - (ds_a * dz_a))
    dx, ds, dz, dy = solve_kkt_rhs(Q, G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

    # linesearch and update primal & dual vars
    alpha = 0.99 * jnp.min(jnp.array([1.0, 0.99 * ort_linesearch(s, ds), 0.99 * ort_linesearch(z, dz)]))

    x = x + alpha * dx
    s = s + alpha * ds
    z = z + alpha * dz
    y = y + alpha * dy

    if len(r4) == 0:
        r4 = jnp.zeros(1)
    print(
        "%3d   %9.2e   %9.2e  %9.2e  %9.2e   % 6.4f"
        % (
            pdip_iter,
            jnp.linalg.norm(r1, ord=jnp.inf),
            jnp.linalg.norm(r2, ord=jnp.inf),
            jnp.linalg.norm(r3, ord=jnp.inf),
            jnp.linalg.norm(r4, ord=jnp.inf),
            alpha,
        )
    )

    return (Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter + 1)


def while_loop_debug(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def solve_qp_debug(Q, q, A, b, G, h, solver_tol=1e-3):
    """Debug solving with verbose printing."""
    # make sure each matrix is 2D
    Q = jnp.atleast_2d(Q)
    A = jnp.atleast_2d(A)
    G = jnp.atleast_2d(G)

    if (len(b) == 0) and (len(h) == 0):
        # no constraints so we can solve directly
        x = jnp.linalg.solve(Q, -q)
        return x, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), 1, 0

    if len(h) == 0:
        # only equality constraints, no need for PDIP method
        x, y = solve_eq_only(Q, q, A, b)

        return x, jnp.zeros(0), jnp.zeros(0), y, 1, 0

    # any inf's in constraints bound and we remove the constraint
    G, h, h_mask_is_inf = remove_inf_constraints(G, h)

    # symmetrize Q
    Q = 0.5 * (Q + Q.T)

    x, s, z, y = initialize(Q, q, A, b, G, h)

    # continuation criteria for normal predictor-corrector
    def pc_continuation_criteria(inputs):
        converged = inputs[11]
        pdip_iter = inputs[12]

        return jnp.logical_and(pdip_iter < 30, converged == 0)

    converged = 0
    pdip_iter = 0
    init_inputs = (Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter)

    print("iter      r1          r2         r3         r4        alpha")
    print("------------------------------------------------------")

    outputs = while_loop_debug(pc_continuation_criteria, pdip_pc_step_debug, init_inputs)

    x, s, z, y = outputs[6:10]
    # set z to 0 where h was inf
    z = jnp.where(h_mask_is_inf, 0, z)

    # set s to inf where h was inf (this is not necessary)
    s = jnp.where(h_mask_is_inf, jnp.inf, s)
    converged = outputs[11]
    pdip_iter = outputs[12]

    return x, s, z, y, converged, pdip_iter
