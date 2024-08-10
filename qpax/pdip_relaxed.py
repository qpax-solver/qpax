import jax
import jax.numpy as jnp

from qpax.pdip import factorize_kkt, ort_linesearch, solve_kkt_rhs


def pdip_newton_step(inputs):
    """
    Algorithm 3 Relaxing a Quadratic Program
    """

    # unpack inputs
    Q, q, A, b, G, h, x, s, z, y, solver_tol, converged, pdip_iter, target_kappa = inputs

    # evaluate relaxed KKT conditions
    r1 = Q @ x + q + A.T @ y + G.T @ z
    r2 = s * z - target_kappa  # we added this (relaxed complementarity)
    r3 = G @ x + s - h
    r4 = A @ x - b

    # check convergence
    kkt_res = jnp.concatenate((r1, r2, r3, r4))
    converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

    # calculate and take Newton step
    P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, z)
    dx, ds, dz, dy = solve_kkt_rhs(Q, G, A, s, z, P_inv_vec, L_H, L_F, -r1, -r2, -r3, -r4)

    # linesearch and update primal & dual vars
    alpha = 0.99 * jnp.min(jnp.array([1.0, 0.99 * ort_linesearch(s, ds), 0.99 * ort_linesearch(z, dz)]))

    x = x + alpha * dx
    s = s + alpha * ds
    z = z + alpha * dz
    y = y + alpha * dy

    return (
        Q,
        q,
        A,
        b,
        G,
        h,
        x,
        s,
        z,
        y,
        solver_tol,
        converged,
        pdip_iter + 1,
        target_kappa,
    )


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


def relax_qp(Q, q, A, b, G, h, x, s, z, y, solver_tol=1e-5, target_kappa=1e-5, max_iter=30):
    # continuation criteria for normal predictor-corrector
    def relaxed_continuation_criteria(inputs):
        converged = inputs[11]
        pdip_iter = inputs[12]

        return jnp.logical_and(pdip_iter < max_iter, converged == 0)

    converged = 0
    pdip_iter = 0
    init_inputs = (
        Q,
        q,
        A,
        b,
        G,
        h,
        x,
        s,
        z,
        y,
        solver_tol,
        converged,
        pdip_iter,
        target_kappa,
    )

    outputs = jax.lax.while_loop(relaxed_continuation_criteria, pdip_newton_step, init_inputs)

    x_rlx, s_rlx, z_rlx, y_rlx = outputs[6:10]
    converged = outputs[11]
    pdip_iter = outputs[12]

    return x_rlx, s_rlx, z_rlx, y_rlx, converged, pdip_iter
