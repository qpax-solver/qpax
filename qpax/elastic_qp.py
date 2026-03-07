import jax
import jax.numpy as jnp

from qpax.pdip import centering_params, ort_linesearch, qr_solve

DEBUG_FLAG = False


def solve_init_elastic_ls(Q, q, G, h, penalty):
    ns = len(h)
    r1 = -q
    r2 = penalty * jnp.ones(ns)
    r4 = h

    L_H = jnp.linalg.qr(Q + 0.5 * G.T @ G)
    x = qr_solve(L_H, r1 - 0.5 * G.T @ (r2 - r4))
    z2 = 0.5 * (G @ x + r2 - r4)
    z1 = r2 - z2
    t = -z1

    x_big = jnp.concatenate((x, t))
    z_big = jnp.concatenate((z1, z2))

    return x_big, z_big


def initialize_elastic(Q, q, G, h, penalty):
    x_big, z_big = solve_init_elastic_ls(Q, q, G, h, penalty)

    alpha_p = -jnp.min(-z_big)

    s_big = jnp.where(alpha_p < 0, -z_big, -z_big + (1 + alpha_p))

    alpha_d = -jnp.min(z_big)

    z_big = jnp.where(alpha_d >= 0, z_big + (1 + alpha_d), z_big)

    nx = len(q)
    ns = len(h)
    x = x_big[:nx]
    t = x_big[nx:]
    s1 = s_big[:ns]
    s2 = s_big[ns:]
    z1 = z_big[:ns]
    z2 = z_big[ns:]

    return x, t, s1, s2, z1, z2


def solve_elastic_kkt_affine(s1, z1, s2, z2, Q, G, r1, r2, r3, r4, r5, r6):
    # Algorithm 5 Solve Elastic KKT Linear System
    # solve main KKT linear systems
    a1 = s1 / z1
    a2 = s2 / z2
    w1 = r3 / z1
    w2 = r4 / z2
    p1 = r5 - r6 + w2 - w1 - a1 * r2
    a3 = a1 + a2

    # one linear system solve
    H = Q + G.T @ (G.T * (1 / a3)).T
    L_H = jnp.linalg.qr(H)
    dx = qr_solve(L_H, r1 - G.T @ (p1 / a3))

    # rest is easy
    dz2 = (p1 + G @ dx) / a3
    dz1 = -r2 - dz2
    ds1 = (r3 - s1 * dz1) / z1
    ds2 = (r4 - s2 * dz2) / z2
    dt = ds1 - r5

    return dx, dt, ds1, ds2, dz1, dz2, L_H


def solve_elastic_kkt_cc(L_H, s1, z1, s2, z2, Q, G, r1, r2, r3, r4, r5, r6):
    # Algorithm 5 Solve Elastic KKT Linear System
    # solve main KKT linear systems
    a1 = s1 / z1
    a2 = s2 / z2
    w1 = r3 / z1
    w2 = r4 / z2
    p1 = r5 - r6 + w2 - w1 - a1 * r2
    a3 = a1 + a2

    # one linear system solve
    # dx = jnp.linalg.solve(Q + G.T @ (G.T * (1 / a3)).T, r1 - G.T @ (p1 / a3))
    dx = qr_solve(L_H, r1 - G.T @ (p1 / a3))

    # rest is easy
    dz2 = (p1 + G @ dx) / a3
    dz1 = -r2 - dz2
    ds1 = (r3 - s1 * dz1) / z1
    ds2 = (r4 - s2 * dz2) / z2
    dt = ds1 - r5

    return dx, dt, ds1, ds2, dz1, dz2


def pdip_pc_step_elastic(inputs):
    # Algorithm 4 PDIP Method for Elastic Quadratic Programs
    # unpack inputs
    Q, q, G, h, penalty, x, t, s1, s2, z1, z2, solver_tol, converged, pdip_iter = inputs

    # evaluate KKT conditions and check convergence
    r1 = Q @ x + q + G.T @ z2
    r2 = -z1 - z2 + penalty * jnp.ones(len(h))
    r3 = s1 * z1
    r4 = s2 * z2
    r5 = -t + s1
    r6 = G @ x - t + s2 - h

    # check convergence
    kkt_res = jnp.concatenate((r1, r2, r3, r4, r5, r6))
    converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

    # calculate affine step direction Alg. (5)
    _, _, ds1_a, ds2_a, dz1_a, dz2_a, L_H = solve_elastic_kkt_affine(s1, z1, s2, z2, Q, G, -r1, -r2, -r3, -r4, -r5, -r6)

    s = jnp.concatenate((s1, s2))
    z = jnp.concatenate((z1, z2))
    ds_a = jnp.concatenate((ds1_a, ds2_a))
    dz_a = jnp.concatenate((dz1_a, dz2_a))

    # change centering + correcting params
    sigma, mu = centering_params(s, z, ds_a, dz_a)
    r3 = r3 - (sigma * mu - (ds1_a * dz1_a))
    r4 = r4 - (sigma * mu - (ds2_a * dz2_a))

    dx, dt, ds1, ds2, dz1, dz2 = solve_elastic_kkt_cc(L_H, s1, z1, s2, z2, Q, G, -r1, -r2, -r3, -r4, -r5, -r6)

    ds = jnp.concatenate((ds1, ds2))
    dz = jnp.concatenate((dz1, dz2))

    # linesearch and update primal & dual vars
    alpha = 0.99 * jnp.min(jnp.array([1.0, 0.99 * ort_linesearch(s, ds), 0.99 * ort_linesearch(z, dz)]))

    x = x + alpha * dx
    t = t + alpha * dt
    s1 = s1 + alpha * ds1
    s2 = s2 + alpha * ds2
    z1 = z1 + alpha * dz1
    z2 = z2 + alpha * dz2

    if DEBUG_FLAG:
        print(
            "%3d   %9.2e   %9.2e  %9.2e  %9.2e   %9.2e  %9.2e   %6.4f"
            % (
                pdip_iter,
                jnp.linalg.norm(r1, ord=jnp.inf),
                jnp.linalg.norm(r2, ord=jnp.inf),
                jnp.linalg.norm(r3, ord=jnp.inf),
                jnp.linalg.norm(r4, ord=jnp.inf),
                jnp.linalg.norm(r5, ord=jnp.inf),
                jnp.linalg.norm(r6, ord=jnp.inf),
                alpha,
            )
        )

    return (
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol,
        converged,
        pdip_iter + 1,
    )


def while_loop_debug(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def solve_qp_elastic(Q, q, G, h, penalty, solver_tol=1e-3, max_iter=30):
    # symmetrize Q
    Q = 0.5 * (Q + Q.T)

    x, t, s1, s2, z1, z2 = initialize_elastic(Q, q, G, h, penalty)

    # continuation criteria for normal predictor-corrector
    def pc_continuation_criteria(inputs):
        converged = inputs[12]
        pdip_iter = inputs[13]

        return jnp.logical_and(pdip_iter < max_iter, converged == 0)

    converged = 0
    pdip_iter = 0
    init_inputs = (
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol,
        converged,
        pdip_iter,
    )

    if DEBUG_FLAG:
        print("iter      r1          r2         r3         r4        r5        r6        alpha")
        print("--------------------------------------------------------------------------------")

        outputs = while_loop_debug(pc_continuation_criteria, pdip_pc_step_elastic, init_inputs)
    else:
        outputs = jax.lax.while_loop(pc_continuation_criteria, pdip_pc_step_elastic, init_inputs)

    x, t, s1, s2, z1, z2 = outputs[5:11]
    converged = outputs[12]
    pdip_iter = outputs[13]

    return x, t, s1, s2, z1, z2, converged, pdip_iter


# if not DEBUG_FLAG:
#     solve_qp_elastic = jax.jit(solve_qp_elastic)


def pdip_newton_step_elastic(inputs):
    # unpack inputs
    (
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol,
        converged,
        pdip_iter,
        target_kappa,
    ) = inputs

    # residuals
    r1 = Q @ x + q + G.T @ z2
    r2 = z1 + z2 - penalty * jnp.ones(len(h))
    r3 = s1 * z1 - target_kappa
    r4 = s2 * z2 - target_kappa
    r5 = t + s1
    r6 = G @ x + t + s2 - h

    # check convergence
    kkt_res = jnp.concatenate((r1, r2, r3, r4, r5, r6))
    converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

    # affine step
    dx, dt, ds1, ds2, dz1, dz2, _ = solve_elastic_kkt_affine(s1, z1, s2, z2, Q, G, -r1, -r2, -r3, -r4, -r5, -r6)

    s = jnp.concatenate((s1, s2))
    z = jnp.concatenate((z1, z2))
    ds = jnp.concatenate((ds1, ds2))
    dz = jnp.concatenate((dz1, dz2))

    # linesearch and update primal & dual vars
    alpha = 0.99 * jnp.min(jnp.array([1.0, 0.99 * ort_linesearch(s, ds), 0.99 * ort_linesearch(z, dz)]))

    x = x + alpha * dx
    t = t + alpha * dt
    s1 = s1 + alpha * ds1
    s2 = s2 + alpha * ds2
    z1 = z1 + alpha * dz1
    z2 = z2 + alpha * dz2

    if DEBUG_FLAG:
        print(
            "%3d   %9.2e   %9.2e  %9.2e  %9.2e   %9.2e  %9.2e   % 6.4f"
            % (
                pdip_iter,
                jnp.linalg.norm(r1, ord=jnp.inf),
                jnp.linalg.norm(r2, ord=jnp.inf),
                jnp.linalg.norm(r3, ord=jnp.inf),
                jnp.linalg.norm(r4, ord=jnp.inf),
                jnp.linalg.norm(r5, ord=jnp.inf),
                jnp.linalg.norm(r6, ord=jnp.inf),
                alpha,
            )
        )

    return (
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol,
        converged,
        pdip_iter + 1,
        target_kappa,
    )


def relax_qp_elastic(Q, q, G, h, penalty, x, t, s1, s2, z1, z2, solver_tol=1e-3, target_kappa=1e-3, max_iter=30):
    # continuation criteria for normal predictor-corrector
    def pc_continuation_criteria(inputs):
        converged = inputs[12]
        pdip_iter = inputs[13]

        return jnp.logical_and(pdip_iter < max_iter, converged == 0)

    converged = 0
    pdip_iter = 0
    init_inputs = (
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol,
        converged,
        pdip_iter,
        target_kappa,
    )

    if DEBUG_FLAG:
        print("iter      r1          r2         r3         r4        r5        r6        alpha")
        print("--------------------------------------------------------------------------------")

        outputs = while_loop_debug(pc_continuation_criteria, pdip_newton_step_elastic, init_inputs)
    else:
        outputs = jax.lax.while_loop(pc_continuation_criteria, pdip_newton_step_elastic, init_inputs)

    x_rlx, t_rlx, s1_rlx, s2_rlx, z1_rlx, z2_rlx = outputs[5:11]
    converged = outputs[12]
    pdip_iter = outputs[13]

    return x_rlx, t_rlx, s1_rlx, s2_rlx, z1_rlx, z2_rlx, converged, pdip_iter


# if not DEBUG_FLAG:
#     relax_qp_elastic = jax.jit(relax_qp_elastic)


def optnet_derivatives_elastic(dz, dlam, z, lam):
    dl_dQ = 0.5 * (jnp.outer(dz, z) + jnp.outer(z, dz))
    dl_dG = jnp.diag(lam) @ (jnp.outer(dlam, z) + jnp.outer(lam, dz))  # TODO

    dl_dq = dz
    dl_dh = -lam * dlam

    return dl_dQ, dl_dq, dl_dG, dl_dh


# def diff_qp_elastic(Q,q,G,h,
#                     z,
#                     s,
#                     lam,
#                     dl_dz):
def diff_qp_elastic(Q, q, G, h, x, t, s1, s2, lam1, lam2, dl_dz):
    #     xr, tr, s1r, s2r, z1r, z2r
    # everything normal size
    nz = len(q)
    ns = len(h)

    zns = jnp.zeros(ns)
    dx, dt, ds1, ds2, dlam1, dlam2, _ = solve_elastic_kkt_affine(
        s1, lam1, s2, lam2, Q, G, -dl_dz, zns, zns, zns, zns, zns
    )

    # augment the sizes to make them big again
    dz = jnp.concatenate((dx, dt))
    z = jnp.concatenate((x, t))
    # ds = jnp.concatenate((ds1, ds2))
    # s = jnp.concatenate((s1, s2))
    lam = jnp.concatenate((lam1, lam2))
    dlam_tilde = jnp.concatenate((dlam1, dlam2))

    # recover real dlam from our modified (symmetrized) KKT system
    dlam = dlam_tilde / lam

    dl_dQ, dl_dq, dl_dG, dl_dh = optnet_derivatives_elastic(dz, dlam, z, lam)

    # reduce these down to our normal sizes again
    #     G2 = jnp.block([
    #         [zsx, jnp.eye(ns)],
    #         [G,   jnp.eye(ns)]
    #     ])
    #     h2 = jnp.block([jnp.zeros(ns), h])

    dl_dQ = dl_dQ[:nz, :nz]
    dl_dq = dl_dq[:nz]
    dl_dG = dl_dG[ns:, :nz]
    dl_dh = dl_dh[ns:]

    return dl_dQ, dl_dq, dl_dG, dl_dh


@jax.custom_vjp
def solve_qp_elastic_primal(Q, q, G, h, penalty, solver_tol=1e-5, target_kappa=1e-3, max_iter=30):
    # solve qp as normal and return primal solution (use any solver)
    x, t, s1, s2, z1, z2, converged, pdip_iter = solve_qp_elastic(Q, q, G, h, penalty, solver_tol=solver_tol, max_iter=max_iter)

    return x


"""
these two functions are only called when we diff solve_qp_x
"""


def solve_qp_elastic_primal_fwd(Q, q, G, h, penalty, solver_tol=1e-5, target_kappa=1e-3, max_iter=30):
    # solve qp as normal and return primal solution (use any solver)

    x, t, s1, s2, z1, z2, converged1, pdip_iter1 = solve_qp_elastic(Q, q, G, h, penalty, solver_tol=solver_tol, max_iter=max_iter)

    # relax this solution by taking vanilla Newton steps on relaxed KKT
    xr, tr, s1r, s2r, z1r, z2r, converged2, pdip_iter2 = relax_qp_elastic(
        Q,
        q,
        G,
        h,
        penalty,
        x,
        t,
        s1,
        s2,
        z1,
        z2,
        solver_tol=solver_tol,
        target_kappa=target_kappa,
        max_iter=max_iter,
    )

    # return real solution x, and save the relaxed variables for backward
    return x, (Q, q, G, h, penalty, xr, tr, s1r, s2r, z1r, z2r)


def solve_qp_elastic_primal_bwd(res, input_grad):
    # unpack relaxed solution
    Q, q, G, h, penalty, xr, tr, s1r, s2r, z1r, z2r = res

    # return all the normal derivatives, then None's for penalty and kwargs
    dl_dQ, dl_dq, dl_dG, dl_dh = diff_qp_elastic(Q, q, G, h, xr, tr, s1r, s2r, z1r, z2r, input_grad)

    # Final Nones for penalty and kwargs
    return (dl_dQ, dl_dq, dl_dG, dl_dh, None, None, None, None)


solve_qp_elastic_primal.defvjp(solve_qp_elastic_primal_fwd, solve_qp_elastic_primal_bwd)
