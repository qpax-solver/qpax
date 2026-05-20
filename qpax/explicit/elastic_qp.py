"""Elastic QP solver using PDIP."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from qpax.explicit.pdip import (
    LinearSolver,
    SolverParams,
    _factor,
    _solve,
    centering_params,
    ort_linesearch,
)

DEBUG_FLAG = False


class ElasticQPData(NamedTuple):
    Q: jax.Array
    q: jax.Array
    G: jax.Array
    h: jax.Array
    penalty: jax.Array


class ElasticQPState(NamedTuple):
    x: jax.Array
    t: jax.Array
    s1: jax.Array
    s2: jax.Array
    z1: jax.Array
    z2: jax.Array


def solve_init_elastic_ls(qp: ElasticQPData, solver: LinearSolver):
    Q, q, G, h, penalty = qp
    ns = len(h)
    r1 = -q
    r2 = penalty * jnp.ones(ns)
    r4 = h

    L_H = _factor(Q + 0.5 * G.T @ G, solver)
    x = _solve(L_H, r1 - 0.5 * G.T @ (r2 - r4), solver)
    z2 = 0.5 * (G @ x + r2 - r4)
    z1 = r2 - z2
    t = -z1

    x_big = jnp.concatenate((x, t))
    z_big = jnp.concatenate((z1, z2))

    return x_big, z_big


def initialize_elastic(
    qp: ElasticQPData,
    solver: LinearSolver = LinearSolver.QR,
) -> ElasticQPState:
    x_big, z_big = solve_init_elastic_ls(qp, solver)

    alpha_p = -jnp.min(-z_big)
    s_big = jnp.where(alpha_p < 0, -z_big, -z_big + (1 + alpha_p))

    alpha_d = -jnp.min(z_big)
    z_big = jnp.where(alpha_d >= 0, z_big + (1 + alpha_d), z_big)

    nx = len(qp.q)
    ns = len(qp.h)
    return ElasticQPState(
        x=x_big[:nx],
        t=x_big[nx:],
        s1=s_big[:ns],
        s2=s_big[ns:],
        z1=z_big[:ns],
        z2=z_big[ns:],
    )


def solve_elastic_kkt_affine(
    s1,
    z1,
    s2,
    z2,
    Q,
    G,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    solver: LinearSolver = LinearSolver.QR,
):
    # Algorithm 5 Solve Elastic KKT Linear System
    a1 = s1 / z1
    a2 = s2 / z2
    w1 = r3 / z1
    w2 = r4 / z2
    p1 = r5 - r6 + w2 - w1 - a1 * r2
    a3 = a1 + a2

    H = Q + G.T @ (G.T * (1 / a3)).T
    L_H = _factor(H, solver)
    dx = _solve(L_H, r1 - G.T @ (p1 / a3), solver)

    dz2 = (p1 + G @ dx) / a3
    dz1 = -r2 - dz2
    ds1 = (r3 - s1 * dz1) / z1
    ds2 = (r4 - s2 * dz2) / z2
    dt = ds1 - r5

    return dx, dt, ds1, ds2, dz1, dz2, L_H


def solve_elastic_kkt_cc(
    L_H,
    s1,
    z1,
    s2,
    z2,
    G,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    solver: LinearSolver = LinearSolver.QR,
):
    # Algorithm 5 Solve Elastic KKT Linear System (reuse factorization)
    a1 = s1 / z1
    a2 = s2 / z2
    w1 = r3 / z1
    w2 = r4 / z2
    p1 = r5 - r6 + w2 - w1 - a1 * r2
    a3 = a1 + a2

    dx = _solve(L_H, r1 - G.T @ (p1 / a3), solver)

    dz2 = (p1 + G @ dx) / a3
    dz1 = -r2 - dz2
    ds1 = (r3 - s1 * dz1) / z1
    ds2 = (r4 - s2 * dz2) / z2
    dt = ds1 - r5

    return dx, dt, ds1, ds2, dz1, dz2


def solve_qp_elastic(Q, q, G, h, penalty, solver_tol=1e-3, max_iter=30):
    Q = 0.5 * (Q + Q.T)

    params = SolverParams(tol=solver_tol, max_iter=max_iter)
    qp = ElasticQPData(Q, q, G, h, penalty)
    state = initialize_elastic(qp)
    ls = LinearSolver.QR

    nonlocal_tol = params.tol

    def _step(inputs):
        qp, st, converged, pdip_iter = inputs
        Q, q, G, h, penalty = qp
        x, t, s1, s2, z1, z2 = st

        r1 = Q @ x + q + G.T @ z2
        r2 = -z1 - z2 + penalty * jnp.ones(len(h))
        r3 = s1 * z1
        r4 = s2 * z2
        r5 = -t + s1
        r6 = G @ x - t + s2 - h

        kkt_res = jnp.concatenate((r1, r2, r3, r4, r5, r6))
        converged = jnp.where(
            jnp.linalg.norm(kkt_res, ord=jnp.inf) < nonlocal_tol, 1, 0
        )

        # affine step
        _, _, ds1_a, ds2_a, dz1_a, dz2_a, L_H = solve_elastic_kkt_affine(
            s1, z1, s2, z2, Q, G, -r1, -r2, -r3, -r4, -r5, -r6, ls
        )

        s = jnp.concatenate((s1, s2))
        z = jnp.concatenate((z1, z2))
        ds_a = jnp.concatenate((ds1_a, ds2_a))
        dz_a = jnp.concatenate((dz1_a, dz2_a))

        # centering + correcting
        sigma, mu = centering_params(s, z, ds_a, dz_a)
        r3 = r3 - (sigma * mu - (ds1_a * dz1_a))
        r4 = r4 - (sigma * mu - (ds2_a * dz2_a))

        dx, dt, ds1, ds2, dz1, dz2 = solve_elastic_kkt_cc(
            L_H, s1, z1, s2, z2, G, -r1, -r2, -r3, -r4, -r5, -r6, ls
        )

        ds = jnp.concatenate((ds1, ds2))
        dz = jnp.concatenate((dz1, dz2))

        alpha = 0.99 * jnp.min(
            jnp.array(
                [
                    1.0,
                    0.99 * ort_linesearch(s, ds),
                    0.99 * ort_linesearch(z, dz),
                ]
            )
        )

        take = converged == 0
        new_state = ElasticQPState(
            jnp.where(take, x + alpha * dx, x),
            jnp.where(take, t + alpha * dt, t),
            jnp.where(take, s1 + alpha * ds1, s1),
            jnp.where(take, s2 + alpha * ds2, s2),
            jnp.where(take, z1 + alpha * dz1, z1),
            jnp.where(take, z2 + alpha * dz2, z2),
        )

        if DEBUG_FLAG:
            nr1 = jnp.linalg.norm(r1, ord=jnp.inf)
            nr2 = jnp.linalg.norm(r2, ord=jnp.inf)
            nr3 = jnp.linalg.norm(r3, ord=jnp.inf)
            nr4 = jnp.linalg.norm(r4, ord=jnp.inf)
            nr5 = jnp.linalg.norm(r5, ord=jnp.inf)
            nr6 = jnp.linalg.norm(r6, ord=jnp.inf)
            print(
                f"{pdip_iter:3d}   {nr1:9.2e}   {nr2:9.2e}"
                f"  {nr3:9.2e}  {nr4:9.2e}   {nr5:9.2e}"
                f"  {nr6:9.2e}   {alpha:6.4f}"
            )

        return (qp, new_state, converged, pdip_iter + 1)

    def _cond(inputs):
        _, _, converged, pdip_iter = inputs
        return jnp.logical_and(pdip_iter < params.max_iter, converged == 0)

    init = (qp, state, 0, 0)

    if DEBUG_FLAG:
        print(
            "iter      r1          r2         r3         r4        r5        r6"
            "        alpha"
        )
        print("-" * 80)
        val = init
        while _cond(val):
            val = _step(val)
        outputs = val
    else:
        outputs = jax.lax.while_loop(_cond, _step, init)

    _, final_state, converged, pdip_iter = outputs
    x, t, s1, s2, z1, z2 = final_state

    return x, t, s1, s2, z1, z2, converged, pdip_iter


def relax_qp_elastic(
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
    solver_tol=1e-3,
    target_kappa=1e-3,
    max_iter=30,
):
    params = SolverParams(tol=solver_tol, max_iter=max_iter)
    qp = ElasticQPData(Q, q, G, h, penalty)
    state = ElasticQPState(x, t, s1, s2, z1, z2)
    ls = LinearSolver.QR

    nonlocal_tol = params.tol
    nonlocal_kappa = target_kappa

    def _step(inputs):
        qp, st, converged, pdip_iter = inputs
        Q, q, G, h, penalty = qp
        x, t, s1, s2, z1, z2 = st

        r1 = Q @ x + q + G.T @ z2
        r2 = z1 + z2 - penalty * jnp.ones(len(h))
        r3 = s1 * z1 - nonlocal_kappa
        r4 = s2 * z2 - nonlocal_kappa
        r5 = t + s1
        r6 = G @ x + t + s2 - h

        kkt_res = jnp.concatenate((r1, r2, r3, r4, r5, r6))
        converged = jnp.where(
            jnp.linalg.norm(kkt_res, ord=jnp.inf) < nonlocal_tol, 1, 0
        )

        dx, dt, ds1, ds2, dz1, dz2, _ = solve_elastic_kkt_affine(
            s1, z1, s2, z2, Q, G, -r1, -r2, -r3, -r4, -r5, -r6, ls
        )

        s = jnp.concatenate((s1, s2))
        z = jnp.concatenate((z1, z2))
        ds = jnp.concatenate((ds1, ds2))
        dz = jnp.concatenate((dz1, dz2))

        alpha = 0.99 * jnp.min(
            jnp.array(
                [
                    1.0,
                    0.99 * ort_linesearch(s, ds),
                    0.99 * ort_linesearch(z, dz),
                ]
            )
        )

        take = converged == 0
        new_state = ElasticQPState(
            jnp.where(take, x + alpha * dx, x),
            jnp.where(take, t + alpha * dt, t),
            jnp.where(take, s1 + alpha * ds1, s1),
            jnp.where(take, s2 + alpha * ds2, s2),
            jnp.where(take, z1 + alpha * dz1, z1),
            jnp.where(take, z2 + alpha * dz2, z2),
        )

        if DEBUG_FLAG:
            nr1 = jnp.linalg.norm(r1, ord=jnp.inf)
            nr2 = jnp.linalg.norm(r2, ord=jnp.inf)
            nr3 = jnp.linalg.norm(r3, ord=jnp.inf)
            nr4 = jnp.linalg.norm(r4, ord=jnp.inf)
            nr5 = jnp.linalg.norm(r5, ord=jnp.inf)
            nr6 = jnp.linalg.norm(r6, ord=jnp.inf)
            print(
                f"{pdip_iter:3d}   {nr1:9.2e}   {nr2:9.2e}"
                f"  {nr3:9.2e}  {nr4:9.2e}   {nr5:9.2e}"
                f"  {nr6:9.2e}   {alpha:6.4f}"
            )

        return (qp, new_state, converged, pdip_iter + 1)

    def _cond(inputs):
        _, _, converged, pdip_iter = inputs
        return jnp.logical_and(pdip_iter < params.max_iter, converged == 0)

    init = (qp, state, 0, 0)

    if DEBUG_FLAG:
        print(
            "iter      r1          r2         r3         r4        r5        r6"
            "        alpha"
        )
        print("-" * 80)
        val = init
        while _cond(val):
            val = _step(val)
        outputs = val
    else:
        outputs = jax.lax.while_loop(_cond, _step, init)

    _, final_state, converged, pdip_iter = outputs
    x_rlx, t_rlx, s1_rlx, s2_rlx, z1_rlx, z2_rlx = final_state

    return x_rlx, t_rlx, s1_rlx, s2_rlx, z1_rlx, z2_rlx, converged, pdip_iter


def optnet_derivatives_elastic(dz, dlam, z, lam):
    dl_dQ = 0.5 * (jnp.outer(dz, z) + jnp.outer(z, dz))
    dl_dG = jnp.diag(lam) @ (jnp.outer(dlam, z) + jnp.outer(lam, dz))  # TODO

    dl_dq = dz
    dl_dh = -lam * dlam

    return dl_dQ, dl_dq, dl_dG, dl_dh


def diff_qp_elastic(Q, q, G, h, x, t, s1, s2, lam1, lam2, dl_dz):
    nz = len(q)
    ns = len(h)

    zns = jnp.zeros(ns)
    dx, dt, ds1, ds2, dlam1, dlam2, _ = solve_elastic_kkt_affine(
        s1, lam1, s2, lam2, Q, G, -dl_dz, zns, zns, zns, zns, zns
    )

    dz = jnp.concatenate((dx, dt))
    z = jnp.concatenate((x, t))
    lam = jnp.concatenate((lam1, lam2))
    dlam_tilde = jnp.concatenate((dlam1, dlam2))

    dlam = dlam_tilde / lam

    dl_dQ, dl_dq, dl_dG, dl_dh = optnet_derivatives_elastic(dz, dlam, z, lam)

    dl_dQ = dl_dQ[:nz, :nz]
    dl_dq = dl_dq[:nz]
    dl_dG = dl_dG[ns:, :nz]
    dl_dh = dl_dh[ns:]

    return dl_dQ, dl_dq, dl_dG, dl_dh


@jax.custom_vjp
def solve_qp_elastic_primal(
    Q, q, G, h, penalty, solver_tol=1e-5, target_kappa=1e-3, max_iter=30
):
    x, t, s1, s2, z1, z2, converged, pdip_iter = solve_qp_elastic(
        Q, q, G, h, penalty, solver_tol=solver_tol, max_iter=max_iter
    )
    return x


def solve_qp_elastic_primal_fwd(
    Q, q, G, h, penalty, solver_tol=1e-5, target_kappa=1e-3, max_iter=30
):
    x, t, s1, s2, z1, z2, converged1, pdip_iter1 = solve_qp_elastic(
        Q, q, G, h, penalty, solver_tol=solver_tol, max_iter=max_iter
    )

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

    return x, (Q, q, G, h, penalty, xr, tr, s1r, s2r, z1r, z2r)


def solve_qp_elastic_primal_bwd(res, input_grad):
    Q, q, G, h, penalty, xr, tr, s1r, s2r, z1r, z2r = res

    dl_dQ, dl_dq, dl_dG, dl_dh = diff_qp_elastic(
        Q, q, G, h, xr, tr, s1r, s2r, z1r, z2r, input_grad
    )

    return (dl_dQ, dl_dq, dl_dG, dl_dh, None, None, None, None)


solve_qp_elastic_primal.defvjp(solve_qp_elastic_primal_fwd, solve_qp_elastic_primal_bwd)
