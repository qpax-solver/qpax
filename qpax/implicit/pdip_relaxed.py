import jax
import jax.numpy as jnp

from qpax._verbose import print_footer, print_header

from .pdip import (
    factorize_implicit_kkt,
    ort_linesearch,
    retraction_map,
    solve_implicit_kkt_rhs,
)


def pdip_newton_step(inputs, verbose: bool = False):
    """One relaxed Newton step for relaxed QP"""
    (
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
        _Bp_prev,
        _Bn_prev,
        _c_prev,
        _lu_prev,
        _piv_prev,
    ) = inputs

    v = z - s
    kappa = jnp.maximum(jnp.dot(s, z) / len(s), 1e-14)

    rt = Q @ x + q + A.T @ y + G.T @ z
    rc = s * z - target_kappa
    ri = G @ x + s - h
    re = A @ x - b
    rz = z - retraction_map(v, kappa)
    rs = s - retraction_map(-v, kappa)

    kkt_res = jnp.concatenate((rt, rc, ri, re))

    converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)
    Bp_vec, Bn_vec, c_vec, L_J = factorize_implicit_kkt(Q, A, G, v, kappa)

    # TEST  ------------- STEP -------------
    rk = kappa - target_kappa

    dx, ds, dz, dy, dv, dk = solve_implicit_kkt_rhs(
        G, Bn_vec, Bp_vec, c_vec, L_J, rt, re, ri, rz, rs, rk
    )
    # TEST  ------------- STEP -------------

    alpha = 0.99 * jnp.min(jnp.array([ort_linesearch(s, ds), ort_linesearch(z, dz)]))

    if verbose:
        re_print = re if len(re) > 0 else jnp.zeros(1)
        print(
            f"{pdip_iter:3d}   "
            f"{kappa:9.2e}   "
            f"{jnp.linalg.norm(rt, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(rc, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(ri, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(re_print, ord=jnp.inf):9.2e}   "
            f"{alpha:6.4f}   {target_kappa:9.2e}"
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

    return (
        Q,
        q,
        A,
        b,
        G,
        h,
        x_new,
        s_new,
        z_new,
        y_new,
        solver_tol,
        converged,
        pdip_iter + 1,
        target_kappa,
        Bp_vec,
        Bn_vec,
        c_vec,
        L_J[0],
        L_J[1],
    )


def relax_qp(
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
    solver_tol=1e-5,
    target_kappa=1e-3,
    max_iter=30,
    sigma: float = 0.125,
    verbose: bool = False,
):
    """Relaxed solve that also returns the factorization from the last Newton step."""

    def relaxed_continuation_criteria(inputs):
        converged = inputs[11]
        pdip_iter = inputs[12]

        return jnp.logical_and(pdip_iter < max_iter, converged == 0)

    converged = 0
    pdip_iter = 0

    nz = G.shape[0]
    ny = A.shape[0]
    dim = G.shape[1] + nz + ny

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
        jnp.zeros(nz, dtype=Q.dtype),
        jnp.zeros(nz, dtype=Q.dtype),
        jnp.zeros(nz, dtype=Q.dtype),
        jnp.zeros((dim, dim), dtype=Q.dtype),
        jnp.zeros(dim, dtype=jnp.int32),
    )

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
            "iter      κ          r1          r2         r3         r4"
            "         alpha      target"
        )
        print(
            "-----------------------------------------------------------------------------------------------"
        )
        outputs = init_inputs
        while relaxed_continuation_criteria(outputs):
            outputs = pdip_newton_step(outputs, verbose=True)
    else:
        outputs = jax.lax.while_loop(
            relaxed_continuation_criteria, pdip_newton_step, init_inputs
        )

    x_rlx, s_rlx, z_rlx, y_rlx = outputs[6:10]
    converged = outputs[11]
    pdip_iter = outputs[12]
    Bp_vec, Bn_vec, c_vec = outputs[14:17]
    L_J = outputs[17], outputs[18]

    if verbose:
        print_footer(converged, 0.5 * x_rlx @ Q @ x_rlx + q @ x_rlx, pdip_iter)

    return x_rlx, s_rlx, z_rlx, y_rlx, Bp_vec, Bn_vec, c_vec, L_J, converged, pdip_iter
