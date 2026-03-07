import jax
import jax.numpy as jnp

from qpax.pdip import factorize_kkt, solve_kkt_rhs, solve_qp
from qpax.pdip_relaxed import relax_qp


def optnet_derivatives(dz, dlam, dnu, z, lam, nu):
    dl_dQ = 0.5 * (jnp.outer(dz, z) + jnp.outer(z, dz))
    dl_dA = jnp.outer(dnu, z) + jnp.outer(nu, dz)
    dl_dG = jnp.diag(lam) @ (jnp.outer(dlam, z) + jnp.outer(lam, dz))  # TODO

    dl_dq = dz
    dl_db = -dnu
    dl_dh = -lam * dlam

    return dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh


def diff_qp(Q, q, A, b, G, h, z, s, lam, nu, dl_dz):
    ns = len(h)
    nnu = len(b)

    # solve using same KKT solving functions
    P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, lam)

    # stilde = G @ z - h
    dz, ds, dlam_tilde, dnu = solve_kkt_rhs(
        Q,
        G,
        A,
        s,
        lam,
        P_inv_vec,
        L_H,
        L_F,
        -dl_dz,
        jnp.zeros(ns),
        jnp.zeros(ns),
        jnp.zeros(nnu),
    )

    # recover real dlam from our modified (symmetrized) KKT system
    dlam = dlam_tilde / lam

    return optnet_derivatives(dz, dlam, dnu, z, lam, nu)


@jax.custom_vjp
def solve_qp_primal(Q, q, A, b, G, h, solver_tol=1e-5, target_kappa=1e-3, max_iter=30):
    # solve qp as normal and return primal solution (use any solver)
    x, s, z, y, converged, iters1 = solve_qp(Q, q, A, b, G, h, solver_tol=solver_tol, max_iter=max_iter)
    return x


"""
these two functions are only called when we diff solve_qp_x
"""


def solve_qp_primal_forward(Q, q, A, b, G, h, solver_tol=1e-5, target_kappa=1e-3, max_iter=30):
    # solve qp as normal and return primal solution (use any solver)
    x, s, z, y, converged1, iters1 = solve_qp(Q, q, A, b, G, h, solver_tol=solver_tol, max_iter=max_iter)

    # relax this solution by taking vanilla Newton steps on relaxed KKT
    xr, sr, zr, yr, converged2, iters2 = relax_qp(
        Q, q, A, b, G, h, x, s, z, y, solver_tol=solver_tol, target_kappa=target_kappa, max_iter=max_iter
    )

    # return real solution x, and save the relaxed variables for backward
    return x, (Q, q, A, b, G, h, xr, sr, zr, yr)


def solve_qp_primal_backward(res, input_grad):
    # unpack relaxed solution
    Q, q, A, b, G, h, xr, sr, zr, yr = res

    # return all the normal derivatives, then None's for kwargs
    return (*diff_qp(Q, q, A, b, G, h, xr, sr, zr, yr, input_grad), None, None, None)


solve_qp_primal.defvjp(solve_qp_primal_forward, solve_qp_primal_backward)
