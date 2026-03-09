import jax
import jax.numpy as jnp
import numpy as np

import qpax

jax.config.update("jax_enable_x64", True)


def load_problem_data():
    d = np.load("test/problem_data_pos.npz")
    Q = jnp.array(d["Q"])
    q = jnp.array(d["q"])
    G = jnp.array(d["G"])
    h = jnp.array(d["h"])
    penalty = d["penalty"]
    x_sol = jnp.array(d["x"])
    t_sol = jnp.array(d["t"])
    s1_sol = jnp.array(d["s1"])
    s2_sol = jnp.array(d["s2"])
    z1_sol = jnp.array(d["z1"])
    z2_sol = jnp.array(d["z2"])

    return Q, q, G, h, penalty, x_sol, t_sol, s1_sol, s2_sol, z1_sol, z2_sol


def load_initial_values():
    d = np.load("test/initial_values_pos.npz")
    x = jnp.array(d["x"])
    t = jnp.array(d["t"])
    s1 = jnp.array(d["s1"])
    s2 = jnp.array(d["s2"])
    z1 = jnp.array(d["z1"])
    z2 = jnp.array(d["z2"])

    return x, t, s1, s2, z1, z2


def create_elastic_problem(Q, q, G, h, penalty):
    nx = len(q)
    ns = len(h)
    zxs = jnp.zeros((nx, ns))
    # zxx = 0 * jnp.eye(nx)
    zss = 0 * jnp.eye(ns)
    zsx = jnp.zeros((ns, nx))
    Q2 = jnp.block([[Q, zxs], [zsx, zss]])
    q2 = jnp.concatenate((q, penalty * jnp.ones(ns)))

    G2 = jnp.block([[zsx, -jnp.eye(ns)], [G, -jnp.eye(ns)]])
    h2 = jnp.block([jnp.zeros(ns), h])

    A2 = jnp.zeros((0, nx + ns))
    b2 = jnp.zeros(0)

    return Q2, q2, A2, b2, G2, h2


def test_with_qpax():
    Q, q, G, h, penalty, x_sol, t_sol, s1_sol, s2_sol, z1_sol, z2_sol = (
        load_problem_data()
    )

    x_i, t_i, s1_i, s2_i, z1_i, z2_i = load_initial_values()

    # create elastic problem and solve it with normal qpax
    Q2, q2, A2, b2, G2, h2 = create_elastic_problem(Q, q, G, h, penalty)
    x_big, s_big, z_big, y, converged_big, pdip_iter_big = jax.jit(qpax.solve_qp)(
        Q2, q2, A2, b2, G2, h2, solver_tol=1e-10
    )

    # check if the solution is correct
    assert converged_big == 1
    assert jnp.linalg.norm(x_big - jnp.concatenate((x_sol, t_sol))) < 1e-6
    assert jnp.linalg.norm(s_big - jnp.concatenate((s1_sol, s2_sol))) < 1e-6
    assert jnp.linalg.norm(z_big - jnp.concatenate((z1_sol, z2_sol))) < 1e-6

    # remove the variables we don't need
    del x_big, s_big, z_big, y, converged_big, pdip_iter_big

    # now let's check the initialization
    x_big_i, s_big_i, z_big_i, y_big_i = qpax.pdip.initialize(qpax.pdip.QPData(Q2, q2, A2, b2, G2, h2))
    assert jnp.linalg.norm(x_big_i - jnp.concatenate((x_i, t_i))) < 1e-6
    assert jnp.linalg.norm(s_big_i - jnp.concatenate((s1_i, s2_i))) < 1e-6
    assert jnp.linalg.norm(z_big_i - jnp.concatenate((z1_i, z2_i))) < 1e-6

    del x_big_i, s_big_i, z_big_i, y_big_i

    # qpax elastic tests

    # check the qpax elastic initialization
    _x_i, _t_i, _s1_i, _s2_i, _z1_i, _z2_i = qpax.elastic_qp.initialize_elastic(
        qpax.elastic_qp.ElasticQPData(Q, q, G, h, penalty)
    )
    assert jnp.linalg.norm(x_i - _x_i) < 1e-6
    assert jnp.linalg.norm(t_i - _t_i) < 1e-6
    assert jnp.linalg.norm(s1_i - _s1_i) < 1e-6
    assert jnp.linalg.norm(s2_i - _s2_i) < 1e-6
    assert jnp.linalg.norm(z1_i - _z1_i) < 1e-6
    assert jnp.linalg.norm(z2_i - _z2_i) < 1e-6

    # check qpax solutions
    _x, _t, _s1, _s2, _z1, _z2, _converged, _pdip_iter = jax.jit(
        qpax.elastic_qp.solve_qp_elastic
    )(Q, q, G, h, penalty, solver_tol=1e-10)
    assert _converged == 1
    assert jnp.linalg.norm(x_sol - _x) < 1e-6
    assert jnp.linalg.norm(t_sol - _t) < 1e-6
    assert jnp.linalg.norm(s1_sol - _s1) < 1e-6
    assert jnp.linalg.norm(s2_sol - _s2) < 1e-6
    assert jnp.linalg.norm(z1_sol - _z1) < 1e-6
    assert jnp.linalg.norm(z2_sol - _z2) < 1e-6

    # check primal version
    _x_primal = jax.jit(qpax.solve_qp_elastic_primal)(
        Q, q, G, h, penalty, solver_tol=1e-10
    )
    assert jnp.linalg.norm(x_sol - _x_primal) < 1e-6

    # check the gradients
    x_des = jnp.array(np.random.randn(len(q)))

    def testf_elastic(_Q, _q, _G, _h):
        _x = qpax.solve_qp_elastic_primal(
            _Q, _q, _G, _h, penalty, solver_tol=1e-6, target_kappa=1e-3
        )
        return jnp.sum((_x - x_des) ** 2)

    def testf_og(_Q, _q, _G, _h):
        _Q2, _q2, _A2, _b2, _G2, _h2 = create_elastic_problem(_Q, _q, _G, _h, penalty)

        _x_qpax = qpax.solve_qp_primal(
            _Q2, _q2, _A2, _b2, _G2, _h2, solver_tol=1e-6, target_kappa=1e-3
        )
        return jnp.sum((_x_qpax[: len(x_des)] - x_des) ** 2)

    grads1 = jax.jit(jax.grad(testf_elastic, (0, 1, 2, 3)))(Q, q, G, h)
    grads2 = jax.jit(jax.grad(testf_og, (0, 1, 2, 3)))(Q, q, G, h)

    print("norm of the gradients: ")
    print(jnp.linalg.norm(grads1[0]))
    print(jnp.linalg.norm(grads1[1]))
    print(jnp.linalg.norm(grads1[2]))
    print(jnp.linalg.norm(grads1[3]))

    print("norm of the errors:")
    print(jnp.linalg.norm(grads1[0] - grads2[0]))
    print(jnp.linalg.norm(grads1[1] - grads2[1]))
    print(jnp.linalg.norm(grads1[2] - grads2[2]))
    print(jnp.linalg.norm(grads1[3] - grads2[3]))

    assert jnp.linalg.norm(grads1[0] - grads2[0]) < 1e-3
    assert jnp.linalg.norm(grads1[1] - grads2[1]) < 1e-3
    assert jnp.linalg.norm(grads1[2] - grads2[2]) < 1e-3
    assert jnp.linalg.norm(grads1[3] - grads2[3]) < 1e-3


test_with_qpax()
