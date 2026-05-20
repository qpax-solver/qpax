"""Tests for the elastic QP backend dispatch."""

import jax.numpy as jnp
import numpy as np
import pytest

import qpax


def test_elastic_explicit_smoke():
    Q = jnp.eye(2)
    q = jnp.zeros(2)
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    h = jnp.array([1.0, 1.0])
    penalty = jnp.array(1.0)

    out = qpax.solve_qp_elastic(Q, q, G, h, penalty, backend="e")
    x = out[0]
    assert x.shape == (2,)
    assert jnp.all(jnp.isfinite(x))


def test_elastic_explicit_preserves_state_on_terminal_iteration():
    Q = jnp.eye(2, dtype=jnp.float32)
    q = jnp.zeros(2, dtype=jnp.float32)
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    h = jnp.array([1.0, 1.0], dtype=jnp.float32)
    penalty = jnp.array(1.0, dtype=jnp.float32)

    full = qpax.solve_qp_elastic(Q, q, G, h, penalty, backend="e")
    iters = int(full[-1])

    assert int(full[-2]) == 1
    assert iters > 1

    capped = qpax.solve_qp_elastic(
        Q, q, G, h, penalty, backend="e", max_iter=iters - 1
    )

    for full_value, capped_value in zip(full[:6], capped[:6], strict=True):
        np.testing.assert_array_equal(
            np.asarray(full_value), np.asarray(capped_value)
        )


def test_elastic_implicit_raises():
    Q = jnp.eye(2)
    q = jnp.zeros(2)
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    h = jnp.array([1.0, 1.0])
    penalty = jnp.array(1.0)

    with pytest.raises(NotImplementedError, match="implicit backend"):
        qpax.solve_qp_elastic(Q, q, G, h, penalty, backend="i")

    with pytest.raises(NotImplementedError, match="implicit backend"):
        qpax.solve_qp_elastic_primal(Q, q, G, h, penalty, backend="i")
