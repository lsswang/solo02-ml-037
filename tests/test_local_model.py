"""Local objective: renormalised teacher, exact gradients, tau=0 degeneracy."""
import numpy as np
import pytest

from starter.local_model import objective, forward, local_target
from starter.validation import InputError
from starter.verify import check_local_gradients


def _rand_model(rng, d, hidden, n_labels, scale=0.8):
    return {
        "A": rng.uniform(-scale, scale, (d, hidden)),
        "b": rng.uniform(-scale, scale, hidden),
        "C": rng.uniform(-scale, scale, (hidden, n_labels)),
        "c": rng.uniform(-scale, scale, n_labels),
    }


def _softmax_rows(Z):
    from scipy.special import logsumexp
    return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))


def test_forward_matches_manual():
    rng = np.random.default_rng(0)
    m = _rand_model(rng, 3, 2, 4)
    X = rng.uniform(-1, 1, (5, 3))
    H = np.tanh(X @ m["A"] + m["b"])
    Z = H @ m["C"] + m["c"]
    H2, Z2, P2 = forward(m, X)
    assert np.allclose(H, H2) and np.allclose(Z, Z2)
    assert np.allclose(P2, _softmax_rows(Z))
    assert np.allclose(P2.sum(axis=1), 1.0)


@pytest.mark.parametrize("d,hidden,L", [(1, 1, 2), (2, 3, 4), (4, 2, 5), (3, 1, 6)])
@pytest.mark.parametrize("tau", [0.0, 0.35, 2.0])
def test_local_gradients_finite_difference(d, hidden, L, tau):
    rng = np.random.default_rng(1)
    K = 6
    S = rng.choice(K, size=L, replace=False)
    model = _rand_model(rng, d, hidden, L)
    X = rng.uniform(-1, 1, (7, d))
    y_local = rng.integers(0, L, size=7)
    w = rng.integers(1, 6, size=7).astype(float)
    R = 4
    public_X = rng.uniform(-1, 1, (R, d))
    public_w = rng.uniform(0.5, 3.0, R)
    Q = rng.uniform(0.05, 1.0, (R, K))
    Q /= Q.sum(axis=1, keepdims=True)
    checks = check_local_gradients(
        model, X, y_local, w, 0.3, Q=Q, S=S, tau=tau,
        public_X=public_X, public_weights=public_w, classes=K,
    )
    for key, (abs_err, rel_err) in checks.items():
        assert abs_err < 2e-7, (key, abs_err)
        assert rel_err < 1e-6 or abs_err < 2e-7, (key, rel_err)


def test_tau_zero_equals_plain_supervised():
    rng = np.random.default_rng(2)
    model = _rand_model(rng, 2, 2, 3)
    X = rng.uniform(-1, 1, (6, 2))
    y = rng.integers(0, 3, 6)
    w = rng.integers(1, 6, 6).astype(float)
    # tau=0 must ignore a (valid) teacher entirely.
    Q = np.full((2, 5), 0.2)
    a = objective(model, X, y, w, 0.4)
    b = objective(model, X, y, w, 0.4, Q=Q, S=[4, 0, 2], tau=0.0,
                  public_X=rng.uniform(-1, 1, (2, 2)),
                  public_weights=[1, 2], classes=5)
    assert a[0] == pytest.approx(b[0], abs=1e-12)
    for k in a[1]:
        assert np.allclose(a[1][k], b[1][k], atol=1e-12)


def test_teacher_is_renormalised_not_truncated():
    # Global teacher concentrated on labels outside S.
    Q = np.array([[0.05, 0.05, 0.60, 0.30]])
    t = local_target(Q, [0, 1], classes=4)
    assert np.allclose(t, [[0.5, 0.5]])          # renormalised, not [0.05, 0.05]
    assert np.allclose(t.sum(axis=1), 1.0)


def test_teacher_constant_no_gradient_into_Q():
    rng = np.random.default_rng(3)
    model = _rand_model(rng, 2, 1, 3)
    X = rng.uniform(-1, 1, (4, 2))
    y = rng.integers(0, 3, 4)
    Q = rng.uniform(0.1, 1, (3, 5)); Q /= Q.sum(1, keepdims=True)
    l1, g1 = objective(model, X, y, np.ones(4), 0.1, Q=Q, S=[1, 3, 4],
                       tau=1.0, public_X=rng.uniform(-1, 1, (3, 2)),
                       public_weights=np.ones(3), classes=5)
    Q2 = Q.copy(); Q2[0, 0] *= 1.0  # same object semantics: Q unchanged
    assert np.array_equal(Q, Q2)


def test_global_labels_remap():
    rng = np.random.default_rng(4)
    model = _rand_model(rng, 2, 2, 3)
    X = rng.uniform(-1, 1, (5, 2))
    y_global = np.array([4, 2, 0, 4, 2])
    S = [4, 2, 0]
    y_local = np.array([0, 1, 2, 0, 1])
    a = objective(model, X, y_local, np.ones(5), 0.2)
    b = objective(model, X, y_global, np.ones(5), 0.2, S=S, classes=5,
                  labels_are_global=True)
    assert a[0] == pytest.approx(b[0])
    for k in a[1]:
        assert np.allclose(a[1][k], b[1][k])


def test_stable_with_large_logits():
    model = {
        "A": np.array([[50.0]]), "b": np.array([0.0]),
        "C": np.array([[1000.0, -1000.0]]), "c": np.array([0.0, 0.0]),
    }
    X = np.array([[1.0]])
    loss, grads = objective(model, X, [0], [1], 0.05, tau=0.0)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(v)) for v in grads.values())
    _, _, P = forward(model, X)
    assert np.all(np.isfinite(P))


def test_invalid_inputs():
    rng = np.random.default_rng(5)
    model = _rand_model(rng, 2, 2, 3)
    X = np.zeros((3, 2)); y = np.array([0, 1, 2]); w = np.ones(3)
    with pytest.raises(InputError):
        objective(model, X, np.array([0, 1, 3]), w, 0.1)        # label OOB
    with pytest.raises(InputError):
        objective(model, X, y, np.array([0, 1, 1]), 0.1)        # zero weight
    with pytest.raises(InputError):
        objective(model, X, y, np.array([1.5, 1, 1]), 0.1)      # non-integer
    with pytest.raises(InputError):
        objective(model, np.full((3, 2), np.nan), y, w, 0.1)    # non-finite
    with pytest.raises(InputError):
        objective({"A": np.zeros((2, 2)), "b": np.zeros(2),
                   "C": np.zeros((1, 2)), "c": np.zeros(2)}, X, y, w, 0.1)
    with pytest.raises(InputError):
        objective(model, X, y, w, 0.1, S=[0, 0, 1], tau=1.0,
                  Q=np.full((3, 3), 1 / 3), public_X=X,
                  public_weights=w, classes=3)                  # duplicate S
    with pytest.raises(InputError):
        objective(model, X, y, w, 0.1, Q=np.full((3, 3), 1 / 3), S=[0, 1, 2],
                  tau=1.0, public_X=X, public_weights=[1, 1, -1], classes=3)
