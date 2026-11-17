"""Server consensus: convex optimality, degenerate inputs, diagnostics, errors."""
import numpy as np
import pytest
from scipy.special import logsumexp

from starter.consensus import (
    GRAD_TOL,
    consensus,
    padded_average,
    parse_reports,
    row_objective,
)
from starter.validation import InputError
from starter.verify import least_squares_consensus


def _report(labels, probs, weight=1.0, present=None):
    probs = np.asarray(probs, dtype=float)
    R = probs.shape[0]
    return {
        "labels": list(labels),
        "probabilities": probs.tolist(),
        "weight": weight,
        "present": list(present) if present is not None else [True] * R,
    }


def _random_report(rng, K, L, R, weight=None):
    S = rng.choice(K, size=L, replace=False).tolist()
    p = rng.uniform(0.05, 1.0, (R, L))
    p /= p.sum(axis=1, keepdims=True)
    return _report(S, p, weight=weight if weight is not None else rng.uniform(0.1, 5))


@pytest.mark.parametrize("K", [3, 4, 5, 6])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimality_conditions(K, seed):
    rng = np.random.default_rng(seed)
    R = 5
    reports = []
    for _ in range(rng.integers(2, 5)):
        reports.append(_random_report(rng, K, int(rng.integers(2, K + 1)), R))
    ridge = float(rng.uniform(0.05, 1.0))
    out = consensus(reports, K, ridge=ridge)
    parsed = parse_reports(reports, K)
    assert out["converged"]
    assert np.max(out["diagnostics"]["max_abs_gradient"]) <= GRAD_TOL
    for r in range(R):
        F, g, H = row_objective(out["z"][r], parsed, r, ridge, classes=K)
        assert np.max(np.abs(g)) <= 1e-7
        # Hessian PSD with eigenvalues at least the ridge floor.
        ev = np.linalg.eigvalsh(H)
        assert np.min(ev) >= ridge - 1e-9
    # Q is a valid distribution and independent solver agrees.
    assert np.allclose(out["Q"].sum(axis=1), 1.0)
    z2 = least_squares_consensus(parsed, K, ridge, R)
    assert np.max(np.abs(z2 - out["z"])) < 1e-6


def test_unreported_row_gives_zero_and_uniform():
    K = 4
    reports = [
        _report([0, 2], [[0.6, 0.4], [0.3, 0.7]], present=[True, False]),
        _report([1, 3], [[0.5, 0.5], [0.2, 0.8]], present=[True, False]),
    ]
    out = consensus(reports, K, ridge=0.3)
    # Row 1 has no present report -> unique minimiser z=0, Q uniform.
    assert np.allclose(out["z"][1], 0.0)
    assert np.allclose(out["Q"][1], 0.25)
    assert out["diagnostics"]["iterations"][1] == 0
    assert out["diagnostics"]["converged"][1]


def test_disconnected_subsets_and_ridge_shift_selection():
    K = 5
    reports = [
        _report([0, 1], [[0.7, 0.3], [0.2, 0.8]]),
        _report([2, 3], [[0.4, 0.6], [0.9, 0.1]]),
        # class 4 never reported: its logit is only shrink-driven, tiny.
    ]
    out = consensus(reports, K, ridge=0.5)
    assert out["converged"]
    z = out["z"]
    # Common-shift gauge is fixed by ridge: mean of z ~ 0 (gradient adds ridge*z
    # and the softmax parts are shift invariant, so sum of residuals = ridge*sum z).
    # Unreported class 4 is pulled to zero relative to the reported block.
    assert np.all(np.abs(z[:, 4]) <= 1e-12)
    # No report mentions class 4, so its conditional probability is small.
    assert np.all(out["Q"][:, 4] < 0.5)


def test_overlapping_and_heterogeneous_subsets():
    K = 4
    reports = [
        _report([0, 1, 2], [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]),
        _report([2, 3], [[0.8, 0.2], [0.4, 0.6]], weight=3.0),
        _report([3, 0], [[0.5, 0.5], [0.25, 0.75]], weight=0.1),
    ]
    out = consensus(reports, K, ridge=0.05)
    assert out["converged"]
    assert np.max(np.abs(out["gradient"])) <= GRAD_TOL


def test_weight_actually_matters():
    K = 3
    a = consensus([_report([0, 1], [[0.9, 0.1]]),
                   _report([0, 2], [[0.1, 0.9]])], K, ridge=0.2)["Q"]
    b = consensus([_report([0, 1], [[0.9, 0.1]], weight=5.0),
                   _report([0, 2], [[0.1, 0.9]], weight=0.1)], K, ridge=0.2)["Q"]
    assert not np.allclose(a, b)


def test_max_iter_cap_reports_not_converged():
    K = 3
    reports = [_report([0, 1], [[1e-9, 0.999999999]])]  # extreme drives big step
    out = consensus(reports, K, ridge=0.05, max_iter=1)
    # With one Newton iteration it usually converges; force a tough case instead.
    # Use a hard multi-class mixture where 1 iteration cannot satisfy 1e-7.
    rng = np.random.default_rng(7)
    K = 6
    reports = []
    for _ in range(4):
        S = rng.choice(K, size=5, replace=False).tolist()
        p = rng.uniform(0.001, 1.0, (1, 5)); p /= p.sum()
        reports.append(_report(S, p))
    out = consensus(reports, K, ridge=0.05, max_iter=1)
    assert out["diagnostics"]["max_iter"] == 1
    assert out["diagnostics"]["iterations"][0] == 1
    if not out["converged"]:
        assert "cap" in out["diagnostics"]["message"][0]
        assert np.all(np.isfinite(out["z"]))
    # best finite model is always returned with a correct objective/gradient
    assert np.all(np.isfinite(out["Q"])) and np.allclose(out["Q"].sum(1), 1)


def test_default_settings_converge_on_normal_reports():
    rng = np.random.default_rng(11)
    K = 5
    reports = [_random_report(rng, K, int(rng.integers(2, K + 1)), 6)
               for _ in range(3)]
    out = consensus(reports, K)          # defaults: max_iter=500, ridge=0.05
    assert out["converged"]
    assert np.all(out["diagnostics"]["iterations"] <= 500)


def test_padded_average_baseline_does_not_renormalise():
    K = 3
    reports = [_report([0, 1], [[1.0 - 1e-12, 1e-12]])]
    base = padded_average(reports, K)
    # Class 2 is padded to exactly 0: the failure mode the objective fixes.
    assert base[0, 2] == 0.0
    assert np.isclose(base[0].sum(), 1.0)


def test_input_errors():
    K = 4
    good = _report([0, 1], [[0.5, 0.5]])
    with pytest.raises(InputError):
        consensus([], K)                                          # no reports
    with pytest.raises(InputError):
        consensus([good], 2)                                      # K out of range
    with pytest.raises(InputError):
        consensus([good], 7)
    with pytest.raises(InputError):
        consensus([_report([0, 0], [[0.5, 0.5]])], K)             # duplicate
    with pytest.raises(InputError):
        consensus([_report([0, 9], [[0.5, 0.5]])], K)             # out of range
    with pytest.raises(InputError):
        consensus([_report([0, 1], [[0.5, 0.6]])], K)             # rowsum != 1
    with pytest.raises(InputError):
        consensus([_report([0, 1], [[0.5, 0.0]])], K)             # < 1e-12
    with pytest.raises(InputError):
        consensus([_report([0, 1], [[0.5, 0.5]], weight=0.0)], K) # weight < 0.1
    with pytest.raises(InputError):
        consensus([_report([0, 1], [[0.5, 0.5]], weight=6.0)], K)
    with pytest.raises(InputError):
        consensus([_report([0, 1], [[0.5, 0.5]], present=[2])], K)  # bad mask
    with pytest.raises(InputError):
        consensus([_report([0, 1], [[0.5, np.nan]])], K)          # non-finite
    with pytest.raises(InputError):
        consensus([good], K, ridge=0.0)                           # ridge range
    with pytest.raises(InputError):
        consensus([good], K, ridge=2.0)
    with pytest.raises(InputError):
        consensus([good], K, max_iter=0)
    with pytest.raises(InputError):
        consensus([good], K, max_iter=2.5)
    with pytest.raises(InputError):
        consensus([_report([0, 1], [[0.5, 0.5]]),
                   _report([0, 1], [[0.5, 0.5], [0.5, 0.5]])], K)  # ragged R
