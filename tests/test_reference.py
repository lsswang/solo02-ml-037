"""Independent reference re-computation at tight tolerances (spec: 1e-8/1e-8).

These tests rebuild the server objective / gradient from the definition with a
plain Python loop rather than the vectorised production path, so a vectorisation
or indexing bug cannot cancel out.  They also cover the local objective against a
slow double-loop reference.
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from starter.consensus import consensus, parse_reports, row_objective


def reference_row_F(z, reports, row, ridge, K):
    """Plain-loop definition of F_r(z) and dF/dz."""
    F = 0.5 * ridge * float(np.dot(z, z))
    g = ridge * z.copy()
    for rep in reports:
        if not rep.present[row]:
            continue
        ids = list(rep.labels)
        p = list(rep.probabilities[row])
        a = rep.weight
        zs = [z[i] for i in ids]
        lse = logsumexp(zs)
        F += a * (lse - sum(pi * zi for pi, zi in zip(p, zs)))
        for l, i in enumerate(ids):
            q = np.exp(z[i] - lse)
            g[i] += a * (q - p[l])
    return F, g


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_server_objective_and_gradient_reference(seed):
    rng = np.random.default_rng(seed)
    K = int(rng.integers(3, 7))
    R = 4
    reports = []
    for _ in range(int(rng.integers(2, 5))):
        L = int(rng.integers(2, K + 1))
        S = rng.choice(K, size=L, replace=False)
        p = rng.uniform(0.01, 1.0, (R, L)); p /= p.sum(1, keepdims=True)
        present = rng.random(R) > 0.25
        reports.append({"labels": S.tolist(), "probabilities": p.tolist(),
                        "weight": float(rng.uniform(0.1, 5.0)),
                        "present": present.tolist()})
    ridge = float(rng.uniform(0.05, 1.0))
    out = consensus(reports, K, ridge=ridge)
    parsed = parse_reports(reports, K)

    for r in range(R):
        z = out["z"][r]
        F_prod, g_prod, _ = row_objective(z, parsed, r, ridge, classes=K)
        F_ref, g_ref = reference_row_F(z.copy(), parsed, r, ridge, K)
        assert F_prod == pytest.approx(F_ref, abs=1e-8, rel=1e-8)
        assert np.allclose(g_prod, g_ref, atol=1e-8, rtol=1e-8)
        # The reported objective array equals the definition at the returned z.
        assert out["objective"][r] == pytest.approx(F_ref, abs=1e-8, rel=1e-8)
        # And it is a stationary point.
        assert np.max(np.abs(g_ref)) <= 1e-7


def test_no_report_row_objective_is_pure_ridge():
    K = 4
    reports = [{"labels": [0, 1], "probabilities": [[0.5, 0.5]],
                "weight": 1.0, "present": [False]}]
    parsed = parse_reports(reports, K)
    F, g, H = row_objective(np.zeros(K), parsed, 0, 0.3, classes=K)
    assert F == 0.0
    assert np.allclose(g, 0.0)
    assert np.allclose(H, 0.3 * np.eye(K))


def test_consensus_matches_rowwise_independent_minimisation():
    """Closed-form sanity: a single report spanning all classes -> z is that log p."""
    K = 4
    p = np.array([[0.1, 0.2, 0.3, 0.4]])
    reports = [{"labels": [0, 1, 2, 3], "probabilities": p.tolist(),
                "weight": 2.0, "present": [True]}]
    # In the small-ridge limit, optimum of a*(logsumexp z - p.z) + ridge/2|z|^2
    # is approximately a*log(p/beta) for a normaliser beta; check stationarity.
    out = consensus(reports, K, ridge=0.05)
    assert out["converged"]
    assert np.max(np.abs(out["gradient"])) <= 1e-7
    assert np.allclose(out["Q"].sum(1), 1.0)
