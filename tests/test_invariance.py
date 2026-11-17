"""Invariance, heterogeneity and degenerate-case coverage for the full round."""
import copy

import numpy as np
import pytest

from starter.local_model import local_probabilities
from starter.round import run_round, summary


def _client(rng, S, d, H, K, *, y_global=None):
    L = len(S)
    A = rng.uniform(-0.7, 0.7, (d, H))
    b = rng.uniform(-0.7, 0.7, H)
    C = rng.uniform(-0.7, 0.7, (H, L))
    c = rng.uniform(-0.7, 0.7, L)
    n = 5
    y = y_global if y_global is not None else [S[i % L] for i in range(n)]
    return {
        "labels": list(S),
        "model": {"A": A.tolist(), "b": b.tolist(),
                  "C": C.tolist(), "c": c.tolist()},
        "X": rng.uniform(-1, 1, (n, d)).tolist(),
        "y": y,
        "weights": [1, 2, 3, 4, 5],
        "public_X": rng.uniform(-1, 1, (4, d)).tolist(),
        "public_weights": [1, 2, 1, 2],
        "present": [True, True, False, True],
        "report_weight": 1.5,
    }


def _cfg(K, clients, **kw):
    base = {"classes": K, "reg": 0.2, "tau": 0.8, "ridge": 0.3, "eta": 0.05,
            "clients": clients}
    base.update(kw)
    return base


@pytest.mark.parametrize("K", [3, 5, 6])
def test_label_order_invariance(K):
    rng = np.random.default_rng(K)
    S = [4 % K, 0, (K - 1) % K] if K >= 3 else [0, 1, 2]
    S = list(dict.fromkeys(S))
    if len(S) < 2:
        S = [0, 1]
    cl = _client(rng, S, 2, 2, K)
    other = _client(np.random.default_rng(99), [K - 1, 1 % K], 3, 1, K)

    base = run_round(_cfg(K, [cl, other]))

    # Permute this client's label order, permuting A? No: permute *columns* of C,
    # entries of c, and reorder S to match; the model function is identical.
    perm = list(reversed(range(len(S))))
    cl2 = copy.deepcopy(cl)
    cl2["labels"] = [S[i] for i in perm]
    cl2["model"]["C"] = np.asarray(cl["model"]["C"])[:, perm].tolist()
    cl2["model"]["c"] = np.asarray(cl["model"]["c"])[perm].tolist()
    # y stays global; mapping is derived from the new S, so it is unchanged.
    permuted = run_round(_cfg(K, [cl2, other]))

    # Reports differ by column order but probabilities per global label match.
    for l_new, i_old in enumerate(perm):
        g = cl2["labels"][l_new]
        assert permuted["reports"][0]["labels"][l_new] == g
    # Global teacher and the untouched client's update are invariant.
    assert np.allclose(base["Q"], permuted["Q"], atol=1e-10)
    for k in ("A", "b", "C", "c"):
        assert np.allclose(base["clients"][1]["updated_model"][k],
                           permuted["clients"][1]["updated_model"][k],
                           atol=1e-10)


def test_two_and_four_clients():
    rng = np.random.default_rng(5)
    K = 4
    two = [_client(rng, [0, 1], 1, 1, K), _client(rng, [2, 3], 4, 3, K)]
    assert run_round(_cfg(K, two))["server"]["converged"]
    four = [
        _client(rng, [0, 1, 2], 2, 1, K), _client(rng, [1, 2, 3], 3, 2, K),
        _client(rng, [0, 3], 1, 2, K), _client(rng, [0, 1, 2, 3], 4, 3, K),
    ]
    rec = run_round(_cfg(K, four))
    assert rec["server"]["converged"]
    assert len(rec["clients"]) == 4


def test_disconnected_overlapping_and_missing_reports():
    rng = np.random.default_rng(6)
    K = 6
    c0 = _client(rng, [0, 1], 2, 2, K)
    c1 = _client(rng, [2, 3], 3, 1, K)
    c2 = _client(rng, [4, 5], 1, 3, K)
    # Disjoint cover, one client silent on every public row.
    c2["present"] = [False] * 4
    rec = run_round(_cfg(K, [c0, c1, c2]))
    assert rec["server"]["converged"]
    # The row is still driven by c0 and c1; no report claims class 4/5.
    assert np.all(np.isfinite(rec["Q"]))

    # All clients silent on one row -> uniform Q there.
    for c in (c0, c1, c2):
        c = c
    c0["present"] = [False, True, True, True]
    c1["present"] = [False, True, True, True]
    c2["present"] = [False, True, True, True]
    rec = run_round(_cfg(K, [c0, c1, c2]))
    assert np.allclose(rec["Q"][0], 1.0 / K)
    assert np.allclose(rec["server"]["z"][0], 0.0)


def test_single_class_per_row_not_required():
    """A report that sees only two labels still yields a full valid Q."""
    rng = np.random.default_rng(7)
    K = 6
    c0 = _client(rng, [0, 5], 2, 1, K)
    c1 = _client(rng, [0, 5], 2, 1, K)
    rec = run_round(_cfg(K, [c0, c1]))
    for r in range(4):
        assert rec["Q"][r].shape == (K,)
        assert abs(rec["Q"][r].sum() - 1) < 1e-12


def test_summary_is_json_serializable_and_finite():
    import json
    rng = np.random.default_rng(8)
    K = 5
    cfg = _cfg(K, [_client(rng, [0, 1, 2], 2, 2, K),
                   _client(rng, [2, 3, 4], 3, 1, K)])
    txt = json.dumps(summary(run_round(cfg)))
    assert "nan" not in txt.lower() and "infinity" not in txt.lower()


def test_eta_clip_extremes_stay_bounded():
    rng = np.random.default_rng(9)
    K = 4
    cfg = _cfg(K, [_client(rng, [0, 1], 2, 2, K),
                   _client(rng, [1, 2, 3], 3, 3, K)],
               eta=0.2, reg=1.0, tau=2.0, ridge=0.05)
    rec = run_round(cfg)
    for c in rec["clients"]:
        for v in c["updated_model"].values():
            assert np.all(np.abs(v) <= 2.0 + 1e-15)
            assert np.all(np.isfinite(v))


def test_public_probs_are_local_order():
    rng = np.random.default_rng(10)
    K = 5
    S = [4, 1, 3]
    cl = _client(rng, S, 2, 2, K)
    rec = run_round(_cfg(K, [cl, _client(rng, [0, 2], 1, 1, K)]))
    rep = rec["reports"][0]
    assert rep["labels"] == S
    assert np.asarray(rep["probabilities"]).shape == (4, 3)
    assert np.allclose(rec["clients"][0]["pre_public_probabilities"],
                       rep["probabilities"])
