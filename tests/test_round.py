"""Collaborative round: report construction, single-step update, replays,
interventions that change Q and are actually consumed by downstream clients."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from starter.errors import InputError
from starter.local_model import objective
from starter.round import (
    PARAM_KEYS,
    _model_to_list,
    build_reports,
    client_update,
    intervention_contrast,
    run_round,
    summary,
)

ROOT = Path(__file__).resolve().parent.parent


def _fixture():
    cfg = json.loads((ROOT / "fixtures" / "clients.json").read_text())
    return cfg


def _all_finite(obj):
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_finite(v) for v in obj)
    if isinstance(obj, (int, float, np.floating, np.integer)):
        return bool(np.isfinite(obj))
    return True


def test_round_structure_and_replay():
    cfg = _fixture()
    rec = run_round(cfg)
    assert rec["server"]["converged"]
    assert len(rec["reports"]) == len(cfg["clients"])
    assert len(rec["clients"]) == len(cfg["clients"])
    assert _all_finite(summary(rec))

    # Each report is exactly the round-initial model's public probabilities in S
    # order, and is a valid conditional distribution.
    for cl, rep, client_cfg in zip(rec["clients"], rec["reports"],
                                   cfg["clients"]):
        probs = np.asarray(rep["probabilities"])
        assert probs.shape == (len(client_cfg["public_X"]),
                               len(client_cfg["labels"]))
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.array_equal(np.asarray(rep["labels"]),
                              np.asarray(client_cfg["labels"]))

    # Independent replay: rebuild reports from the round-initial fixtures and
    # recompute Q and the gradients; numbers must match bit-for-bit (same code
    # path) and, more importantly, the update is exactly one clipped step.
    reports2 = build_reports(
        [{"labels": np.array(c["labels"]), "public_X": np.array(c["public_X"]),
          "report_weight": 1.0, "present": np.array([True] * 2, dtype=bool),
          "R": 2}
         for c in cfg["clients"]],
        [c["model"] for c in cfg["clients"]],
    )
    for a, b in zip(reports2, rec["reports"]):
        assert np.allclose(a["probabilities"], b["probabilities"])

    for cl_cfg, rec_c in zip(cfg["clients"], rec["clients"]):
        eta = cfg.get("eta", cfg.get("step_size"))
        for key in PARAM_KEYS:
            expected = np.clip(np.asarray(cl_cfg["model"][key])
                               - eta * rec_c["gradient"][key], -2, 2)
            assert np.allclose(rec_c["updated_model"][key], expected)
            assert np.all(np.abs(rec_c["updated_model"][key]) <= 2 + 1e-15)

    # Updated public probabilities correspond to the updated model (checked in
    # test_updated_public_probabilities_match_model).


def test_updated_public_probabilities_match_model():
    from starter.local_model import local_probabilities
    cfg = _fixture()
    rec = run_round(cfg)
    for cl_cfg, rec_c in zip(cfg["clients"], rec["clients"]):
        got = local_probabilities(rec_c["updated_model"], cl_cfg["public_X"])
        assert np.allclose(got, rec_c["post_public_probabilities"])
        before = local_probabilities(cl_cfg["model"], cl_cfg["public_X"])
        assert np.allclose(before, rec_c["pre_public_probabilities"])


def test_q_is_consumed_not_just_printed():
    """The gradient at the *old* Q differs from a gradient at a stale/base Q."""
    cfg = _fixture()
    rec = run_round(cfg)
    Q = rec["Q"]
    consumer = 1
    cl = cfg["clients"][consumer]
    _, g_used = objective(
        cl["model"], cl["X"], cl["y"], cl["weights"], cfg["reg"],
        Q=Q, S=cl["labels"], tau=cfg["tau"], public_X=cl["public_X"],
        public_weights=cl["public_weights"], classes=cfg["classes"],
        labels_are_global=True,
    )
    for key in PARAM_KEYS:
        assert np.allclose(g_used[key], rec["clients"][consumer]["gradient"][key])
    # With tau=0 the teacher cannot influence the gradient at all.
    _, g_notau = objective(
        cl["model"], cl["X"], cl["y"], cl["weights"], cfg["reg"],
        Q=Q, S=cl["labels"], tau=0.0, public_X=cl["public_X"],
        public_weights=cl["public_weights"], classes=cfg["classes"],
        labels_are_global=True,
    )
    assert any(not np.allclose(g_used[k], g_notau[k]) for k in PARAM_KEYS)


def test_intervention_changes_q_and_downstream_update():
    cfg = _fixture()
    base = cfg["clients"][0]["model"]["A"]
    repl = {k: np.asarray(v, dtype=float).copy()
            for k, v in cfg["clients"][0]["model"].items()}
    repl["A"] = np.clip(np.asarray(base) + np.array([[0.3, -0.2], [0.15, 0.25]]),
                        -2, 2)
    con = intervention_contrast(cfg, 0, repl)
    assert con["report_max_abs_change"] > 1e-6
    assert con["Q_changed"]
    assert con["Q_max_abs_change"] > 1e-6
    for consumer in con["consumers"]:
        assert consumer["teacher_actually_consumed"]
        assert consumer["loss_before_oldQ"] != consumer["loss_before_newQ"]
        assert max(consumer["update_max_abs_diff_from_teacher_swap"].values()) > 0
        assert max(consumer["gradient_consumption_max_abs_diff"].values()) > 0


def test_eta_and_clip_bounds():
    cfg = _fixture()
    cfg = copy.deepcopy(cfg)
    cfg["eta"] = 0.2
    rec = run_round(cfg)
    for rec_c in rec["clients"]:
        for k in PARAM_KEYS:
            assert np.all(np.abs(rec_c["updated_model"][k]) <= 2 + 1e-15)


def test_tau_zero_round_has_no_teacher_dependence():
    cfg = copy.deepcopy(_fixture())
    cfg["tau"] = 0.0
    rec = run_round(cfg)
    # Q is still produced (server step runs) but gradients equal supervised ones.
    for cl_cfg, rec_c in zip(cfg["clients"], rec["clients"]):
        local = [cl_cfg["labels"].index(y) for y in cl_cfg["y"]]
        _, g = objective(cl_cfg["model"], cl_cfg["X"], local,
                         cl_cfg["weights"], cfg["reg"])
        for k in PARAM_KEYS:
            assert np.allclose(g[k], rec_c["gradient"][k], atol=1e-14)


def test_invalid_round_inputs():
    cfg = _fixture()
    with pytest.raises(InputError):
        run_round({**cfg, "classes": 1})
    with pytest.raises(InputError):
        run_round({**cfg, "eta": 0.0})
    with pytest.raises(InputError):
        run_round({**cfg, "eta": 0.5})
    bad = copy.deepcopy(cfg)
    bad["clients"][0]["y"] = [0, 0, 2]          # label not in S
    with pytest.raises(InputError):
        run_round(bad)
    bad = copy.deepcopy(cfg)
    bad["clients"] = bad["clients"][:1]         # too few clients
    with pytest.raises(InputError):
        run_round(bad)
    bad = copy.deepcopy(cfg)
    bad["clients"][1]["public_X"] = [[0.0, 0.0]]  # wrong dim / row mismatch
    with pytest.raises(InputError):
        run_round(bad)
    bad = copy.deepcopy(cfg)
    bad["clients"][0]["weights"] = [1, 0, 1]
    with pytest.raises(InputError):
        run_round(bad)
    bad = copy.deepcopy(cfg)
    bad["clients"][0]["model"]["A"] = [[0.0]]  # hidden/C mismatch
    with pytest.raises(InputError):
        run_round(bad)
