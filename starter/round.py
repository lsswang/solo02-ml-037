"""One fully replayable heterogeneous-label collaborative-distillation round.

Fixed order (no re-aggregation of updated models, no inner training loop):

1. Each client runs its *round-initial* model on its ``public_X`` and emits a
   conditional report in its own label order ``S``.
2. The single server entry point :func:`starter.consensus.consensus` turns those
   reports into the current global teacher ``Q``.
3. Each client takes **one** gradient step of its *full* local objective
   (exact labels + renormalised teacher from that same ``Q``) and clips every
   parameter to ``[-2, 2]``:  ``theta_new = clip(theta - eta * grad, -2, 2)``.
4. The updated models and their public probabilities are reported so the whole
   round can be replayed independently.

The old ``Q`` is consumed by the clients (the gradient is computed with it), not
merely printed: :func:`intervention_contrast` shows that changing one client's
round-initial model changes ``Q`` and thereby another client's actual gradient
and update.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .consensus import consensus
from .errors import InputError
from .local_model import local_probabilities, objective
from .validation import (
    as_bool_mask,
    as_feature_matrix,
    as_int_vector,
    as_label_vector,
    as_range_scalar,
    as_weight_vector,
    prepare_model,
)

__all__ = [
    "build_reports",
    "run_round",
    "client_update",
    "intervention_contrast",
    "PARAM_KEYS",
]

PARAM_KEYS = ("A", "b", "C", "c")
MODEL_BOUND = 2.0


def _validate_client(spec: Mapping, idx: int, classes: int) -> dict:
    if not isinstance(spec, Mapping):
        raise InputError(f"clients[{idx}] must be a mapping")
    for key in ("labels", "model", "X", "y", "weights", "public_X",
                "public_weights"):
        if key not in spec:
            raise InputError(f"clients[{idx}] is missing {key!r}")
    A, b, C, c = prepare_model(spec["model"], f"clients[{idx}].model")
    for key, val in (("A", A), ("b", b), ("C", C), ("c", c)):
        if np.any(np.abs(val) > MODEL_BOUND):
            raise InputError(
                f"clients[{idx}].model.{key} must have all parameters in "
                f"[-{MODEL_BOUND}, {MODEL_BOUND}]"
            )
    labels = as_label_vector(f"clients[{idx}].labels", spec["labels"], classes,
                             min_length=2, max_length=classes)
    if C.shape[1] != labels.shape[0]:
        raise InputError(
            f"clients[{idx}].model has {C.shape[1]} output columns but "
            f"{labels.shape[0]} labels"
        )
    d = A.shape[0]
    X = as_feature_matrix(f"clients[{idx}].X", spec["X"], d,
                          min_rows=1, max_rows=30, cols_max=1.0)
    y = as_int_vector(f"clients[{idx}].y", spec["y"], length=X.shape[0],
                      min_value=0, max_value=classes - 1)
    if not np.all(np.isin(y, labels)):
        raise InputError(f"clients[{idx}].y must all belong to its labels S")
    weights = as_weight_vector(f"clients[{idx}].weights", spec["weights"],
                               length=X.shape[0], integer=True, lo=1, hi=5)
    public_X = as_feature_matrix(f"clients[{idx}].public_X",
                                 spec["public_X"], d, min_rows=1, max_rows=8,
                                 cols_max=1.0)
    R = public_X.shape[0]
    public_weights = as_weight_vector(f"clients[{idx}].public_weights",
                                      spec["public_weights"], length=R)
    report_weight = as_range_scalar(
        f"clients[{idx}].report_weight", spec.get("report_weight", 1.0),
        0.1, 5.0)
    present = as_bool_mask(f"clients[{idx}].present",
                           spec.get("present", [True] * R), R)
    return {
        "labels": labels,
        "model": {"A": A, "b": b, "C": C, "c": c},
        "X": X,
        "y": y,
        "weights": weights,
        "public_X": public_X,
        "public_weights": public_weights,
        "report_weight": report_weight,
        "present": present,
        "R": R,
    }


def _validate_round_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise InputError("config must be a mapping")
    for key in ("classes", "clients"):
        if key not in config:
            raise InputError(f"config is missing {key!r}")
    classes = config["classes"]
    # Reuse consensus' class validation cheaply.
    try:
        K = int(classes)
    except (TypeError, ValueError) as exc:
        raise InputError("classes must be an integer") from exc
    if float(classes) != K or not (3 <= K <= 6):
        raise InputError("classes must be an integer in [3, 6]")
    clients_raw = config["clients"]
    if not isinstance(clients_raw, Sequence) or isinstance(
            clients_raw, (str, bytes)):
        raise InputError("clients must be a sequence")
    if not (2 <= len(clients_raw) <= 4):
        raise InputError("need 2 to 4 clients")
    clients = [_validate_client(c, i, K) for i, c in enumerate(clients_raw)]
    row_counts = {c["R"] for c in clients}
    if len(row_counts) != 1:
        raise InputError("all clients must share the same number of public rows R")
    reg = as_range_scalar("reg", config.get("reg", 0.1), 0.05, 1.0)
    tau = as_range_scalar("tau", config.get("tau", 0.0), 0.0, 2.0)
    ridge = as_range_scalar("ridge", config.get("ridge", 0.05), 0.05, 1.0)
    # ``step_size`` is accepted as an alias for the supplied fixture; ``eta``
    # takes precedence when both are present.
    eta_raw = config["eta"] if "eta" in config else config.get("step_size", 0.1)
    eta = as_range_scalar("eta", eta_raw, 0.001, 0.2)
    max_iter = config.get("max_iter", 500)
    return {
        "classes": K,
        "clients": clients,
        "reg": reg,
        "tau": tau,
        "ridge": ridge,
        "eta": eta,
        "max_iter": max_iter,
        "R": clients[0]["R"],
    }


def build_reports(clients: Sequence[dict], round_models: Sequence[Mapping]) -> list[dict]:
    """Step 1: turn round-initial models into conditional reports in S order."""
    reports = []
    for cl, model in zip(clients, round_models):
        probs = local_probabilities(model, cl["public_X"])
        if probs.shape[1] != cl["labels"].shape[0]:
            raise InputError("model public-probability width does not match S")
        reports.append({
            "labels": cl["labels"].tolist(),
            "weight": cl["report_weight"],
            "present": cl["present"].tolist(),
            "probabilities": probs.tolist(),
        })
    return reports


def client_update(
    client: dict,
    model: Mapping,
    Q: Any,
    classes: int,
    tau: float,
    reg: float,
    eta: float,
) -> dict:
    """Step 3 for one client: loss/grad at ``model`` with ``Q`` then one clipped step."""
    A0, b0, C0, c0 = prepare_model(model)
    model_arr = {"A": A0.copy(), "b": b0.copy(), "C": C0.copy(), "c": c0.copy()}

    loss, grads = objective(
        model_arr,
        client["X"],
        client["y"],
        client["weights"],
        reg,
        Q=Q,
        S=client["labels"],
        tau=tau,
        public_X=client["public_X"],
        public_weights=client["public_weights"],
        classes=classes,
        labels_are_global=True,
    )

    updated = {}
    for key in PARAM_KEYS:
        theta = np.asarray(model_arr[key], dtype=float)
        new = np.clip(theta - eta * grads[key], -MODEL_BOUND, MODEL_BOUND)
        updated[key] = new

    pre_public = local_probabilities(model_arr, client["public_X"])
    post_public = local_probabilities(updated, client["public_X"])
    return {
        "loss_before": loss,
        "gradient": {k: grads[k].copy() for k in PARAM_KEYS},
        "updated_model": updated,
        "pre_public_probabilities": pre_public,
        "post_public_probabilities": post_public,
    }


def run_round(config: Mapping) -> dict:
    """Execute one complete round and return a fully reproducible record."""
    cfg = _validate_round_config(config)
    classes = cfg["classes"]
    clients = cfg["clients"]

    round_models = [c["model"] for c in clients]

    # 1. Client reports from round-initial models, labelled in S order.
    reports = build_reports(clients, round_models)

    # 2. Single server entry point -> current global teacher.
    server = consensus(reports, classes, ridge=cfg["ridge"],
                       max_iter=cfg["max_iter"])
    Q = server["Q"]

    # 3./4. One full local gradient step per client, consuming that Q.
    client_records = []
    for cl, model in zip(clients, round_models):
        rec = client_update(cl, model, Q, classes, cfg["tau"], cfg["reg"],
                            cfg["eta"])
        rec["client_report"] = None  # filled below
        client_records.append(rec)

    for rec, rep in zip(client_records, reports):
        rec["client_report"] = rep

    return {
        "classes": classes,
        "reg": cfg["reg"],
        "tau": cfg["tau"],
        "ridge": cfg["ridge"],
        "eta": cfg["eta"],
        "reports": reports,
        "server": {
            "z": server["z"],
            "Q": Q,
            "objective": server["objective"],
            "total_objective": server["total_objective"],
            "gradient": server["gradient"],
            "converged": server["converged"],
            "diagnostics": server["diagnostics"],
        },
        "Q": Q,
        "clients": client_records,
    }


def _model_to_list(model: Mapping) -> dict:
    return {k: np.asarray(model[k], dtype=float).tolist() for k in PARAM_KEYS}


def summary(record: Mapping) -> dict:
    """JSON-friendly view of a :func:`run_round` result (finite, no histories)."""
    diag = record["server"]["diagnostics"]
    out = {
        "classes": record["classes"],
        "reg": record["reg"],
        "tau": record["tau"],
        "ridge": record["ridge"],
        "eta": record["eta"],
        "reports": [
            {
                "labels": np.asarray(r["labels"]).tolist(),
                "weight": r["weight"],
                "present": list(r["present"]),
                "probabilities": np.asarray(r["probabilities"]).tolist(),
            }
            for r in record["reports"]
        ],
        "server": {
            "z": record["server"]["z"].tolist(),
            "Q": record["server"]["Q"].tolist(),
            "objective": record["server"]["objective"].tolist(),
            "total_objective": record["server"]["total_objective"],
            "gradient": record["server"]["gradient"].tolist(),
            "converged": record["server"]["converged"],
            "iterations": diag["iterations"].tolist(),
            "max_abs_gradient": diag["max_abs_gradient"].tolist(),
            "row_converged": diag["converged"].tolist(),
            "messages": list(diag["message"]),
        },
        "clients": [
            {
                "labels": np.asarray(rec["client_report"]["labels"]).tolist(),
                "loss_before": rec["loss_before"],
                "gradient": {k: rec["gradient"][k].tolist() for k in PARAM_KEYS},
                "updated_model": _model_to_list(rec["updated_model"]),
                "pre_public_probabilities":
                    rec["pre_public_probabilities"].tolist(),
                "post_public_probabilities":
                    rec["post_public_probabilities"].tolist(),
            }
            for rec in record["clients"]
        ],
    }
    return out


def intervention_contrast(
    config: Mapping,
    client_index: int,
    replacement_model: Mapping,
) -> dict:
    """Show that one client's round-initial model changes Q and another's update.

    Runs the baseline round, then reruns it with ``client_index``'s round-initial
    model replaced.  The *consuming* comparison uses the old vs new ``Q`` to
    compute the other clients' full local gradients, proving ``Q`` is actually
    consumed rather than merely printed.
    """
    cfg = _validate_round_config(config)
    clients = cfg["clients"]
    n = len(clients)
    if not (0 <= client_index < n):
        raise InputError(f"client_index must be in [0, {n - 1}]")
    replacement = prepare_model(replacement_model, "replacement_model")
    if any(np.any(np.abs(v) > MODEL_BOUND) for v in replacement):
        raise InputError("replacement model parameters must lie in [-2, 2]")
    # Validate the replacement shape matches the replaced client.
    target = clients[client_index]
    if (replacement[0].shape != target["model"]["A"].shape
            or replacement[1].shape != target["model"]["b"].shape
            or replacement[2].shape != target["model"]["C"].shape
            or replacement[3].shape != target["model"]["c"].shape):
        raise InputError("replacement model shape does not match client")

    base = run_round(config)

    clients2 = [dict(c) for c in clients]
    clients2[client_index] = dict(clients2[client_index])
    clients2[client_index]["model"] = {
        "A": replacement[0], "b": replacement[1],
        "C": replacement[2], "c": replacement[3],
    }
    cfg2 = dict(config)
    cfg2["clients"] = clients2
    alt = run_round(cfg2)

    # Identify downstream consumers (every client except the intervened one).
    Q_base = base["Q"]
    Q_alt = alt["Q"]
    consumers = []
    for j in range(n):
        if j == client_index:
            continue
        cl = clients[j]
        grad_oldQ = objective(
            cl["model"], cl["X"], cl["y"], cl["weights"], cfg["reg"],
            Q=Q_base, S=cl["labels"], tau=cfg["tau"],
            public_X=cl["public_X"], public_weights=cl["public_weights"],
            classes=cfg["classes"], labels_are_global=True,
        )
        grad_newQ = objective(
            cl["model"], cl["X"], cl["y"], cl["weights"], cfg["reg"],
            Q=Q_alt, S=cl["labels"], tau=cfg["tau"],
            public_X=cl["public_X"], public_weights=cl["public_weights"],
            classes=cfg["classes"], labels_are_global=True,
        )
        old = base["clients"][j]
        # Update direction under the two different teachers.
        upd_oldQ = {k: np.clip(cl["model"][k] - cfg["eta"] * grad_oldQ[1][k],
                               -MODEL_BOUND, MODEL_BOUND) for k in PARAM_KEYS}
        upd_newQ = {k: np.clip(cl["model"][k] - cfg["eta"] * grad_newQ[1][k],
                               -MODEL_BOUND, MODEL_BOUND) for k in PARAM_KEYS}
        grad_diff = {k: float(np.max(np.abs(grad_newQ[1][k] - grad_oldQ[1][k])))
                     for k in PARAM_KEYS}
        update_diff = {k: float(np.max(np.abs(upd_newQ[k] - upd_oldQ[k])))
                       for k in PARAM_KEYS}
        # The alternative-round consumer result uses the *new* Q (alt run).
        alt_rec = alt["clients"][j]
        consumed_changed = (
            alt_rec["loss_before"] != old["loss_before"]
            or any(np.max(np.abs(alt_rec["gradient"][k] - old["gradient"][k])) > 0
                   for k in PARAM_KEYS)
        )
        consumers.append({
            "client_index": j,
            "labels": cl["labels"].tolist(),
            "loss_before_oldQ": old["loss_before"],
            "loss_before_newQ": alt_rec["loss_before"],
            "gradient_consumption_max_abs_diff": grad_diff,
            "update_max_abs_diff_from_teacher_swap": update_diff,
            "update_max_abs_diff_between_rounds": {
                k: float(np.max(np.abs(
                    alt_rec["updated_model"][k] - old["updated_model"][k])))
                for k in PARAM_KEYS
            },
            "teacher_actually_consumed": bool(consumed_changed),
        })

    intervened = client_index
    rep_base = np.asarray(base["reports"][intervened]["probabilities"])
    rep_alt = np.asarray(alt["reports"][intervened]["probabilities"])
    return {
        "intervened_client": intervened,
        "report_max_abs_change": float(np.max(np.abs(rep_alt - rep_base))),
        "Q_max_abs_change": float(np.max(np.abs(Q_alt - Q_base))),
        "Q_changed": float(np.max(np.abs(Q_alt - Q_base))) > 0.0,
        "base": base,
        "alternative": alt,
        "consumers": consumers,
    }
