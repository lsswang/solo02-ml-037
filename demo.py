"""Demonstration and reproducibility driver for one collaborative round.

Run:  python3 demo.py            (writes artifacts/report.json)
      python3 demo.py --pretty   (also prints the human summary)

Everything is deterministic and derived only from the two fixture files, so the
printed numbers can be recomputed independently.  See ``NUMERICAL_METHODS.md``
for the objective derivation, solver and cost analysis, and ``tests/`` for the
finite-difference and convex-optimality checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from starter.consensus import padded_average
from starter.round import intervention_contrast, run_round, summary
from starter.verify import check_local_gradients, least_squares_consensus

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def _load() -> tuple[dict, dict]:
    clients = json.loads((ROOT / "fixtures" / "clients.json").read_text())
    reports = json.loads((ROOT / "fixtures" / "reports.json").read_text())
    return clients, reports


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)
    clients_cfg, reports_cfg = _load()

    # ---- 1. A full collaborative round (reports -> Q -> one local step) ----
    record = run_round(clients_cfg)
    result = summary(record)

    # ---- 2. Independent gradient + convex-optimality checks ----
    Q = record["Q"]
    grad_checks = {}
    for j, cl in enumerate(clients_cfg["clients"]):
        checks = check_local_gradients(
            cl["model"], cl["X"], cl["y"], cl["weights"], clients_cfg["reg"],
            Q=Q, S=cl["labels"], tau=clients_cfg["tau"],
            public_X=cl["public_X"], public_weights=cl["public_weights"],
            classes=clients_cfg["classes"], labels_are_global=True,
        )
        grad_checks[f"client_{j}"] = {
            k: {"max_abs_err": float(v[0]), "max_rel_err": float(v[1])}
            for k, v in checks.items()
        }

    from starter.consensus import parse_reports
    parsed = parse_reports(record["reports"], clients_cfg["classes"])
    z_ref = least_squares_consensus(
        parsed, clients_cfg["classes"], clients_cfg["ridge"],
        record["server"]["z"].shape[0],
    )
    solver_check = {
        "z_max_abs_diff_vs_trust_ncg": float(
            np.max(np.abs(z_ref - record["server"]["z"]))
        ),
        "converged_all_rows": result["server"]["converged"],
        "max_abs_gradient": float(np.max(result["server"]["max_abs_gradient"])),
        "iterations": result["server"]["iterations"],
    }

    # ---- 3. Intervention: change one client's round-initial model ----
    base_client = clients_cfg["clients"][0]
    A = np.asarray(base_client["model"]["A"], dtype=float)
    perturbed = {
        k: np.asarray(v, dtype=float).copy()
        for k, v in base_client["model"].items()
    }
    perturbed["A"] = np.clip(A + np.array([[0.25, -0.15], [0.1, 0.2]]), -2, 2)
    contrast = intervention_contrast(clients_cfg, 0, perturbed)
    intervention = {
        "intervened_client": contrast["intervened_client"],
        "report_max_abs_change": contrast["report_max_abs_change"],
        "Q_max_abs_change": contrast["Q_max_abs_change"],
        "Q_changed": contrast["Q_changed"],
        "consumers": [
            {
                "client_index": c["client_index"],
                "labels": c["labels"],
                "loss_before_oldQ": c["loss_before_oldQ"],
                "loss_before_newQ": c["loss_before_newQ"],
                "gradient_consumption_max_abs_diff":
                    c["gradient_consumption_max_abs_diff"],
                "update_max_abs_diff_from_teacher_swap":
                    c["update_max_abs_diff_from_teacher_swap"],
                "teacher_actually_consumed": c["teacher_actually_consumed"],
            }
            for c in contrast["consumers"]
        ],
    }

    # ---- 4. Baseline (zero-padded average) for contrast ----
    baseline_Q = padded_average(record["reports"], clients_cfg["classes"])

    # ---- 5. Standalone server on the external fixtures (no client models) ----
    from starter.consensus import consensus
    standalone = consensus(
        reports_cfg["reports"], reports_cfg["classes"], ridge=reports_cfg["ridge"]
    )

    artifact = {
        "round": result,
        "inspections": {
            "local_gradient_finite_difference": grad_checks,
            "server_first_order_optimality": solver_check,
            "intervention": intervention,
            "zero_padded_baseline_Q": baseline_Q.tolist(),
        },
        "standalone_server": {
            "Q": standalone["Q"].tolist(),
            "z": standalone["z"].tolist(),
            "objective": standalone["objective"].tolist(),
            "gradient": standalone["gradient"].tolist(),
            "converged": standalone["converged"],
            "iterations": standalone["diagnostics"]["iterations"].tolist(),
            "max_abs_gradient":
                standalone["diagnostics"]["max_abs_gradient"].tolist(),
        },
    }
    (ARTIFACTS / "report.json").write_text(
        json.dumps(artifact, indent=2, default=_jsonable) + "\n"
    )

    if "--pretty" in sys.argv:
        _print_pretty(artifact)

    print(f"wrote {ARTIFACTS / 'report.json'}")
    return 0


def _print_pretty(artifact: dict) -> None:
    r = artifact["round"]
    insp = artifact["inspections"]
    print("=== collaborative round ===")
    print(f"classes={r['classes']} tau={r['tau']} reg={r['reg']} "
          f"ridge={r['ridge']} eta={r['eta']}")
    print("server converged:", r["server"]["converged"],
          "iterations:", r["server"]["iterations"],
          "max|grad|:", np.max(r["server"]["max_abs_gradient"]))
    print("Q (global teacher):")
    print(np.array2string(np.array(r["server"]["Q"]), precision=5))
    for i, cl in enumerate(r["clients"]):
        print(f"client {i} labels={cl['labels']} loss_before="
              f"{cl['loss_before']:.6f}")
        print("  gradient A:", np.array2string(
            np.array(cl["gradient"]["A"]), precision=5))
        print("  updated A:", np.array2string(
            np.array(cl["updated_model"]["A"]), precision=5))
    print("=== checks ===")
    print("max FD abs err:", max(
        v["max_abs_err"]
        for c in insp["local_gradient_finite_difference"].values()
        for v in c.values()))
    print("server vs independent solver |dz|:",
          insp["server_first_order_optimality"]["z_max_abs_diff_vs_trust_ncg"])
    inv = insp["intervention"]
    print(f"intervention client {inv['intervened_client']}: "
          f"|dQ|={inv['Q_max_abs_change']:.6f}")
    for c in inv["consumers"]:
        print(f"  consumer {c['client_index']}: grad consumed differs="
              f"{c['teacher_actually_consumed']}, "
              f"max|dgrad|={max(c['gradient_consumption_max_abs_diff'].values()):.3e}")


if __name__ == "__main__":
    raise SystemExit(main())
