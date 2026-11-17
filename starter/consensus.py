"""Server-side consensus: strictly convex global-logit aggregation.

Each report ``i`` supplies a label subset ``S_i``, a row-stochastic *conditional*
distribution ``p[i,r,:]`` over ``S_i`` (in ``S_i`` order), a scalar weight
``a_i`` and a per-public-row ``present`` mask.

For every public row ``r`` independently we minimise

    F_r(z) = sum_{i : present[i,r]} a_i * ( logsumexp(z[S_i])
                                             - sum_l p[i,r,l] * z[S_i[l]] )
             + ridge/2 * ||z||^2 ,      z in R^K .

The ``logsumexp`` term is the exact normaliser of the conditional report; it is
what makes the objective a (weighted) sum of KL divergences.  ``ridge > 0``
makes ``F_r`` strictly convex and pins down the common additive shift of ``z``.
Disconnected label subsets and rows with no reports are legal: with ridge the
minimiser exists and is unique (an unreported row has the unique minimiser
``z = 0``, i.e. a uniform ``Q``).

The solver is a damped Newton method with an analytic gradient/Hessian and
backtracking line search.  ``K <= 6``, so each step solves a tiny linear system.
Convergence is only claimed when ``max|grad F_r| <= 1e-7``; otherwise the best
finite iterate is returned with ``converged=False`` and diagnostics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
from scipy.special import logsumexp

from .validation import (
    InputError,
    as_bool_mask,
    as_float_array,
    as_label_vector,
    as_range_scalar,
    validate_report_probabilities,
)

__all__ = [
    "ParsedReport",
    "parse_reports",
    "consensus",
    "aggregate",
    "padded_average",
    "row_objective",
]

GRAD_TOL = 1e-7


class ParsedReport:
    """Validated immutable view of one client report."""

    __slots__ = ("labels", "probabilities", "weight", "present")

    def __init__(self, labels, probabilities, weight, present):
        self.labels = labels                # int64 (L,)
        self.probabilities = probabilities  # float64 (R, L), rows sum to 1
        self.weight = float(weight)
        self.present = present              # bool (R,)

    @property
    def n_rows(self) -> int:
        return int(self.probabilities.shape[0])


def parse_reports(
    reports: Sequence[Mapping], classes: Any, n_rows: Optional[int] = None
) -> list[ParsedReport]:
    """Validate a list of reports against a global class count ``classes``.

    ``classes`` must be an integer in ``[3, 6]``.
    """
    if isinstance(classes, (bool, np.bool_)):
        raise InputError("classes must be an integer")
    try:
        K = int(classes)
    except (TypeError, ValueError) as exc:
        raise InputError("classes must be an integer") from exc
    if K != float(classes):
        raise InputError("classes must be an integer")
    if not (3 <= K <= 6):
        raise InputError(f"classes must be in [3, 6], got {K}")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        raise InputError("reports must be a sequence")
    if len(reports) == 0:
        raise InputError("at least one report is required")

    parsed: list[ParsedReport] = []
    rows = n_rows
    for idx, rep in enumerate(reports):
        if not isinstance(rep, Mapping):
            raise InputError(f"reports[{idx}] must be a mapping")
        for key in ("labels", "probabilities", "weight", "present"):
            if key not in rep:
                raise InputError(f"reports[{idx}] is missing {key!r}")
        labels = as_label_vector(f"reports[{idx}].labels", rep["labels"], K,
                                 min_length=2, max_length=K)
        prob = np.asarray(rep["probabilities"])
        if prob.ndim != 2:
            raise InputError(f"reports[{idx}].probabilities must be 2-D")
        if prob.shape[1] != labels.shape[0]:
            raise InputError(
                f"reports[{idx}].probabilities must have {labels.shape[0]} "
                f"columns (one per label), got {prob.shape[1]}"
            )
        if rows is None:
            rows = prob.shape[0]
        if rows < 1:
            raise InputError("reports must have at least one public row")
        if rows > 8:
            raise InputError("reports may have at most 8 public rows")
        norm = validate_report_probabilities(
            prob, f"reports[{idx}].probabilities", n_rows=rows
        )
        weight = as_range_scalar(f"reports[{idx}].weight", rep["weight"], 0.1, 5.0)
        present = as_bool_mask(f"reports[{idx}].present", rep["present"], rows)
        parsed.append(ParsedReport(labels, norm, weight, present))
    return parsed


def _row_terms(z: np.ndarray, reports: list[ParsedReport], row: int,
               ridge: float):
    """Return ``(F, grad, hess)`` for one public row at ``z`` (shape ``(K,)``)."""
    K = z.shape[0]
    grad = ridge * z.copy()
    hess = ridge * np.eye(K)
    F = 0.5 * ridge * float(z @ z)
    for rep in reports:
        if not rep.present[row]:
            continue
        ids = rep.labels
        p = rep.probabilities[row]
        a = rep.weight
        zs = z[ids]
        lse = logsumexp(zs)
        # Conditional distribution of this report under z restricted to S_i.
        q = np.exp(zs - lse)
        F += a * (lse - float(p @ zs))
        diff = a * (q - p)                     # d/dz on the subset
        grad[ids] += diff
        # Hessian on the subset: a * (diag(q) - q q^T).
        hs = a * (np.diag(q) - np.outer(q, q))
        hess[np.ix_(ids, ids)] += hs
    return float(F), grad, hess


def row_objective(z: Any, reports: list[ParsedReport], row: int,
                  ridge: float, classes: Optional[int] = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Public helper: ``(F, grad, hess)`` of one public row at ``z``.

    ``classes`` defaults to one past the largest label seen in the reports.
    """
    zarr = as_float_array("z", z, 1)
    if classes is not None:
        K = int(classes)
    else:
        K = int(max(rep.labels.max() for rep in reports)) + 1
    if zarr.shape[0] != K:
        raise InputError(f"z must have length {K}, got {zarr.shape[0]}")
    ridge_v = as_range_scalar("ridge", ridge, 0.0, np.inf)
    return _row_terms(zarr, reports, row, ridge_v)


def _solve_row(
    reports: list[ParsedReport],
    row: int,
    K: int,
    ridge: float,
    max_iter: int,
) -> dict:
    """Damped Newton for a single row.  Returns a diagnostics dict."""
    any_present = any(rep.present[row] for rep in reports)
    z = np.zeros(K)
    F, g, H = _row_terms(z, reports, row, ridge)
    gnorm = float(np.max(np.abs(g))) if K else 0.0
    history = [F]

    if not any_present:
        # Only the ridge term: unique minimiser z = 0, exact gradient 0.
        return {
            "z": z,
            "objective": float(F),
            "gradient": g,
            "iterations": 0,
            "grad_norm": 0.0,
            "converged": True,
            "history": history,
            "message": "no present report; exact ridge minimiser z=0",
        }

    iterations = 0
    converged = gnorm <= GRAD_TOL
    message = "converged" if converged else ""
    numerical_failure = False

    while not converged and iterations < max_iter:
        iterations += 1
        try:
            # Symmetric positive-definite system (ridge > 0 guarantees it).
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            numerical_failure = True
            message = "linear solve failed; returning best finite iterate"
            break
        if not np.all(np.isfinite(step)):
            numerical_failure = True
            message = "non-finite Newton step; returning best finite iterate"
            break

        # Backtracking line search on the convex objective.
        slope = float(g @ step)
        alpha = 1.0
        accepted = False
        for _ in range(60):
            z_new = z + alpha * step
            F_new, g_new, H_new = _row_terms(z_new, reports, row, ridge)
            if np.isfinite(F_new) and F_new <= F + 1e-4 * alpha * slope:
                accepted = True
                break
            alpha *= 0.5
        if not accepted or not np.all(np.isfinite(z_new)) or not np.isfinite(F_new):
            numerical_failure = True
            message = "line search failed; returning best finite iterate"
            break

        z, F, g, H = z_new, F_new, g_new, H_new
        history.append(F)
        gnorm = float(np.max(np.abs(g)))
        if gnorm <= GRAD_TOL:
            converged = True
            message = "converged"
            break

    if not converged and not numerical_failure:
        if iterations >= max_iter:
            message = f"iteration cap ({max_iter}) reached; returning best finite iterate"
        else:
            message = "stopped before convergence; returning best finite iterate"

    return {
        "z": z,
        "objective": float(F),
        "gradient": g,
        "iterations": iterations,
        "grad_norm": gnorm,
        "converged": bool(converged),
        "history": history,
        "message": message,
    }


def consensus(
    reports: Sequence[Mapping],
    classes: Any,
    ridge: Any = 0.05,
    max_iter: Any = 500,
    n_rows: Optional[int] = None,
) -> dict:
    """Aggregate reports into a global teacher ``Q = softmax(z)``.

    Parameters
    ----------
    reports, classes : see :func:`parse_reports`.
    ridge : float in ``[0.05, 1]``; makes ``F`` strictly convex.
    max_iter : positive int; per-row Newton iteration cap (default 500).

    Returns
    -------
    dict with ``z`` (R, K), ``Q`` (R, K) row-stochastic, ``objective`` (R,)
    per-row original ``F_r(z)``, ``total_objective``, ``gradient`` (R, K),
    ``converged`` (bool, all rows), and ``diagnostics`` containing per-row
    iteration counts, gradient norms, messages and objectives.
    """
    parsed = parse_reports(reports, classes, n_rows=n_rows)
    K = int(classes)
    ridge_v = as_range_scalar("ridge", ridge, 0.05, 1.0)
    if isinstance(max_iter, (bool, np.bool_)):
        raise InputError("max_iter must be a positive integer")
    try:
        max_iter_i = int(max_iter)
    except (TypeError, ValueError) as exc:
        raise InputError("max_iter must be a positive integer") from exc
    if max_iter_i != float(max_iter) or max_iter_i < 1:
        raise InputError("max_iter must be a positive integer")
    if max_iter_i > 10_000_000:
        raise InputError("max_iter is unreasonably large")

    R = parsed[0].n_rows
    z = np.zeros((R, K))
    obj = np.zeros(R)
    grad = np.zeros((R, K))
    iterations = np.zeros(R, dtype=np.int64)
    grad_norms = np.zeros(R)
    converged = np.zeros(R, dtype=bool)
    messages: list[str] = []
    histories: list[list[float]] = []

    for r in range(R):
        sol = _solve_row(parsed, r, K, ridge_v, max_iter_i)
        z[r] = sol["z"]
        obj[r] = sol["objective"]
        grad[r] = sol["gradient"]
        iterations[r] = sol["iterations"]
        grad_norms[r] = sol["grad_norm"]
        converged[r] = sol["converged"]
        messages.append(sol["message"])
        histories.append(sol["history"])

    Q = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    return {
        "z": z,
        "Q": Q,
        "objective": obj,
        "total_objective": float(obj.sum()),
        "gradient": grad,
        "converged": bool(np.all(converged)),
        "diagnostics": {
            "iterations": iterations,
            "max_abs_gradient": grad_norms,
            "converged": converged,
            "message": messages,
            "history": histories,
            "ridge": ridge_v,
            "max_iter": max_iter_i,
            "n_rows": R,
            "classes": K,
            "grad_tolerance": GRAD_TOL,
        },
    }


def aggregate(reports: Sequence[Mapping], classes: Any, **kwargs) -> np.ndarray:
    """Convenience wrapper returning only the global ``Q`` matrix."""
    return consensus(reports, classes, **kwargs)["Q"]


def padded_average(reports: Sequence[Mapping], classes: Any,
                   n_rows: Optional[int] = None) -> np.ndarray:
    """Baseline: zero-pad each report to all classes and average.

    Kept only for contrast in the demo; it is *not* the specified consensus and
    treats "no such local head" as "impossible", which is exactly the failure
    mode the collaborative objective avoids.
    """
    parsed = parse_reports(reports, classes, n_rows=n_rows)
    K = int(classes)
    R = parsed[0].n_rows
    out = np.zeros((R, K))
    mass = np.zeros(R)
    for rep in parsed:
        for r in range(R):
            if rep.present[r]:
                out[r, rep.labels] += rep.weight * rep.probabilities[r]
                mass[r] += rep.weight
    for r in range(R):
        if mass[r] > 0:
            out[r] /= mass[r]
        else:
            out[r] = 1.0 / K
    return out
