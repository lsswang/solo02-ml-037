"""Independent numerical checks used by the test-suite and the report.

Nothing here is used by the production entry points; these are *verification*
routines that deliberately re-derive things by finite differences or by an
independent solver so a bug in the hand-written gradients cannot hide.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .local_model import objective
from .round import PARAM_KEYS
from .validation import prepare_model

__all__ = [
    "finite_difference_local_grads",
    "check_local_gradients",
    "least_squares_consensus",
]


def _flat_params(model: Mapping) -> tuple[np.ndarray, list[tuple[str, tuple]]]:
    A, b, C, c = prepare_model(model)
    order = [("A", A), ("b", b), ("C", C), ("c", c)]
    vec = np.concatenate([v.ravel() for _, v in order])
    return vec, [(k, v.shape) for k, v in order]


def _unflatten(vec: np.ndarray, shapes: list[tuple[str, tuple]]) -> dict:
    out = {}
    i = 0
    for key, shape in shapes:
        size = int(np.prod(shape))
        out[key] = vec[i:i + size].reshape(shape).copy()
        i += size
    return out


def finite_difference_local_grads(
    model: Mapping,
    X: Any,
    labels: Any,
    weights: Any,
    reg: Any,
    *,
    Q: Any = None,
    S: Any = None,
    tau: Any = 0.0,
    public_X: Any = None,
    public_weights: Any = None,
    classes: Any = None,
    labels_are_global: bool = False,
    eps: float = 1e-6,
) -> dict:
    """Central-difference gradient of the local objective, same shapes as model."""
    vec, shapes = _flat_params(model)
    grads = {}
    for i in range(vec.size):
        plus = vec.copy(); plus[i] += eps
        minus = vec.copy(); minus[i] -= eps
        kw = dict(Q=Q, S=S, tau=tau, public_X=public_X,
                  public_weights=public_weights, classes=classes,
                  labels_are_global=labels_are_global)
        lp, _ = objective(_unflatten(plus, shapes), X, labels, weights, reg, **kw)
        lm, _ = objective(_unflatten(minus, shapes), X, labels, weights, reg, **kw)
        grads[i] = (lp - lm) / (2 * eps)
    flat = np.array([grads[i] for i in range(vec.size)])
    return _unflatten(flat, shapes)


def check_local_gradients(
    model: Mapping,
    X: Any,
    labels: Any,
    weights: Any,
    reg: Any,
    *,
    Q: Any = None,
    S: Any = None,
    tau: Any = 0.0,
    public_X: Any = None,
    public_weights: Any = None,
    classes: Any = None,
    labels_are_global: bool = False,
    eps: float = 1e-6,
) -> dict:
    """Return ``{param: (max_abs_err, max_rel_err)}`` versus finite differences."""
    kw = dict(Q=Q, S=S, tau=tau, public_X=public_X,
              public_weights=public_weights, classes=classes,
              labels_are_global=labels_are_global)
    _, analytic = objective(model, X, labels, weights, reg, **kw)
    numeric = finite_difference_local_grads(
        model, X, labels, weights, reg, eps=eps, **kw
    )
    out = {}
    for key in PARAM_KEYS:
        a = np.asarray(analytic[key], dtype=float)
        n = np.asarray(numeric[key], dtype=float)
        abs_err = float(np.max(np.abs(a - n)))
        denom = np.maximum(np.abs(n), 1e-12)
        rel_err = float(np.max(np.abs(a - n) / denom))
        out[key] = (abs_err, rel_err)
    return out


def least_squares_consensus(
    reports: list,
    classes: int,
    ridge: float,
    n_rows: int,
    z_init: np.ndarray | None = None,
) -> np.ndarray:
    """Independent cross-check of the consensus minimiser.

    Uses SciPy's mature ``trust-ncg`` trust-region optimiser with the analytic
    gradient and Hessian for each row.  This is algorithmically distinct from the
    hand-written damped Newton in :mod:`starter.consensus`, so an error in one
    solver is unlikely to be mirrored by the other.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    K = int(classes)
    z = np.zeros((n_rows, K)) if z_init is None else z_init.astype(float).copy()
    out = np.zeros((n_rows, K))
    for r in range(n_rows):
        def fg(x):
            F = 0.5 * ridge * float(x @ x)
            g = ridge * x.copy()
            H = ridge * np.eye(K)
            for rep in reports:
                if not rep.present[r]:
                    continue
                ids = np.asarray(rep.labels)
                p = np.asarray(rep.probabilities[r])
                a = rep.weight
                zs = x[ids]
                lse = logsumexp(zs)
                q = np.exp(zs - lse)
                F += a * (lse - float(p @ zs))
                g[ids] += a * (q - p)
                H[np.ix_(ids, ids)] += a * (np.diag(q) - np.outer(q, q))
            return F, g, H

        res = minimize(
            lambda x: fg(x)[0],
            x0=z[r],
            jac=lambda x: fg(x)[1],
            hess=lambda x: fg(x)[2],
            method="trust-ncg",
            options={"gtol": 1e-12, "maxiter": 5000},
        )
        out[r] = res.x
    return out
