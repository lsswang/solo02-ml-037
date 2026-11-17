"""Local tanh classifier: forward pass and collaborative-distillation objective.

Public API
----------
forward(model, X) -> (H, Z, P)
objective(model, X, labels, weights, reg, *, Q, S, tau, public_X,
          public_weights, classes, labels_are_global) -> (loss, grads)

The local objective is

    L = sum_i w_i * (-log p_local[i, y_i]) / sum_i w_i
        + tau * sum_r pw_r * (-sum_l t[r,l] log p_local_public[r,l]) / sum_r pw_r
        + reg/2 * ||A||^2 + ||b||^2 + ||C||^2 + ||c||^2

with the *renormalised* local teacher

    t[r,l] = Q[r, S[l]] / sum_{k in S} Q[r,k] .

``t`` is treated as a constant: no gradient flows back into ``Q``.  With
``tau=0`` (the default) the teacher term vanishes and this is exactly the
ordinary exact-label supervised objective.  All softmax/log-sum-exp operations
are shifted for numerical stability.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
from scipy.special import logsumexp

from .validation import (
    InputError,
    as_feature_matrix,
    as_int_vector,
    as_label_vector,
    as_range_scalar,
    as_weight_vector,
    prepare_model,
    teacher_from_global,
)

__all__ = ["forward", "objective", "local_probabilities", "local_target"]


def _softmax(Z: np.ndarray) -> np.ndarray:
    return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))


def _forward_arrays(
    A: np.ndarray, b: np.ndarray, C: np.ndarray, c: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = np.tanh(X @ A + b)
    Z = H @ C + c
    return H, Z, _softmax(Z)


def forward(model: Mapping, X: Any):
    """Return ``(hidden, logits, probabilities)``."""
    A, b, C, c = prepare_model(model)
    Xv = as_feature_matrix("X", X, A.shape[0])
    return _forward_arrays(A, b, C, c, Xv)


def local_probabilities(model: Mapping, X: Any) -> np.ndarray:
    """Probabilities whose column ``l`` corresponds to local label ``S[l]``."""
    return forward(model, X)[2]


def local_target(Q: Any, S: Any, classes: Optional[int] = None) -> np.ndarray:
    """Renormalised local teacher ``t`` from global teacher ``Q`` and labels ``S``."""
    return teacher_from_global(Q, S, classes=classes)


def objective(
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
    classes: Optional[int] = None,
    labels_are_global: bool = False,
):
    """Local objective and gradients for ``A, b, C, c``.

    ``labels`` are local column indices ``0..|S|-1`` by default.  With
    ``labels_are_global=True`` they are exact global labels that must belong to
    ``S`` and are remapped to local columns.  The teacher term is evaluated on
    ``public_X`` (the same public sample identity the server aggregates).
    """
    A, b, C, c = prepare_model(model)
    d = A.shape[0]
    n_local_classes = C.shape[1]

    Xv = as_feature_matrix("X", X, d, min_rows=1, max_rows=30)
    n = Xv.shape[0]
    y = as_int_vector("labels", labels, length=n, min_value=0)
    w = as_weight_vector("weights", weights, length=n, integer=True, lo=1, hi=5)
    reg_v = as_range_scalar("reg", reg, 0.05, 1.0)
    tau_v = as_range_scalar("tau", tau, 0.0, 2.0)

    # Optional S: needed to renormalise the teacher and, with
    # labels_are_global, to remap global labels to local columns.
    Sarr = None
    if S is not None:
        K = classes if classes is not None else n_local_classes
        Sarr = as_label_vector("S", S, int(K), min_length=2, max_length=int(K))
        if Sarr.shape[0] != n_local_classes:
            raise InputError(
                f"S has {Sarr.shape[0]} labels but model outputs "
                f"{n_local_classes} columns"
            )

    if labels_are_global:
        if Sarr is None:
            raise InputError("labels_are_global=True requires S")
        position = {int(k): i for i, k in enumerate(Sarr)}
        if not np.all(np.isin(y, Sarr)):
            raise InputError("every global label y must belong to S")
        y = np.array([position[int(v)] for v in y], dtype=np.int64)
    elif np.any(y >= n_local_classes):
        raise InputError(f"labels must be local indices in [0, {n_local_classes - 1}]")

    # ---- supervised exact-label cross-entropy (weight-normalised) ----
    Htr, Ztr, Ptr = _forward_arrays(A, b, C, c, Xv)
    wsum = float(w.sum())
    logp = Ztr - logsumexp(Ztr, axis=1, keepdims=True)
    loss = float(np.sum(w * -logp[np.arange(n), y]) / wsum)
    Gtr = Ptr.copy()
    Gtr[np.arange(n), y] -= 1.0
    Gtr *= (w / wsum)[:, None]

    # ---- optional renormalised global-teacher distillation ----
    if tau_v > 0.0:
        if Q is None or Sarr is None or public_X is None or public_weights is None:
            raise InputError("tau > 0 requires Q, S, public_X and public_weights")
        Pup = as_feature_matrix("public_X", public_X, d)
        pn = Pup.shape[0]
        pw = as_weight_vector("public_weights", public_weights, length=pn)
        if Q is not None and np.asarray(Q).ndim == 2 and classes is not None:
            if np.asarray(Q).shape[1] != int(classes):
                raise InputError("Q does not have `classes` columns")
        t = teacher_from_global(Q, Sarr, classes=classes)
        if t.shape != (pn, n_local_classes):
            raise InputError(
                f"teacher shape {t.shape} != public rows x local classes "
                f"{(pn, n_local_classes)}"
            )
        Hp, Zp, Pp = _forward_arrays(A, b, C, c, Pup)
        pwsum = float(pw.sum())
        # -log p is guarded: validated teacher has entries >= 1e-300, and the
        # softmax underflows only at ~1e-308; clip the log for total safety.
        logPp = np.log(np.maximum(Pp, 1e-300))
        loss += tau_v * float(-np.sum(pw[:, None] * t * logPp) / pwsum)
        Gp = tau_v * (pw[:, None] * (Pp - t) / pwsum)
        H = np.vstack([Htr, Hp])
        dZ = np.vstack([Gtr, Gp])
        Xall = np.vstack([Xv, Pup])
        gC = Htr.T @ Gtr + Hp.T @ Gp
        gc = Gtr.sum(axis=0) + Gp.sum(axis=0)
    else:
        H = Htr
        dZ = Gtr
        Xall = Xv
        gC = Htr.T @ Gtr
        gc = Gtr.sum(axis=0)

    dH = (dZ @ C.T) * (1.0 - H * H)
    gA = Xall.T @ dH
    gb = dH.sum(axis=0)

    # ---- L2 over every parameter ----
    loss += 0.5 * reg_v * (
        float(np.sum(A * A)) + float(np.sum(b * b))
        + float(np.sum(C * C)) + float(np.sum(c * c))
    )
    grads = {
        "A": gA + reg_v * A,
        "b": gb + reg_v * b,
        "C": gC + reg_v * C,
        "c": gc + reg_v * c,
    }
    return float(loss), grads
