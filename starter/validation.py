"""Strict input validation shared by local and server entry points.

Every public numerical entry point funnels through these helpers so malformed
shapes, non-finite numbers, bad probabilities, bad masks and bad weights raise
:class:`starter.errors.InputError` (a ``ValueError``) instead of producing a
silent ``nan`` downstream.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Sequence

import numpy as np

from .errors import InputError

__all__ = [
    "InputError",
    "as_float_array",
    "as_int_vector",
    "as_label_vector",
    "as_feature_matrix",
    "as_weight_vector",
    "as_bool_mask",
    "as_range_scalar",
    "prepare_model",
    "validate_global_teacher",
    "validate_report_probabilities",
    "teacher_from_global",
]


def as_float_array(name: str, value: Any, ndim: int) -> np.ndarray:
    """Convert ``value`` to a finite float64 array with exactly ``ndim`` axes."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:  # ragged nested lists, strings, ...
        raise InputError(f"{name} must be a numeric array") from exc
    if arr.ndim != ndim:
        raise InputError(f"{name} must be {ndim}-dimensional, got {arr.ndim}")
    if arr.size == 0:
        raise InputError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{name} contains a non-finite value")
    return arr


def as_int_vector(
    name: str,
    value: Any,
    length: Optional[int] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> np.ndarray:
    """Convert to a 1-D finite integral int64 array."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be an integer vector") from exc
    if raw.ndim != 1:
        raise InputError(f"{name} must be 1-dimensional, got {raw.ndim}")
    if raw.size == 0:
        raise InputError(f"{name} must not be empty")
    if raw.dtype == bool:
        raise InputError(f"{name} must contain integers, not booleans")
    try:
        floats = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must contain integers") from exc
    if not np.all(np.isfinite(floats)):
        raise InputError(f"{name} contains a non-finite value")
    if not np.all(floats == np.floor(floats)):
        raise InputError(f"{name} must contain integers")
    if length is not None and floats.shape[0] != length:
        raise InputError(f"{name} must have length {length}, got {floats.shape[0]}")
    out = floats.astype(np.int64)
    if min_value is not None and np.any(out < min_value):
        raise InputError(f"{name} has a value below {min_value}")
    if max_value is not None and np.any(out > max_value):
        raise InputError(f"{name} has a value above {max_value}")
    return out


def as_label_vector(
    name: str,
    value: Any,
    classes: int,
    min_length: int = 1,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """Validate a class-subset list ``S``: distinct ints inside ``[0, classes)``."""
    arr = as_int_vector(name, value, min_value=0, max_value=classes - 1)
    if arr.shape[0] < min_length:
        raise InputError(f"{name} must contain at least {min_length} labels")
    if max_length is not None and arr.shape[0] > max_length:
        raise InputError(f"{name} must contain at most {max_length} labels")
    if np.unique(arr).shape[0] != arr.shape[0]:
        raise InputError(f"{name} contains duplicate labels")
    return arr


def as_feature_matrix(name: str, value: Any, dim: int, min_rows: int = 1,
                      max_rows: Optional[int] = None,
                      cols_max: Optional[float] = None) -> np.ndarray:
    arr = as_float_array(name, value, 2)
    if arr.shape[1] != dim:
        raise InputError(f"{name} must have {dim} columns, got {arr.shape[1]}")
    if arr.shape[0] < min_rows:
        raise InputError(f"{name} must have at least {min_rows} row(s)")
    if max_rows is not None and arr.shape[0] > max_rows:
        raise InputError(f"{name} must have at most {max_rows} row(s)")
    if cols_max is not None and np.any(np.abs(arr) > cols_max):
        raise InputError(f"{name} entries must lie in [-{cols_max}, {cols_max}]")
    return arr


def as_weight_vector(
    name: str,
    value: Any,
    length: Optional[int] = None,
    *,
    integer: bool = False,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> np.ndarray:
    arr = as_float_array(name, value, 1)
    if length is not None and arr.shape[0] != length:
        raise InputError(f"{name} must have length {length}, got {arr.shape[0]}")
    if np.any(arr <= 0.0):
        raise InputError(f"{name} must be strictly positive")
    if integer and not np.all(arr == np.floor(arr)):
        raise InputError(f"{name} must contain integers")
    if lo is not None and np.any(arr < lo):
        raise InputError(f"{name} has a value below {lo}")
    if hi is not None and np.any(arr > hi):
        raise InputError(f"{name} has a value above {hi}")
    return arr


def as_bool_mask(name: str, value: Any, length: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a boolean mask") from exc
    if raw.ndim != 1:
        raise InputError(f"{name} must be 1-dimensional, got {raw.ndim}")
    if raw.shape[0] != length:
        raise InputError(f"{name} must have length {length}, got {raw.shape[0]}")
    if raw.dtype == bool:
        return raw.astype(bool)
    try:
        vals = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a boolean mask") from exc
    if not np.all(np.isfinite(vals)):
        raise InputError(f"{name} contains a non-finite value")
    if not np.all((vals == 0.0) | (vals == 1.0)):
        raise InputError(f"{name} must contain only booleans/0/1")
    return vals.astype(bool)


def as_range_scalar(name: str, value: Any, lo: float, hi: float) -> float:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a finite scalar") from exc
    if arr.ndim != 0:
        raise InputError(f"{name} must be a scalar")
    v = float(arr)
    if not np.isfinite(v):
        raise InputError(f"{name} must be finite")
    if v < lo or v > hi:
        raise InputError(f"{name} must lie in [{lo}, {hi}], got {v}")
    return v


def prepare_model(model: Mapping, name: str = "model") -> tuple[np.ndarray, ...]:
    """Return validated ``(A, b, C, c)`` for the tanh classifier.

    Shapes: ``A (d, H)``, ``b (H,)``, ``C (H, L)``, ``c (L,)`` with the
    problem-bounded ``1 <= d <= 4`` and ``1 <= H <= 3``.
    """
    if not isinstance(model, Mapping):
        raise InputError(f"{name} must be a mapping with keys A, b, C, c")
    for key in ("A", "b", "C", "c"):
        if key not in model:
            raise InputError(f"{name} is missing parameter {key!r}")
    A = as_float_array(f"{name}.A", model["A"], 2)
    b = as_float_array(f"{name}.b", model["b"], 1)
    C = as_float_array(f"{name}.C", model["C"], 2)
    c = as_float_array(f"{name}.c", model["c"], 1)
    d, H = A.shape
    if not (1 <= d <= 4):
        raise InputError(f"{name}.A: input dimension must be in [1, 4], got {d}")
    if not (1 <= H <= 3):
        raise InputError(f"{name}.A: hidden width must be in [1, 3], got {H}")
    if b.shape != (H,):
        raise InputError(f"{name}.b must have shape ({H},), got {b.shape}")
    if C.shape[0] != H:
        raise InputError(f"{name}.C must have {H} rows, got {C.shape[0]}")
    L = C.shape[1]
    if L < 1:
        raise InputError(f"{name}.C must have at least one column")
    if c.shape != (L,):
        raise InputError(f"{name}.c must have shape ({L},), got {c.shape}")
    return A, b, C, c


def validate_global_teacher(Q: Any, name: str = "Q") -> np.ndarray:
    """Validate an external global teacher and return row-normalised ``Q``.

    Required: every entry ``>= 1e-300`` and ``|rowsum - 1| <= 1e-12``.
    """
    arr = as_float_array(name, Q, 2)
    if np.any(arr < 1e-300):
        raise InputError(f"{name} entries must all be >= 1e-300")
    sums = arr.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > 1e-12):
        raise InputError(f"{name} row sums must be within 1e-12 of 1")
    return arr / sums[:, None]


def validate_report_probabilities(
    p: Any, name: str, n_rows: Optional[int] = None
) -> np.ndarray:
    """Validate strict conditional probabilities and return normalised rows.

    Required: every entry ``>= 1e-12`` and ``|rowsum - 1| <= 1e-12``.
    """
    arr = as_float_array(name, p, 2)
    if n_rows is not None and arr.shape[0] != n_rows:
        raise InputError(f"{name} has {arr.shape[0]} rows, expected {n_rows}")
    if np.any(arr < 1e-12):
        raise InputError(f"{name} entries must all be >= 1e-12")
    sums = arr.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > 1e-12):
        raise InputError(f"{name} row sums must be within 1e-12 of 1")
    return arr / sums[:, None]


def teacher_from_global(
    Q: Any,
    S: Sequence[int],
    classes: Optional[int] = None,
    name: str = "Q",
) -> np.ndarray:
    """Build the conditional local teacher ``t[r,l] = Q[r,S[l]] / sum_{k in S} Q[r,k]``.

    This is a *renormalised* restriction to ``S``; it never leaves rows summing
    to less than one.
    """
    Qn = validate_global_teacher(Q, name)
    K = Qn.shape[1]
    if classes is not None and int(classes) != K:
        raise InputError(f"{name} has {K} columns but classes={classes}")
    Sarr = as_label_vector("S", S, K)
    denom = Qn[:, Sarr].sum(axis=1)
    # Entries are >= 1e-300 and |S| >= 1, so the denominator is strictly > 0.
    if np.any(denom <= 0.0) or not np.all(np.isfinite(denom)):
        raise InputError("teacher denominator is not strictly positive")
    return Qn[:, Sarr] / denom[:, None]
