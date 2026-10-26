"""Least Mean Squares (LMS) adaptive filter."""

from __future__ import annotations

import warnings

import numpy as np


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = 16000,
    order: int = 64,
    mu: float = 0.002,
    **params: object,
) -> np.ndarray:
    """Enhance noisy signal using the Least Mean Squares (LMS) adaptive filter."""
    _ = sr, params
    if reference is None:
        raise ValueError("Reference signal must be provided for LMS enhancement.")
    if len(reference) != len(noisy):
        raise ValueError(
            f"Reference length ({len(reference)}) must match noisy length ({len(noisy)})."
        )
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}.")

    n_samples = len(noisy)
    if n_samples == 0:
        return np.empty(0, dtype=np.float32)

    d = np.asarray(noisy, dtype=np.float64)
    x = np.asarray(reference, dtype=np.float64)

    ref_power = float(np.mean(x**2))
    stability_bound = 2.0 / (order * ref_power) if ref_power > 0.0 else float("inf")
    if mu >= stability_bound:
        warnings.warn(
            f"Step size mu={mu} exceeds or equals stability bound 2/(order*mean(ref**2))={stability_bound:.6f}.",
            UserWarning,
            stacklevel=2,
        )

    e = np.zeros(n_samples, dtype=np.float64)
    w = np.zeros(order, dtype=np.float64)
    xb = np.zeros(order, dtype=np.float64)

    for n in range(n_samples):
        xb[1:] = xb[:-1]
        xb[0] = x[n]

        y = np.dot(w, xb)
        err = d[n] - y
        e[n] = err

        w += mu * err * xb

    out = e.astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("LMS output contains NaN or infinite values.")
    return out
