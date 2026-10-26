"""Normalized Least Mean Squares (NLMS) adaptive filter."""

from __future__ import annotations

import numpy as np


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = 16000,
    order: int = 64,
    mu: float = 0.5,
    eps: float = 1e-6,
    **params: object,
) -> np.ndarray:
    """Enhance noisy signal using the Normalized Least Mean Squares (NLMS) adaptive filter."""
    _ = sr, params
    if reference is None:
        raise ValueError("Reference signal must be provided for NLMS enhancement.")
    if len(reference) != len(noisy):
        raise ValueError(
            f"Reference length ({len(reference)}) must match noisy length ({len(noisy)})."
        )
    if not (0.0 < mu < 2.0):
        raise ValueError(f"Step size mu must satisfy 0 < mu < 2, got {mu}.")
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}.")

    n_samples = len(noisy)
    if n_samples == 0:
        return np.empty(0, dtype=np.float32)

    d = np.asarray(noisy, dtype=np.float64)
    x = np.asarray(reference, dtype=np.float64)
    e = np.zeros(n_samples, dtype=np.float64)

    w = np.zeros(order, dtype=np.float64)
    xb = np.zeros(order, dtype=np.float64)

    for n in range(n_samples):
        # Update delay line: xb[0] is most recent sample x[n]
        xb[1:] = xb[:-1]
        xb[0] = x[n]

        y = np.dot(w, xb)
        err = d[n] - y
        e[n] = err

        norm = eps + np.dot(xb, xb)
        w += (mu / norm) * err * xb

    out = e.astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("NLMS output contains NaN or infinite values.")
    return out
