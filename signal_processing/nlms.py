"""Normalized Least Mean Squares (NLMS) adaptive filter."""

from __future__ import annotations

import numpy as np

from signal_processing._adaptive import (
    validate_order,
    validate_pair,
    validate_positive,
    validate_sr,
)


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
    _ = params
    if reference is None:
        raise ValueError("Reference signal must be provided for NLMS enhancement.")

    d_arr, x_arr = validate_pair(noisy, reference)
    _ = validate_sr(sr)
    order_val = validate_order(order)
    mu_val = validate_positive(mu, "mu")
    if mu_val >= 2.0:
        raise ValueError(f"mu must be < 2 for NLMS, got {mu!r}")
    eps_val = validate_positive(eps, "eps")

    n_samples = len(d_arr)
    d = d_arr.astype(np.float64)
    x = x_arr.astype(np.float64)
    e = np.zeros(n_samples, dtype=np.float64)

    w = np.zeros(order_val, dtype=np.float64)
    xb = np.zeros(order_val, dtype=np.float64)

    for n in range(n_samples):
        # Update delay line: xb[0] is most recent sample x[n]
        xb[1:] = xb[:-1]
        xb[0] = x[n]

        y = np.dot(w, xb)
        err = d[n] - y
        e[n] = err

        norm = eps_val + np.dot(xb, xb)
        w += (mu_val / norm) * err * xb

    out = e.astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("NLMS filter diverged: output contains NaN or infinite values.")
    return out
