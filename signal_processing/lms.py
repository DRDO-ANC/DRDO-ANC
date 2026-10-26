"""Least Mean Squares (LMS) adaptive filter."""

from __future__ import annotations

import warnings

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
    mu: float = 0.002,
    **params: object,
) -> np.ndarray:
    """Enhance noisy signal using the Least Mean Squares (LMS) adaptive filter.

    Notes
    -----
    The stability check ``mu >= 2 / (order * mean(reference**2))`` is a
    classical *heuristic* derived from the independence assumption (i.e.
    the reference taps are uncorrelated white noise with constant power).
    It is **not** a universal stability guarantee: actual convergence
    depends on signal statistics, correlation structure, and non-
    stationarity.  In practice, convergence may require a significantly
    smaller ``mu`` than the bound suggests, especially for coloured or
    non-stationary reference signals.
    """
    _ = params
    if reference is None:
        raise ValueError("Reference signal must be provided for LMS enhancement.")

    d_arr, x_arr = validate_pair(noisy, reference)
    _ = validate_sr(sr)
    order_val = validate_order(order)
    mu_val = validate_positive(mu, "mu")

    n_samples = len(d_arr)
    d = d_arr.astype(np.float64)
    x = x_arr.astype(np.float64)

    # Stability heuristic: 2 / (order * E[x^2]).  This is a necessary
    # condition only under the independence assumption (white, stationary
    # reference).  Coloured or non-stationary signals may require a much
    # smaller mu for convergence.  See Notes above.
    ref_power = float(np.mean(x**2))
    stability_bound = 2.0 / (order_val * ref_power) if ref_power > 0.0 else float("inf")
    if mu_val >= stability_bound:
        warnings.warn(
            f"Step size mu={mu_val} exceeds or equals stability bound 2/(order*mean(ref**2))={stability_bound:.6f}.",
            UserWarning,
            stacklevel=2,
        )

    e = np.zeros(n_samples, dtype=np.float64)
    w = np.zeros(order_val, dtype=np.float64)
    xb = np.zeros(order_val, dtype=np.float64)

    for n in range(n_samples):
        xb[1:] = xb[:-1]
        xb[0] = x[n]

        y = np.dot(w, xb)
        err = d[n] - y
        e[n] = err

        w += mu_val * err * xb

    out = e.astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("LMS filter diverged: output contains NaN or infinite values.")
    return out
