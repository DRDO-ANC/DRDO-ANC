"""Internal validation helpers for adaptive filters."""

from __future__ import annotations

import numbers

import numpy as np


def validate_signal(x: object, name: str) -> np.ndarray:
    """Validate and convert an input signal to a 1-D float32 array."""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    # 1. dtype must be real numeric (int or float; reject bool, complex, str, object)
    if not (
        np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)
    ) or x.dtype == bool:
        raise ValueError(
            f"{name} must be a real numeric array (int or float), got dtype {x.dtype}"
        )

    # 2. ndim != 1
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {x.shape}")

    # 3. empty
    if x.size == 0:
        raise ValueError(f"{name} must not be empty")

    # 4. Convert with astype(np.float32) under np.errstate(over="ignore")
    with np.errstate(over="ignore"):
        x_f32 = x.astype(np.float32)

    # 5. Check non-finite values (NaN, inf, or overflowed to inf)
    if not np.all(np.isfinite(x_f32)):
        raise ValueError(
            f"{name} contains non-finite values (NaN, inf, or values too large for float32)"
        )

    return x_f32


def validate_pair(noisy: object, reference: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate noisy and reference signals, ensuring identical length."""
    noisy_arr = validate_signal(noisy, "noisy")
    ref_arr = validate_signal(reference, "reference")
    if len(noisy_arr) != len(ref_arr):
        raise ValueError(
            f"noisy and reference must have the same length, got {len(noisy_arr)} and {len(ref_arr)}"
        )
    return noisy_arr, ref_arr


def validate_sr(sr: object) -> int:
    """Validate sampling rate is a positive integer."""
    if (
        isinstance(sr, bool)
        or not (isinstance(sr, (int, numbers.Integral, np.integer)))
        or sr <= 0
    ):
        raise ValueError(f"sr must be a positive integer (Hz), got {sr!r}")
    return int(sr)


def validate_order(order: object) -> int:
    """Validate filter order is an integer >= 1."""
    if (
        isinstance(order, bool)
        or not (isinstance(order, (int, numbers.Integral, np.integer)))
        or order < 1
    ):
        raise ValueError(f"order must be an integer >= 1, got {order!r}")
    return int(order)


def validate_positive(value: object, name: str) -> float:
    """Validate value is a real, scalar, finite, strictly positive numeric value.

    Rejects complex numbers, NumPy complex scalars, booleans, arrays,
    NaN, infinity, zero, negative values, unrepresentable numbers, and
    non-numeric types with ``ValueError``.
    """
    if (
        isinstance(value, (bool, complex, np.complexfloating))
        or not isinstance(value, (int, float, numbers.Real, np.integer, np.floating))
    ):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")  # noqa: TRY004

    try:
        val_f = float(value)
    except (TypeError, OverflowError, ValueError) as err:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}") from err

    if not np.isfinite(val_f) or val_f <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")

    return val_f
