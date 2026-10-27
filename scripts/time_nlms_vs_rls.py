"""Benchmark comparing NLMS and throwaway RLS adaptive filters."""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import time
from typing import Any

import numpy as np
from scipy import signal

# Lazy attempt to import signal_processing.nlms
_HAS_OFFICIAL_NLMS = False
try:
    from signal_processing import nlms as official_nlms  # type: ignore

    _HAS_OFFICIAL_NLMS = True
except ImportError:
    official_nlms = None


def local_nlms_filter(
    noisy: np.ndarray,
    reference: np.ndarray,
    order: int = 64,
    mu: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Enhance noisy signal using fallback local copy of NLMS update."""
    n_samples = len(noisy)
    d = noisy.astype(np.float64)
    x = reference.astype(np.float64)
    w = np.zeros(order, dtype=np.float64)
    xb = np.zeros(order, dtype=np.float64)
    e = np.zeros(n_samples, dtype=np.float64)

    for n in range(n_samples):
        xb[1:] = xb[:-1]
        xb[0] = x[n]

        y = np.dot(w, xb)
        err = d[n] - y
        e[n] = err

        norm = eps + np.dot(xb, xb)
        w += (mu / norm) * err * xb

    return e.astype(np.float32)


def run_nlms(
    noisy: np.ndarray,
    reference: np.ndarray,
    order: int = 64,
    mu: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Run NLMS using signal_processing.nlms if importable, else local copy."""
    if _HAS_OFFICIAL_NLMS and official_nlms is not None:
        return official_nlms.enhance(noisy, reference=reference, sr=16000, order=order, mu=mu, eps=eps)
    return local_nlms_filter(noisy, reference, order=order, mu=mu, eps=eps)


def throwaway_rls_filter(
    noisy: np.ndarray,
    reference: np.ndarray,
    order: int = 64,
    lmbda: float = 0.999,
    delta: float = 1.0,
) -> np.ndarray:
    """Enhance noisy signal using throwaway RLS filter (throwaway, for timing only)."""
    n_samples = len(noisy)
    d = noisy.astype(np.float64)
    x = reference.astype(np.float64)
    w = np.zeros(order, dtype=np.float64)
    P = np.eye(order, dtype=np.float64) * (1.0 / delta)
    xb = np.zeros(order, dtype=np.float64)
    e = np.zeros(n_samples, dtype=np.float64)
    inv_lmbda = 1.0 / lmbda

    for n in range(n_samples):
        xb[1:] = xb[:-1]
        xb[0] = x[n]

        y = np.dot(w, xb)
        err = d[n] - y
        e[n] = err

        pi_vec = P @ xb
        denom = lmbda + np.dot(xb, pi_vec)
        k = pi_vec / denom
        w += k * err
        P = (P - np.outer(k, pi_vec)) * inv_lmbda

    return e.astype(np.float32)


def compute_convergence_samples(
    error: np.ndarray,
    sr: int = 16000,
    drop_db: float = 20.0,
    win_ms: float = 100.0,
) -> int | None:
    """Calculate sample index where 100 ms-averaged error power drops by drop_db."""
    win_len = int(sr * win_ms / 1000.0)
    if len(error) < win_len:
        return None

    e2 = np.asarray(error, dtype=np.float64) ** 2
    cum = np.cumsum(np.insert(e2, 0, 0.0))
    num_windows = len(error) - win_len + 1
    avgs = (cum[win_len : win_len + num_windows] - cum[:num_windows]) / win_len

    start_pow = avgs[0]
    if start_pow <= 0.0:
        return 0

    thresh = start_pow * (10.0 ** (-drop_db / 10.0))
    below = np.where(avgs <= thresh)[0]
    if len(below) == 0:
        return None
    return int(below[0])


def format_table(
    headers: list[str],
    rows: list[list[Any]],
) -> str:
    """Format tabular data into a GitHub markdown table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    data_lines = [
        "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def run_benchmark(
    seed: int = 42,
    seconds: float = 1.0,
    quick: bool = False,
    sr: int = 16000,
) -> tuple[str, str, dict[str, str]]:
    """Run execution timing and convergence analysis for NLMS vs RLS."""
    n_samples = int(seconds * sr)
    rng = np.random.default_rng(seed)

    ref = rng.standard_normal(n_samples).astype(np.float32)
    fir = np.zeros(19, dtype=np.float32)
    fir[3] = 0.8
    fir[4:19] = 0.05
    noise = signal.lfilter(fir, [1.0], ref).astype(np.float32)
    noisy = noise.copy()

    lengths = [32, 64] if quick else [32, 64, 128, 256]
    hop_samples = 320
    budget_ms = 20.0

    timing_headers = [
        "Filter Length",
        "Method",
        "RTF",
        "us / sample",
        "ms / 20 ms hop",
        "Budget Status",
    ]
    timing_rows = []

    # Warmup length
    warm_samples = min(n_samples, 1000)

    for L in lengths:
        # NLMS
        run_nlms(noisy[:warm_samples], ref[:warm_samples], order=L, mu=0.5)
        nlms_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = run_nlms(noisy, ref, order=L, mu=0.5)
            nlms_times.append(time.perf_counter() - t0)
        best_nlms = min(nlms_times)
        rtf_nlms = best_nlms / seconds
        us_nlms = (best_nlms / n_samples) * 1e6
        hop_nlms = (best_nlms / n_samples) * hop_samples * 1e3
        status_nlms = "WITHIN" if hop_nlms <= budget_ms else "OVER"

        timing_rows.append([
            str(L),
            "NLMS",
            f"{rtf_nlms:.4f}",
            f"{us_nlms:.2f}",
            f"{hop_nlms:.3f}",
            status_nlms,
        ])

        # RLS (throwaway, for timing only)
        throwaway_rls_filter(noisy[:warm_samples], ref[:warm_samples], order=L, lmbda=0.999, delta=1.0)
        rls_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = throwaway_rls_filter(noisy, ref, order=L, lmbda=0.999, delta=1.0)
            rls_times.append(time.perf_counter() - t0)
        best_rls = min(rls_times)
        rtf_rls = best_rls / seconds
        us_rls = (best_rls / n_samples) * 1e6
        hop_rls = (best_rls / n_samples) * hop_samples * 1e3
        status_rls = "WITHIN" if hop_rls <= budget_ms else "OVER"

        timing_rows.append([
            str(L),
            "RLS",
            f"{rtf_rls:.4f}",
            f"{us_rls:.2f}",
            f"{hop_rls:.3f}",
            status_rls,
        ])

    timing_table = format_table(timing_headers, timing_rows)

    # Convergence test on 2.0 s signal
    conv_dur = 2.0
    conv_samples = int(conv_dur * sr)
    rng_c = np.random.default_rng(seed)
    ref_c = rng_c.standard_normal(conv_samples).astype(np.float32)
    noise_c = signal.lfilter(fir, [1.0], ref_c).astype(np.float32)

    conv_headers = ["Filter Length", "NLMS (mu=0.5)", "RLS (lambda=0.999)"]
    conv_rows = []

    for L in [32, 64]:
        e_nlms = run_nlms(noise_c, ref_c, order=L, mu=0.5)
        c_nlms = compute_convergence_samples(e_nlms, sr=sr, drop_db=20.0, win_ms=100.0)
        str_nlms = f"{c_nlms} samples ({c_nlms / sr * 1000.0:.1f} ms)" if c_nlms is not None else "not reached"

        e_rls = throwaway_rls_filter(noise_c, ref_c, order=L, lmbda=0.999, delta=1.0)
        c_rls = compute_convergence_samples(e_rls, sr=sr, drop_db=20.0, win_ms=100.0)
        str_rls = f"{c_rls} samples ({c_rls / sr * 1000.0:.1f} ms)" if c_rls is not None else "not reached"

        conv_rows.append([str(L), str_nlms, str_rls])

    conv_table = format_table(conv_headers, conv_rows)

    env_info = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "NumPy": np.__version__,
        "CPU Count": str(os.cpu_count() or 1),
        "Date": datetime.date.today().isoformat(),
        "NLMS Implementation": "signal_processing.nlms" if _HAS_OFFICIAL_NLMS else "local copy (fallback)",
    }

    return timing_table, conv_table, env_info


def main(argv: list[str] | None = None) -> int:
    """Run NLMS vs RLS timing and convergence benchmark CLI."""
    parser = argparse.ArgumentParser(description="Benchmark NLMS vs throwaway RLS filters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--seconds", type=float, default=1.0, help="Signal duration in seconds (default: 1.0)")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer lengths")

    args = parser.parse_args(argv)

    timing_table, conv_table, env_info = run_benchmark(
        seed=args.seed,
        seconds=args.seconds,
        quick=args.quick,
    )

    print("\n# NLMS vs RLS Benchmark Results\n")
    print(f"- **Implementation**: Using {env_info['NLMS Implementation']}")
    print(f"- **Platform**: {env_info['Platform']}")
    print(f"- **Python**: {env_info['Python']}")
    print(f"- **NumPy**: {env_info['NumPy']}")
    print(f"- **CPU Count**: {env_info['CPU Count']}")
    print(f"- **Date**: {env_info['Date']}\n")

    print("### Execution Timing Benchmark (Best of 3 after warm-up)\n")
    print(timing_table)
    print("\n")

    print("### Convergence Time (-20 dB Error Power Drop, 100 ms Averaging Window)\n")
    print(conv_table)
    print("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
