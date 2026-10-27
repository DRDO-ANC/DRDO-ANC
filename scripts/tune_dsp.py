"""Parameter tuning and evaluation script for classical DSP audio enhancers."""

from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

# Optional soundfile import
try:
    import soundfile as sf
except ImportError:
    sf = None


def generate_clean_signal(
    sig_type: str,
    duration: float,
    sr: int = 16000,
) -> np.ndarray:
    """Generate deterministic synthetic clean signal (tone, two-tone, or chirp)."""
    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float32) / sr

    if sig_type == "tone":
        clean = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    elif sig_type == "two-tone":
        clean = (
            0.3 * np.sin(2.0 * np.pi * 500.0 * t) + 0.3 * np.sin(2.0 * np.pi * 1500.0 * t)
        ).astype(np.float32)
    elif sig_type == "chirp":
        f0, f1 = 200.0, 3000.0
        phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * (t**2) / duration)
        clean = (0.5 * np.sin(phase)).astype(np.float32)
    else:
        raise ValueError(f"Unknown signal type: {sig_type}")

    return clean


def generate_noise(
    noise_type: str,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Generate deterministic white or coloured noise with specified random seed."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples).astype(np.float32)

    if noise_type == "white":
        return white
    elif noise_type == "coloured":
        b, a = signal.butter(2, 0.25)
        coloured = signal.lfilter(b, a, white).astype(np.float32)
        return coloured
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def create_single_mic_signals(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    lead_in_s: float = 0.5,
    sr: int = 16000,
) -> tuple[np.ndarray, np.ndarray]:
    """Create single-mic clean and noisy signals with noise-only lead-in."""
    lead_in_samples = int(lead_in_s * sr)
    n_active = len(clean)

    s_pwr = float(np.mean(clean**2))
    noise_active = noise[:n_active]
    n_pwr = float(np.mean(noise_active**2))
    target_noise_pwr = s_pwr / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_pwr / (n_pwr + 1e-12)))

    scaled_noise = noise * scale

    full_clean = np.zeros(lead_in_samples + n_active, dtype=np.float32)
    full_clean[lead_in_samples:] = clean

    lead_noise = scaled_noise[:lead_in_samples]
    active_noise = scaled_noise[lead_in_samples : lead_in_samples + n_active]
    if len(active_noise) < n_active:
        pad_len = n_active - len(active_noise)
        active_noise = np.pad(active_noise, (0, pad_len), mode="wrap")

    full_noise = np.concatenate([lead_noise, active_noise])
    full_noisy = full_clean + full_noise
    return full_clean, full_noisy


def create_two_mic_signals(
    clean: np.ndarray,
    ref_noise: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create two-mic signals: reference noise and noisy signal via 16-tap FIR with 3-sample delay."""
    fir = np.zeros(19, dtype=np.float32)
    fir[3] = 0.8
    fir[4:19] = 0.05
    raw_acoustic_noise = signal.lfilter(fir, [1.0], ref_noise).astype(np.float32)

    s_pwr = float(np.mean(clean**2))
    n_pwr = float(np.mean(raw_acoustic_noise**2))
    target_noise_pwr = s_pwr / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_pwr / (n_pwr + 1e-12)))

    acoustic_noise = raw_acoustic_noise * scale
    scaled_ref = ref_noise * scale

    noisy = clean + acoustic_noise
    return clean, noisy, scaled_ref


def compute_snr_gain(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    lead_in_samples: int = 0,
) -> float:
    """Compute SNR improvement in decibels over active speech duration."""
    c = clean[lead_in_samples:]
    n_in = noisy[lead_in_samples:] - c
    n_out = enhanced[lead_in_samples:] - c

    s_pwr = float(np.mean(c**2))
    p_in = float(np.mean(n_in**2))
    p_out = float(np.mean(n_out**2))

    snr_in = 10.0 * np.log10(s_pwr / (p_in + 1e-12))
    snr_out = 10.0 * np.log10(s_pwr / (p_out + 1e-12))
    return float(snr_out - snr_in)


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


def run_spectral_sweep(
    quick: bool = False,
    seeds: list[int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run parameter sweep for spectral subtraction enhancer."""
    try:
        from signal_processing import spectral
    except ImportError:
        print("skipped: spectral not available on this branch")
        return None, []

    if seeds is None:
        seeds = [42, 43]
    if quick:
        seeds = [seeds[0]]

    alphas = [1.0, 2.0, 4.0] if quick else [1.0, 1.5, 2.0, 3.0, 4.0]
    betas = [0.02, 0.05] if quick else [0.01, 0.02, 0.05, 0.1]
    snrs = [-5.0, 10.0] if quick else [-5.0, 0.0, 5.0, 10.0]
    sig_types = ["tone"] if quick else ["tone", "two-tone", "chirp"]
    noise_types = ["white"] if quick else ["white", "coloured"]

    sig = inspect.signature(spectral.enhance)
    has_noise_mode = "noise_mode" in sig.parameters

    noise_modes = ["initial"]
    if has_noise_mode and not quick:
        noise_modes = ["initial", "percentile", "min_stats"]

    sr = 16000
    dur = 2.0
    lead_s = 0.5
    lead_samples = int(lead_s * sr)

    cached_signals = {}
    for st in sig_types:
        clean = generate_clean_signal(st, dur, sr=sr)
        for seed in seeds:
            for nt in noise_types:
                noise = generate_noise(nt, len(clean) + lead_samples, seed)
                for snr in snrs:
                    key = (st, seed, nt, snr)
                    c_sig, n_sig = create_single_mic_signals(clean, noise, snr, lead_in_s=lead_s, sr=sr)
                    cached_signals[key] = (c_sig, n_sig)

    results = []
    for nm in noise_modes:
        for a in alphas:
            for b in betas:
                snr_gains: dict[float, list[float]] = {s: [] for s in snrs}
                kwargs: dict[str, Any] = {"alpha": a, "beta": b}
                if has_noise_mode:
                    kwargs["noise_mode"] = nm

                for key, (c_sig, n_sig) in cached_signals.items():
                    snr = key[3]
                    try:
                        enhanced = spectral.enhance(n_sig, sr=sr, **kwargs)
                        gain = compute_snr_gain(c_sig, n_sig, enhanced, lead_in_samples=lead_samples)
                        snr_gains[snr].append(gain)
                    except (ValueError, RuntimeError, FloatingPointError):
                        continue

                row: dict[str, Any] = {
                    "method": "spectral",
                    "alpha": a,
                    "beta": b,
                    "noise_mode": nm,
                }
                all_gains = []
                for s in snrs:
                    mean_s = float(np.mean(snr_gains[s])) if snr_gains[s] else float("nan")
                    row[f"gain_{s:g}dB"] = mean_s
                    if snr_gains[s]:
                        all_gains.extend(snr_gains[s])

                row["mean_gain"] = float(np.mean(all_gains)) if all_gains else float("-inf")
                row["worst_gain"] = float(np.min(all_gains)) if all_gains else float("-inf")
                results.append(row)

    return {"snrs": snrs, "has_noise_mode": has_noise_mode}, results


def run_wiener_sweep(
    quick: bool = False,
    seeds: list[int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run parameter sweep for Wiener filter enhancer."""
    try:
        from signal_processing import wiener
    except ImportError:
        print("skipped: wiener not available on this branch")
        return None, []

    if seeds is None:
        seeds = [42, 43]
    if quick:
        seeds = [seeds[0]]

    xi_min_dbs = [-35.0, -25.0, -15.0] if quick else [-35.0, -30.0, -25.0, -20.0, -15.0]
    snrs = [-5.0, 10.0] if quick else [-5.0, 0.0, 5.0, 10.0]
    sig_types = ["tone"] if quick else ["tone", "two-tone", "chirp"]
    noise_types = ["white"] if quick else ["white", "coloured"]

    sig = inspect.signature(wiener.enhance)
    has_noise_mode = "noise_mode" in sig.parameters
    has_dd = "decision_directed" in sig.parameters

    dd_opts = [False]
    if has_dd and not quick:
        dd_opts = [False, True]

    noise_modes = ["initial"]
    if has_noise_mode and not quick:
        noise_modes = ["initial", "percentile", "min_stats"]

    sr = 16000
    dur = 2.0
    lead_s = 0.5
    lead_samples = int(lead_s * sr)

    cached_signals = {}
    for st in sig_types:
        clean = generate_clean_signal(st, dur, sr=sr)
        for seed in seeds:
            for nt in noise_types:
                noise = generate_noise(nt, len(clean) + lead_samples, seed)
                for snr in snrs:
                    key = (st, seed, nt, snr)
                    c_sig, n_sig = create_single_mic_signals(clean, noise, snr, lead_in_s=lead_s, sr=sr)
                    cached_signals[key] = (c_sig, n_sig)

    results = []
    for nm in noise_modes:
        for dd in dd_opts:
            for xi_db in xi_min_dbs:
                xi_val = 10.0 ** (xi_db / 10.0)
                snr_gains: dict[float, list[float]] = {s: [] for s in snrs}
                kwargs: dict[str, Any] = {"xi_min": xi_val}
                if has_noise_mode:
                    kwargs["noise_mode"] = nm
                if has_dd:
                    kwargs["decision_directed"] = dd

                for key, (c_sig, n_sig) in cached_signals.items():
                    snr = key[3]
                    try:
                        enhanced = wiener.enhance(n_sig, sr=sr, **kwargs)
                        gain = compute_snr_gain(c_sig, n_sig, enhanced, lead_in_samples=lead_samples)
                        snr_gains[snr].append(gain)
                    except (ValueError, RuntimeError, FloatingPointError):
                        continue

                row: dict[str, Any] = {
                    "method": "wiener",
                    "xi_min_db": xi_db,
                    "noise_mode": nm,
                    "decision_directed": dd,
                }
                all_gains = []
                for s in snrs:
                    mean_s = float(np.mean(snr_gains[s])) if snr_gains[s] else float("nan")
                    row[f"gain_{s:g}dB"] = mean_s
                    if snr_gains[s]:
                        all_gains.extend(snr_gains[s])

                row["mean_gain"] = float(np.mean(all_gains)) if all_gains else float("-inf")
                row["worst_gain"] = float(np.min(all_gains)) if all_gains else float("-inf")
                results.append(row)

    return {"snrs": snrs, "has_noise_mode": has_noise_mode, "has_dd": has_dd}, results


def run_nlms_sweep(
    quick: bool = False,
    seeds: list[int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run parameter sweep for NLMS adaptive filter enhancer."""
    try:
        from signal_processing import nlms
    except ImportError:
        print("skipped: nlms not available on this branch")
        return None, []

    if seeds is None:
        seeds = [42, 43]
    if quick:
        seeds = [seeds[0]]

    orders = [32, 64] if quick else [16, 32, 64, 128]
    mus = [0.1, 0.5] if quick else [0.01, 0.05, 0.1, 0.5, 1.0]
    snrs = [-5.0, 10.0] if quick else [-5.0, 0.0, 5.0, 10.0]
    sig_types = ["tone"] if quick else ["tone", "two-tone", "chirp"]
    noise_types = ["white"] if quick else ["white", "coloured"]

    sr = 16000
    dur = 2.0

    cached_signals = {}
    for st in sig_types:
        clean = generate_clean_signal(st, dur, sr=sr)
        for seed in seeds:
            for nt in noise_types:
                ref_noise = generate_noise(nt, len(clean), seed)
                for snr in snrs:
                    key = (st, seed, nt, snr)
                    c_sig, n_sig, r_sig = create_two_mic_signals(clean, ref_noise, snr)
                    cached_signals[key] = (c_sig, n_sig, r_sig)

    results = []
    for order in orders:
        for mu in mus:
            snr_gains: dict[float, list[float]] = {s: [] for s in snrs}
            for key, (c_sig, n_sig, r_sig) in cached_signals.items():
                snr = key[3]
                try:
                    enhanced = nlms.enhance(n_sig, reference=r_sig, sr=sr, order=order, mu=mu)
                    gain = compute_snr_gain(c_sig, n_sig, enhanced)
                    snr_gains[snr].append(gain)
                except (ValueError, RuntimeError, FloatingPointError):
                    continue

            row: dict[str, Any] = {
                "method": "nlms",
                "order": order,
                "mu": mu,
            }
            all_gains = []
            for s in snrs:
                mean_s = float(np.mean(snr_gains[s])) if snr_gains[s] else float("nan")
                row[f"gain_{s:g}dB"] = mean_s
                if snr_gains[s]:
                    all_gains.extend(snr_gains[s])

            row["mean_gain"] = float(np.mean(all_gains)) if all_gains else float("-inf")
            row["worst_gain"] = float(np.min(all_gains)) if all_gains else float("-inf")
            results.append(row)

    return {"snrs": snrs}, results


def run_lms_sweep(
    quick: bool = False,
    seeds: list[int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run parameter sweep for LMS adaptive filter enhancer."""
    try:
        from signal_processing import lms
    except ImportError:
        print("skipped: lms not available on this branch")
        return None, []

    if seeds is None:
        seeds = [42, 43]
    if quick:
        seeds = [seeds[0]]

    orders = [16, 64] if quick else [16, 32, 64]
    mus = [0.001, 0.005] if quick else [0.0005, 0.001, 0.002, 0.005]
    snrs = [-5.0, 10.0] if quick else [-5.0, 0.0, 5.0, 10.0]
    sig_types = ["tone"] if quick else ["tone", "two-tone", "chirp"]
    noise_types = ["white"] if quick else ["white", "coloured"]

    sr = 16000
    dur = 2.0

    cached_signals = {}
    for st in sig_types:
        clean = generate_clean_signal(st, dur, sr=sr)
        for seed in seeds:
            for nt in noise_types:
                ref_noise = generate_noise(nt, len(clean), seed)
                for snr in snrs:
                    key = (st, seed, nt, snr)
                    c_sig, n_sig, r_sig = create_two_mic_signals(clean, ref_noise, snr)
                    cached_signals[key] = (c_sig, n_sig, r_sig)

    results = []
    for order in orders:
        for mu in mus:
            snr_gains: dict[float, list[float]] = {s: [] for s in snrs}
            diverged = False
            for key, (c_sig, n_sig, r_sig) in cached_signals.items():
                snr = key[3]
                try:
                    enhanced = lms.enhance(n_sig, reference=r_sig, sr=sr, order=order, mu=mu)
                    if not np.all(np.isfinite(enhanced)):
                        diverged = True
                        break
                    gain = compute_snr_gain(c_sig, n_sig, enhanced)
                    snr_gains[snr].append(gain)
                except (ValueError, FloatingPointError):
                    diverged = True
                    break

            row: dict[str, Any] = {
                "method": "lms",
                "order": order,
                "mu": mu,
                "diverged": diverged,
            }
            if diverged:
                for s in snrs:
                    row[f"gain_{s:g}dB"] = "diverged"
                row["mean_gain"] = float("-inf")
                row["worst_gain"] = float("-inf")
            else:
                all_gains = []
                for s in snrs:
                    mean_s = float(np.mean(snr_gains[s])) if snr_gains[s] else float("nan")
                    row[f"gain_{s:g}dB"] = mean_s
                    if snr_gains[s]:
                        all_gains.extend(snr_gains[s])
                row["mean_gain"] = float(np.mean(all_gains)) if all_gains else float("-inf")
                row["worst_gain"] = float(np.min(all_gains)) if all_gains else float("-inf")

            results.append(row)

    return {"snrs": snrs}, results


def analyze_sweep_results(
    method_name: str,
    meta: dict[str, Any],
    results: list[dict[str, Any]],
    param_keys: list[str],
) -> None:
    """Analyze sweep results, print summary table, best per SNR, and recommended default."""
    if not results:
        return

    snrs = meta["snrs"]
    headers = [k.replace("_", " ").title() for k in param_keys] + [
        f"{s:g} dB" for s in snrs
    ] + ["Mean Gain", "Worst Gain"]

    rows = []
    for r in results:
        row_vals = [r[k] for k in param_keys]
        for s in snrs:
            val = r[f"gain_{s:g}dB"]
            row_vals.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
        m_gain = r["mean_gain"]
        w_gain = r["worst_gain"]
        row_vals.append(f"{m_gain:.2f}" if isinstance(m_gain, (int, float)) and np.isfinite(m_gain) else "N/A")
        row_vals.append(f"{w_gain:.2f}" if isinstance(w_gain, (int, float)) and np.isfinite(w_gain) else "N/A")
        rows.append(row_vals)

    print(f"\n### Results: {method_name.upper()}\n")
    print(format_table(headers, rows))

    # Best per input SNR
    best_per_snr = {}
    for s in snrs:
        valid_rows = [r for r in results if isinstance(r[f"gain_{s:g}dB"], (int, float))]
        if valid_rows:
            best_r = max(valid_rows, key=lambda r: float(r[f"gain_{s:g}dB"]))
            setting_str = ", ".join(f"{k}={best_r[k]}" for k in param_keys)
            best_per_snr[s] = (setting_str, best_r[f"gain_{s:g}dB"])

    print(f"\nBest Settings per Input SNR for {method_name.upper()}:")
    for s, (setting, g) in best_per_snr.items():
        print(f"  - Input SNR {s:g} dB: {setting} (Gain: {g:.2f} dB)")

    # Check discrepancy between low (-5 dB) and high (10 dB)
    low_s = min(snrs)
    high_s = max(snrs)
    if low_s in best_per_snr and high_s in best_per_snr:
        if best_per_snr[low_s][0] != best_per_snr[high_s][0]:
            print(
                f"  * Note: Best setting differs between low ({low_s:g} dB) and high ({high_s:g} dB) SNR: "
                f"[{best_per_snr[low_s][0]}] vs [{best_per_snr[high_s][0]}]."
            )
        else:
            print(
                f"  * Note: Best setting is consistent between low ({low_s:g} dB) and high ({high_s:g} dB) SNR: "
                f"[{best_per_snr[low_s][0]}]."
            )

    # Recommended default: best mean gain over all SNRs
    valid_overall = [r for r in results if np.isfinite(r["mean_gain"])]
    if valid_overall:
        best_overall = max(valid_overall, key=lambda r: r["mean_gain"])
        def_setting = ", ".join(f"{k}={best_overall[k]}" for k in param_keys)
        print(
            f"\nRecommended Default for {method_name.upper()}: {def_setting} "
            f"(Mean Gain: {best_overall['mean_gain']:.2f} dB, Worst-Case Gain: {best_overall['worst_gain']:.2f} dB)\n"
        )


def main(argv: list[str] | None = None) -> int:
    """Run parameter sweep and evaluation CLI."""
    parser = argparse.ArgumentParser(description="Tune classical DSP filter parameters.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--quick", action="store_true", help="Run fast reduced parameter sweep")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional output CSV path")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest CSV path")
    parser.add_argument("--data-root", type=str, default=None, help="Optional dataset root directory")

    args = parser.parse_args(argv)

    if args.manifest or args.data_root:
        if sf is None:
            print("Notice: soundfile is not installed; ignoring manifest and staying on synthetic dataset.")
        else:
            print(f"Notice: manifest/data-root requested ({args.manifest}), running synthetic benchmark subset.")

    base_seed = args.seed
    seeds = [base_seed] if args.quick else [base_seed, base_seed + 1]

    all_csv_records: list[dict[str, Any]] = []

    # 1. Spectral
    meta_spec, res_spec = run_spectral_sweep(quick=args.quick, seeds=seeds)
    if meta_spec and res_spec:
        param_keys = ["alpha", "beta"]
        if meta_spec["has_noise_mode"]:
            param_keys.append("noise_mode")
        analyze_sweep_results("spectral", meta_spec, res_spec, param_keys)
        all_csv_records.extend(res_spec)

    # 2. Wiener
    meta_wien, res_wien = run_wiener_sweep(quick=args.quick, seeds=seeds)
    if meta_wien and res_wien:
        param_keys = ["xi_min_db"]
        if meta_wien["has_noise_mode"]:
            param_keys.append("noise_mode")
        if meta_wien["has_dd"]:
            param_keys.append("decision_directed")
        analyze_sweep_results("wiener", meta_wien, res_wien, param_keys)
        all_csv_records.extend(res_wien)

    # 3. NLMS
    meta_nlms, res_nlms = run_nlms_sweep(quick=args.quick, seeds=seeds)
    if meta_nlms and res_nlms:
        analyze_sweep_results("nlms", meta_nlms, res_nlms, ["order", "mu"])
        all_csv_records.extend(res_nlms)

    # 4. LMS
    meta_lms, res_lms = run_lms_sweep(quick=args.quick, seeds=seeds)
    if meta_lms and res_lms:
        analyze_sweep_results("lms", meta_lms, res_lms, ["order", "mu"])
        all_csv_records.extend(res_lms)

    if args.out_csv and all_csv_records:
        out_p = Path(args.out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        keys = set()
        for rec in all_csv_records:
            keys.update(rec.keys())
        fieldnames = sorted(keys)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_records)
        print(f"Exported sweep results to {out_p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
