"""Self-contained tests for LMS and NLMS adaptive noise cancellers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

import nlms
from signal_processing import lms


def compute_snr_db(clean: np.ndarray, x: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in decibels."""
    err = x - clean
    s_pwr = float(np.mean(clean**2))
    n_pwr = float(np.mean(err**2))
    return float(10.0 * np.log10(s_pwr / (n_pwr + 1e-12)))


def generate_anc_signals(
    sr: int = 16000,
    duration: float = 4.0,
    tone_freq: float = 1000.0,
    tone_amp: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 4 s 1 kHz tone, white-noise reference, and FIR-filtered noisy signal."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float32) / sr

    tone = (tone_amp * np.sin(2.0 * np.pi * tone_freq * t)).astype(np.float32)
    ref = rng.normal(0.0, 1.0, size=n_samples).astype(np.float32)

    fir = np.zeros(16, dtype=np.float32)
    fir[3] = 0.8
    fir[4:] = 0.05
    noise = lfilter(fir, [1.0], ref).astype(np.float32)

    noisy = tone + noise

    max_abs = max(1.0, float(np.max(np.abs(noisy))), float(np.max(np.abs(ref))))
    tone = (tone / max_abs).astype(np.float32)
    noisy = (noisy / max_abs).astype(np.float32)
    ref = (ref / max_abs).astype(np.float32)
    return tone, noisy, ref


def test_nlms_snr_gain_and_error_power() -> None:
    """Verify NLMS achieves > 10 dB SNR gain and last-second error power < 10% of first second."""
    tone, noisy, ref = generate_anc_signals()
    sr = 16000

    in_snr = compute_snr_db(tone, noisy)
    enhanced = nlms.enhance(noisy, reference=ref, sr=sr, order=32, mu=0.5)
    out_snr = compute_snr_db(tone, enhanced)

    assert out_snr - in_snr > 10.0, f"NLMS SNR gain was {out_snr - in_snr:.2f} dB, expected > 10 dB"

    err = enhanced - tone
    first_sec_power = float(np.mean(err[:sr] ** 2))
    last_sec_power = float(np.mean(err[-sr:] ** 2))
    ratio = last_sec_power / first_sec_power

    assert (
        ratio < 0.10
    ), f"Last-second error power ({last_sec_power:.2e}) >= 10% of first second ({first_sec_power:.2e})"


def test_lms_snr_gain() -> None:
    """Verify LMS with order 32 and mu 0.005 achieves > 5 dB SNR gain."""
    tone, noisy, ref = generate_anc_signals()

    in_snr = compute_snr_db(tone, noisy)
    enhanced = lms.enhance(noisy, reference=ref, sr=16000, order=32, mu=0.005)
    out_snr = compute_snr_db(tone, enhanced)

    assert out_snr - in_snr > 5.0, f"LMS SNR gain was {out_snr - in_snr:.2f} dB, expected > 5 dB"


def test_nlms_scaled_gain() -> None:
    """Verify NLMS still achieves > 10 dB SNR gain when signals are scaled x10."""
    tone, noisy, ref = generate_anc_signals()

    scaled_noisy = (noisy * 10.0).astype(np.float32)
    scaled_ref = (ref * 10.0).astype(np.float32)
    scaled_tone = (tone * 10.0).astype(np.float32)

    in_snr = compute_snr_db(scaled_tone, scaled_noisy)
    enhanced = nlms.enhance(scaled_noisy, reference=scaled_ref, sr=16000, order=32, mu=0.5)
    out_snr = compute_snr_db(scaled_tone, enhanced)

    assert (
        out_snr - in_snr > 10.0
    ), f"Scaled NLMS SNR gain was {out_snr - in_snr:.2f} dB, expected > 10 dB"


@pytest.mark.parametrize("module", [nlms, lms])
def test_reference_none_raises_value_error(module) -> None:
    """Verify enhance raises ValueError when reference is None or length mismatches."""
    noisy = np.zeros(1600, dtype=np.float32)
    with pytest.raises(ValueError, match="Reference signal must be provided"):
        module.enhance(noisy, reference=None)

    ref_mismatch = np.zeros(800, dtype=np.float32)
    with pytest.raises(ValueError, match="Reference length"):
        module.enhance(noisy, reference=ref_mismatch)


def test_nlms_invalid_mu_raises_value_error() -> None:
    """Verify NLMS raises ValueError when mu is outside (0, 2)."""
    noisy = np.zeros(160, dtype=np.float32)
    ref = np.zeros(160, dtype=np.float32)
    with pytest.raises(ValueError, match="Step size mu must satisfy 0 < mu < 2"):
        nlms.enhance(noisy, reference=ref, mu=0.0)
    with pytest.raises(ValueError, match="Step size mu must satisfy 0 < mu < 2"):
        nlms.enhance(noisy, reference=ref, mu=2.0)


def test_lms_stability_warning() -> None:
    """Verify LMS issues a warning when mu >= 2 / (order * mean(ref**2))."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=1600).astype(np.float32)
    noisy = np.zeros(1600, dtype=np.float32)
    bound = 2.0 / (32 * float(np.mean(ref**2)))

    with pytest.warns(UserWarning, match="exceeds or equals stability bound"):
        lms.enhance(noisy, reference=ref, order=32, mu=bound + 0.01)


@pytest.mark.parametrize("module", [nlms, lms])
@pytest.mark.parametrize("length", [1000, 16000, 40000])
def test_output_dtype_and_length(module, length: int) -> None:
    """Verify enhance returns float32 array with identical length to noisy input."""
    rng = np.random.default_rng(42)
    noisy = rng.uniform(-0.5, 0.5, size=length).astype(np.float32)
    ref = rng.uniform(-0.5, 0.5, size=length).astype(np.float32)

    enhanced = module.enhance(noisy, reference=ref, sr=16000)
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == (length,)
