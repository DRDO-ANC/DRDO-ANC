"""Tests for classical spectral subtraction and Wiener filtering enhancers.

NOTE: These synthetic tests are sanity checks and are not proof of general
performance on real noise. They use deterministic tones, chirps, and
amplitude-modulated signals with white/coloured Gaussian noise.  Thresholds
are set conservatively below measured gains; they guard against regressions
but do not guarantee real-world enhancement quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from signal_processing import spectral, wiener
from signal_processing._common import from_tf, noise_psd, to_tf, validate_waveform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_snr_db(clean: np.ndarray, noisy_or_enhanced: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in decibels."""
    err = noisy_or_enhanced - clean
    s_pwr = float(np.mean(clean**2))
    n_pwr = float(np.mean(err**2))
    return float(10.0 * np.log10(s_pwr / (n_pwr + 1e-12)))


def _make_signal(sig_type: str, t: np.ndarray) -> np.ndarray:
    """Generate a deterministic signal of the given type."""
    if sig_type == "tone_1k":
        return (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    elif sig_type == "two_tone":
        return (
            0.3 * np.sin(2.0 * np.pi * 500.0 * t) + 0.3 * np.sin(2.0 * np.pi * 1500.0 * t)
        ).astype(np.float32)
    elif sig_type == "chirp":
        f0, f1, dur = 200.0, 4000.0, float(t[-1])
        return (0.5 * np.sin(2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * dur) * t**2))).astype(
            np.float32
        )
    elif sig_type == "am_tone":
        env = (0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t))).astype(np.float32)
        return (0.5 * env * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    raise ValueError(f"Unknown signal type: {sig_type}")


def _make_noise(noise_type: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate white or coloured (low-pass filtered) noise."""
    w = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    if noise_type == "white":
        return w
    elif noise_type == "colored":
        b = np.array([0.1, 0.2, 0.3, 0.2, 0.1], dtype=np.float32)
        return lfilter(b, [1.0], w).astype(np.float32)
    raise ValueError(f"Unknown noise type: {noise_type}")


def generate_test_audio(
    sr: int = 16000,
    tone_duration: float = 3.0,
    lead_in: float = 0.5,
    tone_freq: float = 1000.0,
    snr_db: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 1 kHz tone with noise-only lead-in and 0 dB white noise."""
    rng = np.random.default_rng(seed)
    total_duration = lead_in + tone_duration
    total_samples = int(total_duration * sr)
    lead_in_samples = int(lead_in * sr)

    t = np.arange(total_samples, dtype=np.float32) / sr
    clean = np.zeros(total_samples, dtype=np.float32)
    clean[lead_in_samples:] = 0.5 * np.sin(2.0 * np.pi * tone_freq * t[lead_in_samples:])

    noise = rng.normal(0.0, 1.0, size=total_samples).astype(np.float32)
    s_pwr = np.mean(clean[lead_in_samples:] ** 2)
    n_pwr = np.mean(noise**2)
    scale = np.sqrt(s_pwr / (n_pwr * (10.0 ** (snr_db / 10.0))))
    noise = (noise * scale).astype(np.float32)

    noisy = clean + noise
    max_val = max(float(np.max(np.abs(noisy))), 1.0)
    clean = (clean / max_val).astype(np.float32)
    noisy = (noisy / max_val).astype(np.float32)
    return clean, noisy


def _generate_enhanced_case(
    sig_type: str,
    noise_type: str,
    target_snr: float,
    noise_mode: str,
    sr: int = 16000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (clean_scaled, noisy_scaled) for a parametrised test case."""
    rng = np.random.default_rng(seed)
    lead_in = 0.5 if noise_mode == "initial" else 0.0
    total_dur = 3.0 + lead_in
    n = int(total_dur * sr)
    lead_samples = int(lead_in * sr)
    t = np.arange(n, dtype=np.float32) / sr

    clean = np.zeros(n, dtype=np.float32)
    sig_t = t[lead_samples:] - t[lead_samples]
    clean[lead_samples:] = _make_signal(sig_type, sig_t)

    noise = _make_noise(noise_type, n, rng)
    s_pwr = float(np.mean(clean[lead_samples:] ** 2))
    n_pwr = float(np.mean(noise**2))
    if s_pwr > 0 and n_pwr > 0:
        scale = np.sqrt(s_pwr / (n_pwr * 10.0 ** (target_snr / 10.0)))
        noise = (noise * scale).astype(np.float32)
    noisy = clean + noise

    mx = max(1.0, float(np.max(np.abs(noisy))))
    return (clean / mx).astype(np.float32), (noisy / mx).astype(np.float32)


# ====================================================================
# Existing tests — kept exactly as they were
# ====================================================================

@pytest.mark.parametrize("module", [spectral, wiener])
def test_snr_improvement(module) -> None:
    """Verify that enhance achieves > 3 dB SNR improvement over input."""
    clean, noisy = generate_test_audio()
    in_snr = compute_snr_db(clean, noisy)

    enhanced = module.enhance(noisy, sr=16000)
    out_snr = compute_snr_db(clean, enhanced)

    assert out_snr > in_snr + 3.0, f"{module.__name__} SNR gain {out_snr - in_snr:.2f} dB <= 3 dB"


@pytest.mark.parametrize("module", [spectral, wiener])
@pytest.mark.parametrize("noise_mode", ["initial", "percentile", "min_stats"])
def test_all_noise_modes_run(module, noise_mode: str) -> None:
    """Verify that all noise modes run without error and return matching shape/dtype."""
    _clean, noisy = generate_test_audio()
    enhanced = module.enhance(noisy, sr=16000, noise_mode=noise_mode)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape


@pytest.mark.parametrize("module", [spectral, wiener])
@pytest.mark.parametrize("length", [1000, 16000, 40000])
def test_output_dtype_and_length(module, length: int) -> None:
    """Verify enhance returns float32 and exact matching length for various input sizes."""
    rng = np.random.default_rng(42)
    noisy = rng.uniform(-0.9, 0.9, size=length).astype(np.float32)

    enhanced = module.enhance(noisy, sr=16000)
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == (length,)


@pytest.mark.parametrize("module", [spectral, wiener])
def test_bad_noise_mode_raises_value_error(module) -> None:
    """Verify bad noise mode raises ValueError in enhance."""
    noisy = np.random.default_rng(42).uniform(-0.5, 0.5, size=1600).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown noise PSD estimation mode"):
        module.enhance(noisy, noise_mode="unsupported_mode")


def test_noise_psd_unknown_mode_raises_value_error() -> None:
    """Verify noise_psd directly raises ValueError on unknown mode."""
    p = np.ones((257, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown noise PSD estimation mode"):
        noise_psd(p, mode="invalid")


def test_wiener_decision_directed() -> None:
    """Verify Wiener enhancer with decision-directed approach improves SNR and preserves shape."""
    clean, noisy = generate_test_audio()
    in_snr = compute_snr_db(clean, noisy)

    enhanced = wiener.enhance(noisy, sr=16000, decision_directed=True)
    out_snr = compute_snr_db(clean, enhanced)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert out_snr > in_snr + 3.0


def test_stft_istft_roundtrip() -> None:
    """Verify to_tf and from_tf reconstruct the waveform with low error."""
    rng = np.random.default_rng(42)
    for length in (1000, 16000, 40000):
        x = rng.uniform(-0.8, 0.8, size=length).astype(np.float32)
        _, _, z = to_tf(x)
        rec = from_tf(z, length=length)
        assert rec.dtype == np.float32
        assert rec.shape == (length,)
        assert np.max(np.abs(x - rec)) < 1e-4


# ====================================================================
# FIX 3 — expanded SNR tests with multiple signals, noises, SNRs
# ====================================================================

# --- initial mode: clear improvement expected (measured 6+ dB) -----------

_INITIAL_CASES = [
    (sig, nt, snr)
    for sig in ("tone_1k", "two_tone", "chirp", "am_tone")
    for nt in ("white", "colored")
    for snr in (-5, 0, 5, 10)
]


@pytest.mark.parametrize("sig_type, noise_type, target_snr", _INITIAL_CASES)
@pytest.mark.parametrize("seed", [42, 43])
def test_initial_mode_spectral_gain(
    sig_type: str, noise_type: str, target_snr: int, seed: int
) -> None:
    """Spectral subtraction with initial mode should achieve > 5 dB gain."""
    clean, noisy = _generate_enhanced_case(sig_type, noise_type, target_snr, "initial", seed=seed)
    in_snr = compute_snr_db(clean, noisy)
    enhanced = spectral.enhance(noisy, sr=16000, noise_mode="initial")
    out_snr = compute_snr_db(clean, enhanced)
    assert out_snr > in_snr + 5.0, (
        f"spectral initial: {sig_type}/{noise_type}/{target_snr}dB seed={seed} "
        f"gain={out_snr - in_snr:.2f} dB"
    )


@pytest.mark.parametrize("sig_type, noise_type, target_snr", _INITIAL_CASES)
@pytest.mark.parametrize("seed", [42, 43])
def test_initial_mode_wiener_gain(
    sig_type: str, noise_type: str, target_snr: int, seed: int
) -> None:
    """Wiener filter with initial mode should achieve > 4 dB gain."""
    clean, noisy = _generate_enhanced_case(sig_type, noise_type, target_snr, "initial", seed=seed)
    in_snr = compute_snr_db(clean, noisy)
    enhanced = wiener.enhance(noisy, sr=16000, noise_mode="initial")
    out_snr = compute_snr_db(clean, enhanced)
    assert out_snr > in_snr + 4.0, (
        f"wiener initial: {sig_type}/{noise_type}/{target_snr}dB seed={seed} "
        f"gain={out_snr - in_snr:.2f} dB"
    )


@pytest.mark.parametrize("sig_type, noise_type, target_snr", _INITIAL_CASES)
@pytest.mark.parametrize("seed", [42, 43])
def test_initial_mode_wiener_dd_gain(
    sig_type: str, noise_type: str, target_snr: int, seed: int
) -> None:
    """Wiener decision-directed with initial mode should achieve > 8 dB gain."""
    clean, noisy = _generate_enhanced_case(sig_type, noise_type, target_snr, "initial", seed=seed)
    in_snr = compute_snr_db(clean, noisy)
    enhanced = wiener.enhance(noisy, sr=16000, noise_mode="initial", decision_directed=True)
    out_snr = compute_snr_db(clean, enhanced)
    assert out_snr > in_snr + 8.0, (
        f"wiener_dd initial: {sig_type}/{noise_type}/{target_snr}dB seed={seed} "
        f"gain={out_snr - in_snr:.2f} dB"
    )


# --- percentile mode: no noise-only lead-in, weaker guarantee -----------

_PERCENTILE_CASES = [
    (sig, nt, snr)
    for sig in ("tone_1k", "two_tone", "chirp", "am_tone")
    for nt in ("white", "colored")
    for snr in (-5, 0, 5, 10)
]


@pytest.mark.parametrize("module", [spectral, wiener])
@pytest.mark.parametrize("sig_type, noise_type, target_snr", _PERCENTILE_CASES)
def test_percentile_no_leadio_not_catastrophic(
    module, sig_type: str, noise_type: str, target_snr: int
) -> None:
    """Percentile mode without lead-in: output SNR not worse than input by > 12 dB."""
    clean, noisy = _generate_enhanced_case(
        sig_type, noise_type, target_snr, "percentile", seed=42
    )
    in_snr = compute_snr_db(clean, noisy)
    enhanced = module.enhance(noisy, sr=16000, noise_mode="percentile")
    out_snr = compute_snr_db(clean, enhanced)
    assert out_snr > in_snr - 12.0, (
        f"{module.__name__} percentile: {sig_type}/{noise_type}/{target_snr}dB "
        f"out_snr={out_snr:.2f} vs in_snr={in_snr:.2f} (degradation={in_snr - out_snr:.2f} dB)"
    )


# ====================================================================
# FIX 1 & FIX 2 — input validation tests
# ====================================================================

@pytest.mark.parametrize("module", [spectral, wiener])
def test_enhance_2d_input_raises(module) -> None:
    """2-D input must raise ValueError."""
    x = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        module.enhance(x)


@pytest.mark.parametrize("module", [spectral, wiener])
def test_enhance_0d_input_raises(module) -> None:
    """Scalar (0-D) input must raise ValueError."""
    x = np.float32(1.0)
    with pytest.raises(ValueError, match="1-D"):
        module.enhance(x)


@pytest.mark.parametrize("module", [spectral, wiener])
def test_enhance_empty_input_raises(module) -> None:
    """Empty array must raise ValueError."""
    x = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        module.enhance(x)


@pytest.mark.parametrize("module", [spectral, wiener])
def test_enhance_nan_input_raises(module) -> None:
    """Array containing NaN must raise ValueError."""
    x = np.array([0.1, np.nan, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        module.enhance(x)


@pytest.mark.parametrize("module", [spectral, wiener])
def test_enhance_inf_input_raises(module) -> None:
    """Array containing inf must raise ValueError."""
    x = np.array([0.1, np.inf, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        module.enhance(x)


# --- noise_psd validation tests ----------------------------------------

def test_noise_psd_zero_frames() -> None:
    """P with shape (257, 0) must raise ValueError."""
    p = np.ones((257, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="non-zero"):
        noise_psd(p)


def test_noise_psd_zero_bins() -> None:
    """P with shape (0, 10) must raise ValueError."""
    p = np.ones((0, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="non-zero"):
        noise_psd(p)


def test_noise_psd_1d_raises() -> None:
    """1-D input must raise ValueError."""
    p = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        noise_psd(p)


def test_noise_psd_nan_raises() -> None:
    """P containing NaN must raise ValueError."""
    p = np.ones((10, 5), dtype=np.float32)
    p[2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        noise_psd(p)


def test_noise_psd_negative_raises() -> None:
    """P containing negative values must raise ValueError."""
    p = np.ones((10, 5), dtype=np.float32)
    p[0, 0] = -0.1
    with pytest.raises(ValueError, match="negative"):
        noise_psd(p)


# --- n_frames parameter validation ------------------------------------

@pytest.mark.parametrize("bad_n_frames", [0, -1])
def test_noise_psd_n_frames_non_positive(bad_n_frames: int) -> None:
    """n_frames <= 0 must raise ValueError."""
    p = np.ones((10, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="n_frames"):
        noise_psd(p, n_frames=bad_n_frames)


def test_noise_psd_n_frames_float() -> None:
    """n_frames = 1.5 (not integer) must raise ValueError."""
    p = np.ones((10, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="n_frames"):
        noise_psd(p, n_frames=1.5)  # type: ignore[arg-type]


def test_noise_psd_n_frames_bool() -> None:
    """n_frames = True (bool subclass of int) must raise ValueError."""
    p = np.ones((10, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="n_frames"):
        noise_psd(p, n_frames=True)  # type: ignore[arg-type]


def test_noise_psd_n_frames_larger_than_frames() -> None:
    """n_frames larger than the frame count should be accepted (clamp)."""
    p = np.ones((10, 3), dtype=np.float32)
    result = noise_psd(p, n_frames=100)
    assert result.shape == (10, 1)


# --- percentile parameter validation -----------------------------------

@pytest.mark.parametrize("bad_pct", [-1.0, 101.0, float("nan")])
def test_noise_psd_bad_percentile(bad_pct: float) -> None:
    """percentile values outside [0, 100] or NaN must raise ValueError."""
    p = np.ones((10, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="percentile"):
        noise_psd(p, percentile=bad_pct)


def test_noise_psd_percentile_0_accepted() -> None:
    """percentile=0 is valid (minimum across frames per bin)."""
    p = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = noise_psd(p, mode="percentile", percentile=0.0)
    assert result.shape == (2, 1)


def test_noise_psd_percentile_100_accepted() -> None:
    """percentile=100 is valid (maximum across frames per bin)."""
    p = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = noise_psd(p, mode="percentile", percentile=100.0)
    assert result.shape == (2, 1)


# --- min_stats test: equals per-bin minimum over the whole clip ---------

def test_min_stats_equals_per_bin_minimum() -> None:
    """min_stats must return exactly the per-bin minimum power (no bias)."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0.1, 2.0, size=(20, 50)).astype(np.float32)
    result = noise_psd(p, mode="min_stats")
    expected = np.min(p, axis=1, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# --- validate_waveform standalone tests ----------------------------------

def test_validate_waveform_converts_to_float32() -> None:
    """validate_waveform should accept int arrays and convert to float32."""
    x = np.array([1, 2, 3], dtype=np.int16)
    result = validate_waveform(x)
    assert result.dtype == np.float32


def test_validate_waveform_rejects_string() -> None:
    """Non-numeric input must raise ValueError."""
    with pytest.raises(ValueError, match="numeric"):
        validate_waveform(np.array(["a", "b"]))
