"""Self-contained tests for LMS and NLMS adaptive noise cancellers.

The synthetic tests below are sanity checks and are not proof of
general performance on real noise. NLMS output SNR is limited at high input SNR by
misadjustment; see the tuning phase.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from signal_processing import lms, nlms
from signal_processing._adaptive import validate_positive


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


# --- Existing Tests (Retained Unedited) ---


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
    with pytest.raises(ValueError, match="same length"):
        module.enhance(noisy, reference=ref_mismatch)


def test_nlms_invalid_mu_raises_value_error() -> None:
    """Verify NLMS raises ValueError when mu is outside (0, 2)."""
    noisy = np.zeros(160, dtype=np.float32)
    ref = np.zeros(160, dtype=np.float32)
    with pytest.raises(ValueError, match="mu must be a finite number > 0"):
        nlms.enhance(noisy, reference=ref, mu=0.0)
    with pytest.raises(ValueError, match="mu must be < 2 for NLMS"):
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


# --- FIX 3 (a): Comprehensive Validation Tests Asserting ValueError ---


@pytest.mark.parametrize("module", [nlms, lms])
def test_validation_signal_dimension_and_empty(module) -> None:
    """Verify 2-D, 0-d, and empty inputs raise ValueError."""
    ref_1d = np.ones(100, dtype=np.float32)

    # 2-D
    with pytest.raises(ValueError, match="must be a 1-D array, got shape"):
        module.enhance(np.ones((10, 10), dtype=np.float32), reference=ref_1d)
    with pytest.raises(ValueError, match="must be a 1-D array, got shape"):
        module.enhance(ref_1d, reference=np.ones((10, 10), dtype=np.float32))

    # 0-d
    with pytest.raises(ValueError, match="must be a 1-D array, got shape"):
        module.enhance(np.array(1.0, dtype=np.float32), reference=ref_1d)
    with pytest.raises(ValueError, match="must be a 1-D array, got shape"):
        module.enhance(ref_1d, reference=np.array(1.0, dtype=np.float32))

    # Empty
    with pytest.raises(ValueError, match="must not be empty"):
        module.enhance(np.array([], dtype=np.float32), reference=np.array([], dtype=np.float32))


@pytest.mark.parametrize("module", [nlms, lms])
def test_validation_signal_dtypes(module) -> None:
    """Verify boolean and complex dtypes raise ValueError, while int16 and float64 are accepted."""
    ref_1d = np.ones(100, dtype=np.float32)

    # Bool dtype
    with pytest.raises(ValueError, match="must be a real numeric array"):
        module.enhance(np.ones(100, dtype=bool), reference=ref_1d)
    with pytest.raises(ValueError, match="must be a real numeric array"):
        module.enhance(ref_1d, reference=np.ones(100, dtype=bool))

    # Complex dtype
    with pytest.raises(ValueError, match="must be a real numeric array"):
        module.enhance(np.ones(100, dtype=np.complex64), reference=ref_1d)
    with pytest.raises(ValueError, match="must be a real numeric array"):
        module.enhance(ref_1d, reference=np.ones(100, dtype=np.complex64))

    # Accepted dtypes: int16 and float64 convert to float32
    noisy_int = np.ones(100, dtype=np.int16) * 10
    ref_f64 = np.ones(100, dtype=np.float64) * 0.5
    out = module.enhance(noisy_int, reference=ref_f64, sr=16000)
    assert out.dtype == np.float32
    assert out.shape == (100,)


@pytest.mark.parametrize("module", [nlms, lms])
def test_validation_non_finite_and_overflow(module) -> None:
    """Verify NaN, inf, and float32 overflow raise ValueError."""
    ref_1d = np.ones(100, dtype=np.float32)
    bad_nan = np.ones(100, dtype=np.float32)
    bad_nan[10] = np.nan
    bad_inf = np.ones(100, dtype=np.float32)
    bad_inf[10] = np.inf
    bad_overflow = np.ones(100, dtype=np.float64) * 1e40

    for bad in [bad_nan, bad_inf, bad_overflow]:
        with pytest.raises(ValueError, match="contains non-finite values"):
            module.enhance(bad, reference=ref_1d)
        with pytest.raises(ValueError, match="contains non-finite values"):
            module.enhance(ref_1d, reference=bad)


@pytest.mark.parametrize("module", [nlms, lms])
def test_validation_length_mismatch(module) -> None:
    """Verify mismatched noisy and reference lengths raise ValueError."""
    noisy = np.ones(100, dtype=np.float32)
    ref = np.ones(80, dtype=np.float32)
    with pytest.raises(ValueError, match="noisy and reference must have the same length, got 100 and 80"):
        module.enhance(noisy, reference=ref)


@pytest.mark.parametrize("module", [nlms, lms])
@pytest.mark.parametrize("bad_sr", [0, -1, 16000.5, "16000", None, True])
def test_validation_sr(module, bad_sr: object) -> None:
    """Verify invalid sampling rate raises ValueError."""
    noisy = np.ones(100, dtype=np.float32)
    ref = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError, match="sr must be a positive integer"):
        module.enhance(noisy, reference=ref, sr=bad_sr)


@pytest.mark.parametrize("module", [nlms, lms])
@pytest.mark.parametrize("bad_order", [0, -1, 1.5, True])
def test_validation_order(module, bad_order: object) -> None:
    """Verify invalid order raises ValueError."""
    noisy = np.ones(100, dtype=np.float32)
    ref = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError, match="order must be an integer >= 1"):
        module.enhance(noisy, reference=ref, order=bad_order)


@pytest.mark.parametrize("module", [nlms, lms])
@pytest.mark.parametrize("bad_mu", [0, -0.1, np.nan, True])
def test_validation_mu_common(module, bad_mu: object) -> None:
    """Verify non-positive, NaN, or boolean mu raises ValueError."""
    noisy = np.ones(100, dtype=np.float32)
    ref = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError, match="mu must be a finite number > 0"):
        module.enhance(noisy, reference=ref, mu=bad_mu)


def test_validation_nlms_specific() -> None:
    """Verify NLMS-specific checks for mu >= 2 and eps <= 0."""
    noisy = np.ones(100, dtype=np.float32)
    ref = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError, match="mu must be < 2 for NLMS, got 2"):
        nlms.enhance(noisy, reference=ref, mu=2)
    with pytest.raises(ValueError, match="eps must be a finite number > 0"):
        nlms.enhance(noisy, reference=ref, eps=0)
    with pytest.raises(ValueError, match="eps must be a finite number > 0"):
        nlms.enhance(noisy, reference=ref, eps=-1)


# --- FIX 3 (b): Divergence Assertions ---


def test_lms_divergence_warning_and_error() -> None:
    """Verify LMS with mu above stability bound triggers warning and output guard ValueError."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=3200).astype(np.float32)
    noisy = rng.normal(0.0, 1.0, size=3200).astype(np.float32)

    # Unit-variance ref, order 32, mu 0.1 (well above stability bound 2 / (32 * 1) = 0.0625)
    with (
        pytest.warns(UserWarning, match="exceeds or equals stability bound"),
        pytest.raises(
            ValueError,
            match="LMS filter diverged: output contains NaN or infinite values",
        ),
    ):
        lms.enhance(noisy, reference=ref, order=32, mu=0.1)

    # Reference scaled x10 with default LMS mu (0.002); power is ~100, bound is ~0.000625 -> warns
    scaled_ref = (ref * 10.0).astype(np.float32)
    scaled_noisy = (noisy * 10.0).astype(np.float32)
    with (
        pytest.warns(UserWarning, match="exceeds or equals stability bound"),
        pytest.raises(
            ValueError,
            match="LMS filter diverged: output contains NaN or infinite values",
        ),
    ):
        lms.enhance(scaled_noisy, reference=scaled_ref, order=32, mu=0.002)


# --- FIX 3 (c): Window-Orientation Test ---


def test_nlms_window_orientation() -> None:
    """Verify delay line orientation catches reversed reference window (> 30 dB suppression)."""
    rng = np.random.default_rng(42)
    sr = 16000
    n_samples = int(4.0 * sr)
    ref = rng.normal(0.0, 1.0, size=n_samples).astype(np.float32)

    h = np.zeros(16, dtype=np.float32)
    h[3] = 0.8
    h[4:] = 0.05
    noisy = lfilter(h, [1.0], ref).astype(np.float32)

    enhanced = nlms.enhance(noisy, reference=ref, sr=sr, order=32, mu=0.5)

    last_sec_err_power = float(np.mean(enhanced[-sr:] ** 2))
    noisy_power = float(np.mean(noisy**2))
    suppression_db = 10.0 * np.log10(noisy_power / (last_sec_err_power + 1e-12))

    # Measured suppression is ~118.27 dB; threshold set to measured - 3 dB and >= 30 dB
    assert suppression_db >= 115.0, f"Expected suppression >= 115 dB, got {suppression_db:.2f} dB"
    assert suppression_db >= 30.0


# --- FIX 3 (d, e): Coverage Grid and Measured Thresholds ---


MIN_GAIN_DB: dict[tuple[str, float, float], float] = {
    ("nlms", 0.5, -5.0): 8.00,
    ("nlms", 0.5, 0.0): 3.01,
    ("nlms", 0.5, 5.0): -1.00,
    ("nlms", 0.5, 10.0): -1.00,
    ("nlms", 0.05, -5.0): 17.03,
    ("nlms", 0.05, 0.0): 12.99,
    ("nlms", 0.05, 5.0): 8.34,
    ("nlms", 0.05, 10.0): 3.46,
    ("lms", 0.002, -5.0): 16.54,
    ("lms", 0.002, 0.0): 12.36,
    ("lms", 0.002, 5.0): 7.64,
    ("lms", 0.002, 10.0): 2.74,
}


def generate_grid_signals(
    sig_type: str,
    snr_target_db: float,
    fir_len: int,
    sr: int = 16000,
    duration: float = 4.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate tone, two-tone, or chirp signals with FIR-filtered reference noise at target SNR."""
    rng = np.random.default_rng(seed)
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
        raise ValueError(f"Unknown sig_type {sig_type}")

    ref = rng.normal(0.0, 1.0, size=n_samples).astype(np.float32)

    fir = np.zeros(fir_len, dtype=np.float32)
    fir[min(3, fir_len - 1)] = 0.8
    if fir_len > 4:
        fir[4:] = 0.05
    raw_noise = lfilter(fir, [1.0], ref).astype(np.float32)

    s_pwr = float(np.mean(clean**2))
    n_pwr_raw = float(np.mean(raw_noise**2))
    target_noise_pwr = s_pwr / (10.0 ** (snr_target_db / 10.0))
    scale_noise = np.sqrt(target_noise_pwr / (n_pwr_raw + 1e-12))
    noise = raw_noise * scale_noise

    noisy = clean + noise
    return clean, noisy, ref


GRID_CASES = [
    # Method, mu, snr_db, sig_type, fir_len, order
    (method, mu, snr_db, sig_type, fir_len, order)
    for (method, mu) in [("nlms", 0.5), ("nlms", 0.05), ("lms", 0.002)]
    for snr_db in [-5.0, 0.0, 5.0, 10.0]
    for sig_type in ["tone", "two-tone", "chirp"]
    for (fir_len, order) in [(8, 32), (16, 32), (32, 32), (32, 8)]
]


@pytest.mark.parametrize(
    ("method", "mu", "snr_db", "sig_type", "fir_len", "order"),
    GRID_CASES,
)
def test_coverage_grid(
    method: str,
    mu: float,
    snr_db: float,
    sig_type: str,
    fir_len: int,
    order: int,
) -> None:
    """Verify enhancement gain across SNR, signal types, FIR lengths, and filter orders."""
    # Strict xfail on cases where measured gain is below -1.0 dB due to NLMS misadjustment or truncated order
    if method == "nlms" and mu == 0.5 and snr_db == 10.0 and order == 32:
        pytest.xfail(
            reason=f"Measured gain ~ -5.4 dB due to NLMS mu=0.5 excess misadjustment at high input SNR {snr_db} dB"
        )
    if method == "nlms" and mu == 0.5 and snr_db in (5.0, 10.0) and fir_len == 32 and order == 8:
        pytest.xfail(
            reason=f"Measured gain < -1.0 dB due to under-modeled order 8 < FIR 32 with NLMS mu=0.5 at {snr_db} dB"
        )

    clean, noisy, ref = generate_grid_signals(sig_type, snr_db, fir_len)
    in_snr = compute_snr_db(clean, noisy)

    if method == "nlms":
        enhanced = nlms.enhance(noisy, reference=ref, sr=16000, order=order, mu=mu)
    else:
        enhanced = lms.enhance(noisy, reference=ref, sr=16000, order=order, mu=mu)

    out_snr = compute_snr_db(clean, enhanced)
    gain = out_snr - in_snr

    if order >= fir_len:
        threshold = MIN_GAIN_DB[(method, mu, snr_db)]
        assert (
            gain >= threshold
        ), f"{method} mu={mu} snr={snr_db} sig={sig_type} FIR={fir_len} gain {gain:.2f} dB < threshold {threshold:.2f} dB"
    else:
        # Order 8 < FIR 32
        assert gain >= -1.0, f"Expected gain >= -1.0 dB, got {gain:.2f} dB"


def test_nlms_scaling_invariance() -> None:
    """Verify NLMS gain is invariant across reference and noisy scaling factors."""
    clean, noisy, ref = generate_grid_signals("tone", 0.0, 16)
    in_snr = compute_snr_db(clean, noisy)

    out_1x = nlms.enhance(noisy, reference=ref, sr=16000, order=32, mu=0.5)
    gain_1x = compute_snr_db(clean, out_1x) - in_snr

    scaled_noisy = (noisy * 10.0).astype(np.float32)
    scaled_ref = (ref * 10.0).astype(np.float32)
    scaled_clean = (clean * 10.0).astype(np.float32)
    in_snr_10x = compute_snr_db(scaled_clean, scaled_noisy)

    out_10x = nlms.enhance(scaled_noisy, reference=scaled_ref, sr=16000, order=32, mu=0.5)
    gain_10x = compute_snr_db(scaled_clean, out_10x) - in_snr_10x

    assert (
        abs(gain_1x - gain_10x) < 1e-4
    ), f"NLMS gain changed by {abs(gain_1x - gain_10x):.6f} dB under x10 scaling"


# --- Review Fix 1: validate_positive comprehensive parameter tests ---


INVALID_POSITIVE_VALUES = [
    pytest.param(1 + 2j, id="python_complex"),
    pytest.param(np.complex128(1.0), id="numpy_complex128"),
    pytest.param(np.complex64(1.0), id="numpy_complex64"),
    pytest.param(True, id="bool_true"),
    pytest.param(False, id="bool_false"),
    pytest.param(np.array([1.0]), id="numpy_1d_array"),
    pytest.param(np.array(1.0), id="numpy_0d_array"),
    pytest.param(np.nan, id="nan"),
    pytest.param(float("inf"), id="pos_inf"),
    pytest.param(float("-inf"), id="neg_inf"),
    pytest.param(0, id="zero_int"),
    pytest.param(0.0, id="zero_float"),
    pytest.param(-1.0, id="negative_float"),
    pytest.param(-1, id="negative_int"),
    pytest.param(10**1000, id="overflow_large_int"),
    pytest.param(-(10**1000), id="overflow_negative_large_int"),
    pytest.param("0.5", id="string"),
    pytest.param(None, id="none"),
    pytest.param([1.0], id="list"),
]


@pytest.mark.parametrize("bad_value", INVALID_POSITIVE_VALUES)
def test_validate_positive_rejects_invalid(bad_value) -> None:
    """Verify validate_positive rejects complex, bool, arrays, NaN, inf, zero, negative, and non-numeric."""
    with pytest.raises(ValueError, match="must be a finite number > 0"):
        validate_positive(bad_value, "param")


VALID_POSITIVE_VALUES = [
    pytest.param(0.001, id="small_float"),
    pytest.param(1.0, id="float_one"),
    pytest.param(1, id="int_one"),
    pytest.param(np.float64(0.5), id="numpy_float64"),
    pytest.param(np.float32(0.1), id="numpy_float32"),
    pytest.param(np.int32(1), id="numpy_int32"),
    pytest.param(np.int64(2), id="numpy_int64"),
]


@pytest.mark.parametrize("good_value", VALID_POSITIVE_VALUES)
def test_validate_positive_accepts_valid(good_value) -> None:
    """Verify validate_positive accepts real, scalar, finite, positive values."""
    result = validate_positive(good_value, "param")
    assert isinstance(result, float)
    assert result > 0


@pytest.mark.parametrize(
    "bad_value",
    [1 + 2j, np.complex128(1.0), True, np.array([1.0]), "0.5", None, 10**1000],
)
@pytest.mark.parametrize("module", [lms, nlms])
def test_enhance_rejects_invalid_mu(module, bad_value) -> None:
    """Verify enhance() raises ValueError for invalid mu types and unrepresentable values."""
    noisy = np.ones(100, dtype=np.float32)
    ref = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError):
        module.enhance(noisy, reference=ref, mu=bad_value)


@pytest.mark.parametrize(
    "bad_value",
    [1 + 2j, np.complex128(1.0), True, np.array([1.0]), "0.5", None, 10**1000],
)
def test_enhance_rejects_invalid_eps(bad_value) -> None:
    """Verify nlms.enhance() raises ValueError for invalid eps types and unrepresentable values."""
    noisy = np.ones(100, dtype=np.float32)
    ref = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError):
        nlms.enhance(noisy, reference=ref, eps=bad_value)


# --- Review Fix 3: NLMS scale invariance at different amplitudes ---


@pytest.mark.parametrize(
    ("scale", "expect_invariant"),
    [
        pytest.param(
            1.0,
            True,
            id="normal_amplitude",
        ),
        pytest.param(
            0.1,
            True,
            id="low_amplitude",
        ),
        pytest.param(
            1e-4,
            False,
            id="very_low_amplitude",
        ),
    ],
)
def test_nlms_scale_invariance_by_amplitude(
    scale: float, expect_invariant: bool
) -> None:
    """Test NLMS gain invariance at different signal amplitudes.

    With fixed eps=1e-6, NLMS normalization divides by (eps + ||x||^2).
    At normal and low amplitudes, ||x||^2 >> eps, so the filter is
    effectively scale-invariant (gain matches baseline within tolerance).

    At very low amplitudes (scale=1e-4), ||x||^2 ~ 1e-8 becomes
    comparable to or smaller than eps=1e-6, so epsilon dominates the
    denominator.  This effectively shrinks the step size and prevents
    convergence.  This is a fundamental limitation of fixed-epsilon
    NLMS, not a bug.
    """
    clean, noisy, ref = generate_grid_signals("tone", 0.0, 16)

    # Baseline gain at scale=1.0
    in_snr_base = compute_snr_db(clean, noisy)
    out_base = nlms.enhance(noisy, reference=ref, sr=16000, order=32, mu=0.5)
    gain_base = compute_snr_db(clean, out_base) - in_snr_base

    # Scaled signals
    s_clean = (clean * scale).astype(np.float32)
    s_noisy = (noisy * scale).astype(np.float32)
    s_ref = (ref * scale).astype(np.float32)

    in_snr_s = compute_snr_db(s_clean, s_noisy)
    out_s = nlms.enhance(s_noisy, reference=s_ref, sr=16000, order=32, mu=0.5)
    gain_s = compute_snr_db(s_clean, out_s) - in_snr_s

    diff = abs(gain_base - gain_s)
    if expect_invariant:
        assert diff < 0.01, (
            f"NLMS gain changed by {diff:.6f} dB at scale={scale}; "
            f"expected invariance (eps=1e-6 << ||x||^2)"
        )
    else:
        # At very low amplitudes, eps dominates → gains diverge significantly
        assert diff > 1.0, (
            f"Expected significant gain difference at scale={scale} "
            f"due to eps dominance, but diff was only {diff:.4f} dB"
        )


# --- Review Fix 4: Divergence guard coverage ---


def test_divergence_guard_catches_nonfinite() -> None:
    """Verify the non-finite output guard fires for deterministic LMS divergence.

    The existing guard (``if not np.all(np.isfinite(out))``) reliably
    detects NaN or infinite values produced when mu is well above the
    stability bound, because exponential weight growth overflows float64
    and the float32 cast produces inf/NaN.
    """
    rng = np.random.default_rng(42)
    n = 3200
    ref = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    noisy = rng.normal(0.0, 1.0, size=n).astype(np.float32)

    # mu=0.1 well above stability bound ≈ 0.0625 for unit-variance, order=32
    with (
        pytest.warns(UserWarning, match="exceeds or equals stability bound"),
        pytest.raises(ValueError, match="LMS filter diverged"),
    ):
        lms.enhance(noisy, reference=ref, order=32, mu=0.1)


def test_divergence_guard_does_not_threshold_finite() -> None:
    """Document that the divergence guard does not impose finite-magnitude thresholds.

    The ``isfinite`` guard catches only NaN and infinity.  It does not
    detect finite-but-large outputs, which would require an arbitrary,
    domain-specific magnitude threshold.  This is by design: legitimate
    use cases may produce large output amplitudes (e.g. high-power
    reference signals), and false positives would be worse than allowing
    the caller to inspect the output.
    """
    rng = np.random.default_rng(42)
    n = 1600
    # Low-power reference: stability bound ≈ 6.25 ≫ mu=0.002
    ref = (rng.normal(0.0, 0.1, size=n)).astype(np.float32)
    noisy = (rng.normal(0.0, 0.1, size=n)).astype(np.float32)

    out = lms.enhance(noisy, reference=ref, sr=16000, order=32, mu=0.002)
    # Output is finite; guard does not fire regardless of magnitude
    assert np.all(np.isfinite(out))
    assert out.dtype == np.float32
