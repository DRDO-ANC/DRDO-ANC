"""Unit and smoke tests for DSP tuning and timing scripts.

These tests smoke-test the CLI entrypoints and provide unit tests for SNR gain,
RLS filter convergence, exponential decay detection, and table formatting.
Note: These are smoke and unit tests, not proof of tuning quality.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_script_module(name: str) -> Any:
    """Load a script file by absolute path using importlib without altering sys.path."""
    script_path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------


def test_tune_dsp_main_quick(capsys: pytest.CaptureFixture[str]) -> None:
    """Smoke-test tune_dsp.py main(--quick) returns 0 and prints status."""
    mod = _load_script_module("tune_dsp")
    ret = mod.main(["--quick", "--seed", "42"])
    assert ret == 0
    out = capsys.readouterr().out
    assert ("skipped:" in out) or ("Results:" in out)


def test_tune_dsp_rejects_unsupported_dataset_options() -> None:
    """Verify tune_dsp CLI rejects removed/unsupported --manifest and --data-root options."""
    mod = _load_script_module("tune_dsp")
    with pytest.raises(SystemExit):
        mod.main(["--manifest", "some_manifest.csv"])
    with pytest.raises(SystemExit):
        mod.main(["--data-root", "/some/path"])


def test_time_nlms_vs_rls_main_quick(capsys: pytest.CaptureFixture[str]) -> None:
    """Smoke-test time_nlms_vs_rls.py main(--quick) returns 0 and prints algorithm-only markdown table."""
    mod = _load_script_module("time_nlms_vs_rls")
    ret = mod.main(["--quick", "--seed", "42", "--seconds", "0.2"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Filter Length" in out
    assert "NLMS (algorithm-only)" in out
    assert "RLS (algorithm-only)" in out


def test_time_nlms_vs_rls_seconds_validation(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify time_nlms_vs_rls CLI rejects zero, negative, inf, and nan seconds."""
    mod = _load_script_module("time_nlms_vs_rls")

    for bad_val, label in [("0.0", "zero"), ("-1.5", "negative"), ("inf", "inf"), ("nan", "nan")]:
        with pytest.raises(SystemExit) as exc_info:
            mod.main(["--seconds", bad_val])
        assert exc_info.value.code != 0, f"Expected nonzero exit for --seconds {label}"
        err = capsys.readouterr().err
        assert "--seconds must be finite and strictly positive" in err, (
            f"Missing validation message for --seconds {label}, got: {err!r}"
        )

    # Valid positive value succeeds
    ret = mod.main(["--quick", "--seconds", "0.1"])
    assert ret == 0


def test_missing_module_handling(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that missing signal_processing modules return 0 and print 'skipped:'."""
    # Hide any signal_processing module
    monkeypatch.setitem(sys.modules, "signal_processing", None)
    monkeypatch.setitem(sys.modules, "signal_processing.spectral", None)
    monkeypatch.setitem(sys.modules, "signal_processing.wiener", None)
    monkeypatch.setitem(sys.modules, "signal_processing.nlms", None)
    monkeypatch.setitem(sys.modules, "signal_processing.lms", None)

    mod = _load_script_module("tune_dsp")
    ret = mod.main(["--quick"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "skipped: spectral not available on this branch" in out
    assert "skipped: wiener not available on this branch" in out
    assert "skipped: nlms not available on this branch" in out
    assert "skipped: lms not available on this branch" in out


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_snr_gain_helper() -> None:
    """Verify compute_snr_gain computes correct improvement in decibels."""
    mod = _load_script_module("tune_dsp")
    c = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
    noise = (np.cos(np.linspace(0, 50, 1000)) * 0.1).astype(np.float32)
    noisy = c + noise

    # Gain when output is identical to noisy should be 0.0 dB
    gain_id = mod.compute_snr_gain(c, noisy, noisy)
    assert abs(gain_id) < 1e-4

    # Gain when enhanced is cleaner
    half_noise = c + 0.5 * noise
    gain_half = mod.compute_snr_gain(c, noisy, half_noise)
    # Reducing noise power by 4x is ~6.02 dB gain
    assert abs(gain_half - 6.02) < 0.1


def test_rls_converges_noise_free_system() -> None:
    """Verify throwaway RLS converges to < 1e-6 error on a noise-free 4-tap FIR system."""
    mod = _load_script_module("time_nlms_vs_rls")
    rng = np.random.default_rng(42)
    n_samples = 10000
    x = rng.standard_normal(n_samples).astype(np.float32)
    true_w = np.array([0.5, -0.3, 0.2, -0.1], dtype=np.float32)
    d = np.convolve(x, true_w)[:n_samples].astype(np.float32)

    # Throwaway RLS filter
    err = mod.throwaway_rls_filter(d, x, order=4, lmbda=0.999, delta=1.0)
    final_error = float(np.abs(err[-1]))
    assert final_error < 1e-6, f"RLS error did not drop below 1e-6: got {final_error}"


def test_convergence_samples_synthetic_decay() -> None:
    """Verify compute_convergence_samples accurately locates -20 dB point on exponential decay within 5%."""
    mod = _load_script_module("time_nlms_vs_rls")
    sr = 16000
    tau = 2000.0  # decay constant in samples
    # e[n] = exp(-n / tau)
    # Moving average over window W: avg(n) = exp(-2n / tau) * avg(0)
    # -20 dB drop (factor of 0.01) occurs at n = tau * ln(10) ≈ 4605.17 samples
    n = np.arange(20000)
    error = np.exp(-n / tau).astype(np.float32)

    detected_n = mod.compute_convergence_samples(error, sr=sr, drop_db=20.0, win_ms=100.0)
    assert detected_n is not None
    expected_n = tau * np.log(10.0)
    rel_error = abs(detected_n - expected_n) / expected_n
    assert rel_error < 0.05, f"Relative error {rel_error:.4f} exceeds 5% (detected: {detected_n}, expected: {expected_n})"


def test_format_table() -> None:
    """Verify format_table generates well-formed markdown table."""
    mod = _load_script_module("time_nlms_vs_rls")
    headers = ["Col A", "Col B"]
    rows = [["Val 1", 10.5], ["Val 2", 20.0]]
    table = mod.format_table(headers, rows)
    lines = table.strip().split("\n")
    assert len(lines) == 4
    assert lines[0].startswith("| Col A")
    assert lines[1].startswith("|-")
    assert "Val 1" in lines[2]
    assert "20.0" in lines[3]


def test_deterministic_seed_consistency() -> None:
    """Verify that identical seeds produce identical benchmark outputs."""
    mod = _load_script_module("tune_dsp")
    c1 = mod.generate_clean_signal("tone", 1.0, sr=16000)
    c2 = mod.generate_clean_signal("tone", 1.0, sr=16000)
    assert np.array_equal(c1, c2)

    n1 = mod.generate_noise("coloured", 1000, seed=42)
    n2 = mod.generate_noise("coloured", 1000, seed=42)
    assert np.array_equal(n1, n2)

    # Different seed gives different noise
    n3 = mod.generate_noise("coloured", 1000, seed=43)
    assert not np.array_equal(n1, n3)
