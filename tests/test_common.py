"""Unit tests for common constants, interfaces, and audio check functions."""

import numpy as np
import pytest

from common.constants import (
    FRAME_SAMPLES,
    HOP_LENGTH,
    MANIFEST_COLUMNS,
    N_FFT,
    N_FREQ,
    RESULT_KEYS,
)
from common.interface import Enhancer, StreamingEnhancer, check_audio, check_enhanced


def test_constants_consistency() -> None:
    """Verify mathematical relations and lengths of contract constants."""
    assert N_FREQ == N_FFT // 2 + 1
    assert FRAME_SAMPLES == HOP_LENGTH
    assert len(MANIFEST_COLUMNS) == 9
    assert len(RESULT_KEYS) == 11


def test_check_audio_valid() -> None:
    """Verify check_audio accepts valid float32 mono audio within range [-1, 1]."""
    valid = np.zeros(1600, dtype=np.float32)
    check_audio(valid)

    valid_dynamic = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    check_audio(valid_dynamic)


def test_check_audio_invalid_type() -> None:
    """Verify check_audio rejects non-numpy or non-float32 arrays."""
    with pytest.raises(ValueError, match="numpy.ndarray"):
        check_audio([0.0, 0.5])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="float32"):
        check_audio(np.zeros(100, dtype=np.float64))


def test_check_audio_invalid_shape() -> None:
    """Verify check_audio rejects multi-dimensional arrays."""
    with pytest.raises(ValueError, match="1-dimensional"):
        check_audio(np.zeros((2, 100), dtype=np.float32))


def test_check_audio_non_finite() -> None:
    """Verify check_audio rejects NaN and Inf values."""
    with pytest.raises(ValueError, match="non-finite"):
        check_audio(np.array([0.0, np.nan, 0.5], dtype=np.float32))

    with pytest.raises(ValueError, match="non-finite"):
        check_audio(np.array([0.0, np.inf, 0.5], dtype=np.float32))


def test_check_audio_out_of_range() -> None:
    """Verify check_audio rejects values strictly outside [-1, 1]."""
    with pytest.raises(ValueError, match=r"\[-1.0, 1.0\]"):
        check_audio(np.array([1.01], dtype=np.float32))

    with pytest.raises(ValueError, match=r"\[-1.0, 1.0\]"):
        check_audio(np.array([-1.01], dtype=np.float32))


def test_check_enhanced() -> None:
    """Verify check_enhanced validations against noisy reference."""
    noisy = np.zeros(100, dtype=np.float32)
    enhanced = np.zeros(100, dtype=np.float32)
    check_enhanced(noisy, enhanced)

    # Wrong length
    with pytest.raises(ValueError, match="length"):
        check_enhanced(noisy, np.zeros(99, dtype=np.float32))

    # Wrong dtype
    with pytest.raises(ValueError, match="float32"):
        check_enhanced(noisy, np.zeros(100, dtype=np.float64))

    # Non-finite
    with pytest.raises(ValueError, match="non-finite"):
        nan_arr = np.zeros(100, dtype=np.float32)
        nan_arr[5] = np.nan
        check_enhanced(noisy, nan_arr)


def test_enhancer_protocol_runtime() -> None:
    """Verify runtime checkability of Enhancer protocol."""

    class DummyEnhancer:
        def enhance(
            self,
            noisy: np.ndarray,
            reference: np.ndarray | None = None,
            sr: int = 16000,
        ) -> np.ndarray:
            return noisy

    assert isinstance(DummyEnhancer(), Enhancer)

    class IncompleteEnhancer:
        pass

    assert not isinstance(IncompleteEnhancer(), Enhancer)


def test_streaming_enhancer_protocol_runtime() -> None:
    """Verify runtime checkability of StreamingEnhancer protocol."""

    class DummyStreaming:
        def reset(self) -> None:
            pass

        def process_frame(
            self,
            frame: np.ndarray,
            ref_frame: np.ndarray | None = None,
        ) -> np.ndarray:
            return frame

    assert isinstance(DummyStreaming(), StreamingEnhancer)

    class IncompleteStreaming:
        def reset(self) -> None:
            pass

    assert not isinstance(IncompleteStreaming(), StreamingEnhancer)
