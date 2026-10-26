"""Unit tests for Phase 2 Real-Time Audio Pipeline (P4).

Covers all 9 required verification categories from the Phase 2 specification:
1. Valid 256-sample float32 frames.
2. Invalid frame counts.
3. Incorrect output shape.
4. NaN or infinite output.
5. Latency calculation using monotonic timer.
6. RTF (Real-Time Factor) calculation and budget compliance.
7. Stream callback behavior, ear safety clipping, and buffer status flags.
8. Audio error handling and headless runner robustness.
9. Clean stream shutdown via stop_event, duration, and KeyboardInterrupt.

All tests operate strictly on synthetic NumPy arrays and mocked audio drivers;
no physical soundcard, microphone, headphones, or trained AI weights are required.
"""

from __future__ import annotations

import builtins
import sys
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_engine.inference.realtime.latency import LatencyResult, measure_latency
from ai_engine.inference.realtime.stream import (
    PassthroughStreamingEnhancer,
    StreamStatistics,
    list_audio_devices,
    print_audio_devices,
    run_stream,
)
from common.constants import FRAME_SAMPLES, SAMPLE_RATE
from common.interface import StreamingEnhancer

# ---------------------------------------------------------------------------
# Test Fixtures & Dummy Enhancers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sounddevice() -> MagicMock:
    """Mock sounddevice module in sys.modules to isolate tests from audio hardware."""
    mock_sd = MagicMock()
    mock_stream_instance = MagicMock()
    mock_stream_cls = MagicMock(return_value=mock_stream_instance)
    mock_stream_instance.__enter__.return_value = mock_stream_instance
    mock_stream_instance.__exit__.return_value = None
    mock_sd.Stream = mock_stream_cls
    mock_sd.query_devices = MagicMock(
        return_value=[
            {"name": "Mock Microphone", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Mock Headphones", "max_input_channels": 0, "max_output_channels": 2},
        ]
    )
    mock_sd.default = MagicMock()
    mock_sd.default.device = (0, 1)

    with patch.dict(sys.modules, {"sounddevice": mock_sd}):
        yield mock_sd


class DummyPassthroughEnhancer:
    """Dummy enhancer tracking reset and process counts."""

    def __init__(self) -> None:
        self.reset_count = 0
        self.process_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        self.process_count += 1
        return frame.copy()


class DummyDelayEnhancer:
    """Dummy enhancer injecting sleep delay to test latency timing."""

    def __init__(self, delay_s: float = 0.002) -> None:
        self.delay_s = delay_s
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        time.sleep(self.delay_s)
        return frame.copy()


class DummyShapeMismatchEnhancer:
    """Dummy enhancer returning an invalid output shape."""

    def __init__(self, output_samples: int = 128) -> None:
        self.output_samples = output_samples

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.zeros(self.output_samples, dtype=np.float32)


class DummyNanEnhancer:
    """Dummy enhancer returning NaN values."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        out = frame.copy()
        out[len(out) // 2] = float("nan")
        return out


class DummyInfEnhancer:
    """Dummy enhancer returning Inf values."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        out = frame.copy()
        out[len(out) // 2] = float("inf")
        return out


class DummyDtypeMismatchEnhancer:
    """Dummy enhancer returning float64 instead of float32."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        return frame.astype(np.float64)


class DummyExceptionEnhancer:
    """Dummy enhancer deliberately raising an unhandled exception."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        raise RuntimeError("Synthetic DSP processing failure")


# ---------------------------------------------------------------------------
# 1. Valid 256-sample float32 frames
# ---------------------------------------------------------------------------


class TestP4ValidFrames:
    """Verify processing of valid 256-sample float32 audio frames."""

    def test_valid_256_sample_frame_processing(self) -> None:
        """Verify enhancer accepts 256 float32 samples and returns valid frame."""
        enhancer = DummyPassthroughEnhancer()
        frame = np.ones(FRAME_SAMPLES, dtype=np.float32) * 0.4
        out = enhancer.process_frame(frame)

        assert out.shape == (FRAME_SAMPLES,)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.4)

    def test_passthrough_enhancer_preserves_waveform(self) -> None:
        """Verify PassthroughStreamingEnhancer satisfies protocol and returns distinct copy."""
        enhancer = PassthroughStreamingEnhancer()
        assert isinstance(enhancer, StreamingEnhancer)

        rng = np.random.default_rng(42)
        frame = rng.uniform(-0.8, 0.8, size=FRAME_SAMPLES).astype(np.float32)
        frame_copy = frame.copy()

        out = enhancer.process_frame(frame)
        assert out.shape == (FRAME_SAMPLES,)
        assert out.dtype == np.float32
        assert np.array_equal(out, frame_copy)

        # Mutating output must not modify original input
        out[0] = 999.0
        assert frame[0] != 999.0

    def test_passthrough_enhancer_reset_idempotent(self) -> None:
        """Verify reset() can be called safely without error."""
        enhancer = PassthroughStreamingEnhancer()
        enhancer.reset()
        enhancer.reset()


# ---------------------------------------------------------------------------
# 2. Invalid frame counts
# ---------------------------------------------------------------------------


class TestP4InvalidFrameCounts:
    """Verify validation of frame count parameters."""

    @pytest.mark.parametrize("invalid_frames", [0, -1, -50])
    def test_measure_latency_zero_or_negative_frames(self, invalid_frames: int) -> None:
        """Verify measure_latency raises ValueError for non-positive n_frames."""
        enhancer = DummyPassthroughEnhancer()
        with pytest.raises(ValueError, match="positive integer"):
            measure_latency(enhancer, n_frames=invalid_frames)

    @pytest.mark.parametrize("invalid_type", ["100", 10.5, None, True, [10]])
    def test_measure_latency_non_integer_frame_counts(self, invalid_type: Any) -> None:
        """Verify measure_latency raises TypeError for non-int n_frames."""
        enhancer = DummyPassthroughEnhancer()
        with pytest.raises(TypeError, match="must be an integer"):
            measure_latency(enhancer, n_frames=invalid_type)

    def test_measure_latency_negative_warmup(self) -> None:
        """Verify measure_latency raises ValueError for negative n_warmup."""
        enhancer = DummyPassthroughEnhancer()
        with pytest.raises(ValueError, match="non-negative"):
            measure_latency(enhancer, n_frames=10, n_warmup=-1)

    @pytest.mark.parametrize("invalid_warmup", ["5", 5.0, None, True])
    def test_measure_latency_non_integer_warmup(self, invalid_warmup: Any) -> None:
        """Verify measure_latency raises TypeError for non-int n_warmup."""
        enhancer = DummyPassthroughEnhancer()
        with pytest.raises(TypeError, match="must be an integer"):
            measure_latency(enhancer, n_frames=10, n_warmup=invalid_warmup)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Incorrect output shape
# ---------------------------------------------------------------------------


class TestP4IncorrectOutputShape:
    """Verify rejection of non-256-sample frames."""

    def test_measure_latency_rejects_truncated_shape(self) -> None:
        """Verify measure_latency raises ValueError when enhancer returns fewer samples."""
        enhancer = DummyShapeMismatchEnhancer(output_samples=128)
        with pytest.raises(ValueError, match="shape mismatch"):
            measure_latency(enhancer, n_frames=5, n_warmup=1)

    def test_measure_latency_rejects_oversized_shape(self) -> None:
        """Verify measure_latency raises ValueError when enhancer returns more samples."""
        enhancer = DummyShapeMismatchEnhancer(output_samples=512)
        with pytest.raises(ValueError, match="shape mismatch"):
            measure_latency(enhancer, n_frames=5, n_warmup=1)

    def test_stream_callback_silences_incorrect_shape(
        self, mock_sounddevice: MagicMock
    ) -> None:
        """Verify callback outputs silence and increments failure count on shape mismatch."""
        mock_stream_cls = mock_sounddevice.Stream
        enhancer = DummyShapeMismatchEnhancer(output_samples=128)
        stop_event = threading.Event()
        stop_event.set()

        stats = run_stream(enhancer=enhancer, stop_event=stop_event)
        callback = mock_stream_cls.call_args.kwargs["callback"]

        indata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)
        outdata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)

        callback(indata, outdata, FRAME_SAMPLES, None, None)

        assert stats.callback_failures == 1
        assert np.all(outdata == 0.0), "Expected muted silence on shape mismatch"


# ---------------------------------------------------------------------------
# 4. NaN or infinite output
# ---------------------------------------------------------------------------


class TestP4NanOrInfiniteOutput:
    """Verify detection and handling of non-finite values."""

    def test_measure_latency_rejects_nan(self) -> None:
        """Verify measure_latency raises ValueError on NaN output."""
        enhancer = DummyNanEnhancer()
        with pytest.raises(ValueError, match="non-finite"):
            measure_latency(enhancer, n_frames=5, n_warmup=1)

    def test_measure_latency_rejects_inf(self) -> None:
        """Verify measure_latency raises ValueError on Inf output."""
        enhancer = DummyInfEnhancer()
        with pytest.raises(ValueError, match="non-finite"):
            measure_latency(enhancer, n_frames=5, n_warmup=1)

    def test_stream_callback_silences_nan(self, mock_sounddevice: MagicMock) -> None:
        """Verify callback mutes buffer to silence when NaN produced."""
        mock_stream_cls = mock_sounddevice.Stream
        enhancer = DummyNanEnhancer()
        stop_event = threading.Event()
        stop_event.set()

        stats = run_stream(enhancer=enhancer, stop_event=stop_event)
        callback = mock_stream_cls.call_args.kwargs["callback"]

        indata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)
        outdata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)

        callback(indata, outdata, FRAME_SAMPLES, None, None)

        assert stats.callback_failures == 1
        assert np.all(outdata == 0.0)

    def test_stream_callback_silences_inf(self, mock_sounddevice: MagicMock) -> None:
        """Verify callback mutes buffer to silence when Inf produced."""
        mock_stream_cls = mock_sounddevice.Stream
        enhancer = DummyInfEnhancer()
        stop_event = threading.Event()
        stop_event.set()

        stats = run_stream(enhancer=enhancer, stop_event=stop_event)
        callback = mock_stream_cls.call_args.kwargs["callback"]

        indata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)
        outdata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)

        callback(indata, outdata, FRAME_SAMPLES, None, None)

        assert stats.callback_failures == 1
        assert np.all(outdata == 0.0)


# ---------------------------------------------------------------------------
# 5. Latency calculation
# ---------------------------------------------------------------------------


class TestP4LatencyCalculation:
    """Verify monotonic timing and per-frame latency reporting."""

    def test_latency_monotonic_timer_accuracy(self) -> None:
        """Verify measure_latency accurately captures injected sleep delay."""
        delay_s = 0.002  # 2 ms
        enhancer = DummyDelayEnhancer(delay_s=delay_s)
        res = measure_latency(enhancer, n_frames=10, n_warmup=2)

        assert isinstance(res, tuple)
        assert res.latency_ms >= 1.8, f"Expected >= 1.8 ms, got {res.latency_ms:.3f} ms"
        assert res.n_frames == 10
        assert res.within_budget is True

    def test_latency_result_properties_and_unpacking(self) -> None:
        """Verify LatencyResult supports both tuple unpacking and named attributes."""
        res = LatencyResult(latency_ms=1.5, rtf=0.09375, n_frames=50, within_budget=True)

        # 2-tuple unpacking compatibility
        lat, rtf = res
        assert lat == 1.5
        assert rtf == 0.09375

        # Named attributes
        assert res.latency_ms == 1.5
        assert res.rtf == 0.09375
        assert res.n_frames == 50
        assert res.within_budget is True

        # Summary output
        summary = res.summary()
        assert "Measured Frames        : 50" in summary
        assert "Avg Processing Latency : 1.500 ms" in summary
        assert "PASSED" in summary

    def test_warmup_frames_excluded_from_timing(self) -> None:
        """Verify warm-up frames run on enhancer but do not dilute n_frames count."""
        enhancer = DummyPassthroughEnhancer()
        res = measure_latency(enhancer, n_frames=15, n_warmup=5)

        assert enhancer.process_count == 20
        assert res.n_frames == 15
        assert enhancer.reset_count >= 1


# ---------------------------------------------------------------------------
# 6. RTF calculation
# ---------------------------------------------------------------------------


class TestP4RtfCalculation:
    """Verify Real-Time Factor (RTF) calculations based on 16 ms nominal budget."""

    def test_rtf_within_budget(self) -> None:
        """Verify fast passthrough processing yields RTF << 1.0 and within_budget=True."""
        enhancer = DummyPassthroughEnhancer()
        res = measure_latency(enhancer, n_frames=20, n_warmup=5)

        assert res.rtf < 0.5
        assert res.within_budget is True

    def test_rtf_exceeding_budget_reports_overrun(self) -> None:
        """Verify enhancer exceeding 16 ms budget flags within_budget=False."""
        # 18 ms delay exceeds 16 ms budget
        enhancer = DummyDelayEnhancer(delay_s=0.018)
        res = measure_latency(enhancer, n_frames=5, n_warmup=1)

        assert res.latency_ms >= 16.0
        assert res.rtf > 1.0
        assert res.within_budget is False

        summary = res.summary()
        assert "FAILED (Exceeds 16.0 ms budget)" in summary

    def test_rtf_formula_consistency(self) -> None:
        """Verify RTF is mathematically equal to latency_ms / frame_duration_ms."""
        enhancer = DummyPassthroughEnhancer()
        res = measure_latency(enhancer, n_frames=10, n_warmup=2)

        budget_ms = (FRAME_SAMPLES / SAMPLE_RATE) * 1000.0  # 16.0 ms
        expected_rtf = res.latency_ms / budget_ms
        assert abs(res.rtf - expected_rtf) < 1e-5


# ---------------------------------------------------------------------------
# 7. Stream callback behavior
# ---------------------------------------------------------------------------


class TestP4StreamCallbackBehavior:
    """Verify callback execution, ear safety clipping, and buffer status flags."""

    def test_callback_normal_execution(self, mock_sounddevice: MagicMock) -> None:
        """Verify normal callback execution processes frame and updates statistics."""
        mock_stream_cls = mock_sounddevice.Stream
        enhancer = DummyPassthroughEnhancer()
        stop_event = threading.Event()
        stop_event.set()

        stats = run_stream(enhancer=enhancer, stop_event=stop_event)
        callback = mock_stream_cls.call_args.kwargs["callback"]

        indata = np.full((FRAME_SAMPLES, 1), 0.25, dtype=np.float32)
        outdata = np.zeros((FRAME_SAMPLES, 1), dtype=np.float32)

        callback(indata, outdata, FRAME_SAMPLES, None, None)

        assert outdata.shape == (FRAME_SAMPLES, 1)
        assert outdata.dtype == np.float32
        assert np.allclose(outdata[:, 0], 0.25)
        assert stats.frames_processed == 1
        assert stats.callback_failures == 0
        assert stats.total_latency_ms > 0.0

    def test_callback_ear_safety_clipping(self, mock_sounddevice: MagicMock) -> None:
        """Verify audio amplitudes are strictly clamped to [-1.0, 1.0] to protect hearing."""
        mock_stream_cls = mock_sounddevice.Stream
        enhancer = DummyPassthroughEnhancer()
        stop_event = threading.Event()
        stop_event.set()

        run_stream(enhancer=enhancer, stop_event=stop_event)
        callback = mock_stream_cls.call_args.kwargs["callback"]

        # Loud peaks exceeding [-1.0, 1.0]
        indata = np.array([3.2, -4.5, 1.1, -1.9] * 64, dtype=np.float32).reshape(
            FRAME_SAMPLES, 1
        )
        outdata = np.zeros((FRAME_SAMPLES, 1), dtype=np.float32)

        callback(indata, outdata, FRAME_SAMPLES, None, None)

        assert np.all(outdata <= 1.0)
        assert np.all(outdata >= -1.0)
        assert outdata[0, 0] == 1.0
        assert outdata[1, 0] == -1.0

    def test_callback_status_underflow_overflow_tracked(
        self, mock_sounddevice: MagicMock
    ) -> None:
        """Verify driver status flags are counted in StreamStatistics."""
        mock_stream_cls = mock_sounddevice.Stream
        enhancer = DummyPassthroughEnhancer()
        stop_event = threading.Event()
        stop_event.set()

        stats = run_stream(enhancer=enhancer, stop_event=stop_event)
        callback = mock_stream_cls.call_args.kwargs["callback"]

        indata = np.zeros((FRAME_SAMPLES, 1), dtype=np.float32)
        outdata = np.zeros((FRAME_SAMPLES, 1), dtype=np.float32)

        status = MagicMock()
        status.input_overflow = True
        status.input_underflow = True
        status.output_overflow = True
        status.output_underflow = True

        callback(indata, outdata, FRAME_SAMPLES, None, status)

        assert stats.input_overflows == 1
        assert stats.input_underflows == 1
        assert stats.output_overflows == 1
        assert stats.output_underflows == 1

    def test_callback_silences_dtype_mismatch(
        self, mock_sounddevice: MagicMock
    ) -> None:
        """Verify enhancer returning non-float32 mutes output to silence."""
        mock_stream_cls = mock_sounddevice.Stream
        enhancer = DummyDtypeMismatchEnhancer()
        stop_event = threading.Event()
        stop_event.set()

        stats = run_stream(enhancer=enhancer, stop_event=stop_event)
        callback = mock_stream_cls.call_args.kwargs["callback"]

        indata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)
        outdata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)

        callback(indata, outdata, FRAME_SAMPLES, None, None)

        assert stats.callback_failures == 1
        assert np.all(outdata == 0.0)


# ---------------------------------------------------------------------------
# 8. Audio error handling
# ---------------------------------------------------------------------------


class TestP4AudioErrorHandling:
    """Verify resilient error handling in callback and device discovery."""

    def test_callback_handles_unhandled_exception(
        self, mock_sounddevice: MagicMock
    ) -> None:
        """Verify exception inside process_frame never crashes callback thread."""
        mock_stream_cls = mock_sounddevice.Stream
        enhancer = DummyExceptionEnhancer()
        stop_event = threading.Event()
        stop_event.set()

        stats = run_stream(enhancer=enhancer, stop_event=stop_event)
        callback = mock_stream_cls.call_args.kwargs["callback"]

        indata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)
        outdata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)

        # Must not raise an unhandled exception
        callback(indata, outdata, FRAME_SAMPLES, None, None)

        assert stats.callback_failures == 1
        assert np.all(outdata == 0.0)

    def test_stream_startup_fails_gracefully_without_sounddevice(self) -> None:
        """Verify informative RuntimeError when sounddevice is unavailable."""
        with patch.dict(sys.modules, {"sounddevice": None}):
            enhancer = DummyPassthroughEnhancer()
            with pytest.raises(
                RuntimeError,
                match="Real-time audio streaming requires 'sounddevice'",
            ):
                run_stream(enhancer=enhancer)

    def test_stream_startup_fails_gracefully_on_portaudio_oserror(self) -> None:
        """Verify informative RuntimeError when PortAudio C-library is missing."""
        orig_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sounddevice":
                raise OSError("PortAudio library not found")
            return orig_import(name, *args, **kwargs)

        with patch.dict(sys.modules), patch("builtins.__import__", side_effect=fake_import):
            sys.modules.pop("sounddevice", None)
            enhancer = DummyPassthroughEnhancer()
            with pytest.raises(
                RuntimeError,
                match="PortAudio library not found",
            ):
                run_stream(enhancer=enhancer)

    def test_device_enumeration_handles_headless_environment(self) -> None:
        """Verify list_audio_devices and print_audio_devices handle missing audio hardware."""
        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = RuntimeError("PortAudio host error")
        with patch.dict(sys.modules, {"sounddevice": mock_sd}):
            devices = list_audio_devices()
            assert devices == []

            # print_audio_devices must run cleanly without error
            print_audio_devices()


# ---------------------------------------------------------------------------
# 9. Clean stream shutdown
# ---------------------------------------------------------------------------


class TestP4CleanShutdown:
    """Verify clean stream termination via stop_event, duration, and KeyboardInterrupt."""

    def test_shutdown_via_stop_event(self, mock_sounddevice: MagicMock) -> None:
        """Verify stop_event terminates stream cleanly and returns diagnostics."""
        mock_stream_cls = mock_sounddevice.Stream
        mock_stream_instance = MagicMock()
        mock_stream_cls.return_value.__enter__.return_value = mock_stream_instance

        enhancer = DummyPassthroughEnhancer()
        stop_event = threading.Event()
        stop_event.set()

        stats = run_stream(enhancer=enhancer, stop_event=stop_event)

        assert isinstance(stats, StreamStatistics)
        assert stats.stream_duration_s >= 0.0
        assert enhancer.reset_count == 1

        mock_stream_cls.assert_called_once_with(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            device=(None, None),
            channels=1,
            dtype="float32",
            callback=mock_stream_cls.call_args.kwargs["callback"],
        )

    def test_shutdown_via_duration(self, mock_sounddevice: MagicMock) -> None:
        """Verify stream exits cleanly after duration elapsed."""
        enhancer = DummyPassthroughEnhancer()
        stats = run_stream(enhancer=enhancer, duration=0.01)

        assert isinstance(stats, StreamStatistics)
        assert stats.stream_duration_s >= 0.01

    def test_shutdown_via_keyboard_interrupt(
        self, mock_sounddevice: MagicMock
    ) -> None:
        """Verify KeyboardInterrupt is handled gracefully without unhandled tracebacks."""
        mock_sounddevice.Stream.return_value.__enter__.side_effect = KeyboardInterrupt

        enhancer = DummyPassthroughEnhancer()
        stats = run_stream(enhancer=enhancer)

        assert isinstance(stats, StreamStatistics)
        assert enhancer.reset_count == 1
