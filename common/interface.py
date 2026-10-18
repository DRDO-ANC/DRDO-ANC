"""Shared enhancer protocols and validation helpers for DRDO-ANC."""

from typing import Protocol, runtime_checkable
import numpy as np

from common.constants import FRAME_SAMPLES, SAMPLE_RATE


@runtime_checkable
class Enhancer(Protocol):
    """Protocol for offline speech enhancement methods."""

    def enhance(
        self,
        noisy: np.ndarray,
        reference: np.ndarray | None = None,
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Enhance a noisy audio signal."""
        ...


@runtime_checkable
class StreamingEnhancer(Protocol):
    """Protocol for causal real-time frame-by-frame speech enhancement."""

    def reset(self) -> None:
        """Reset internal filter and model states."""
        ...

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process one real-time audio frame."""
        ...


def check_audio(x: np.ndarray, name: str = "audio") -> None:
    """Validate that input audio is a finite 1-D float32 array in range [-1, 1]."""
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray, got {type(x).__name__}")
    if x.dtype != np.float32:
        raise ValueError(f"{name} must have dtype float32, got {x.dtype}")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    if len(x) > 0:
        min_val = float(np.min(x))
        max_val = float(np.max(x))
        if min_val < -1.0 or max_val > 1.0:
            raise ValueError(
                f"{name} values must be in [-1.0, 1.0], found range [{min_val}, {max_val}]"
            )


def check_enhanced(noisy: np.ndarray, enhanced: np.ndarray) -> None:
    """Validate that enhanced audio is a finite 1-D float32 array matching noisy length."""
    if not isinstance(enhanced, np.ndarray):
        raise ValueError(f"enhanced must be a numpy.ndarray, got {type(enhanced).__name__}")
    if enhanced.dtype != np.float32:
        raise ValueError(f"enhanced must have dtype float32, got {enhanced.dtype}")
    if enhanced.ndim != 1:
        raise ValueError(f"enhanced must be 1-dimensional, got shape {enhanced.shape}")
    if not np.all(np.isfinite(enhanced)):
        raise ValueError("enhanced contains non-finite values (NaN or Inf)")
    if len(enhanced) != len(noisy):
        raise ValueError(
            f"enhanced length ({len(enhanced)}) does not match noisy length ({len(noisy)})"
        )
