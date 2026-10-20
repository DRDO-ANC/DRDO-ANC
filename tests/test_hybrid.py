"""Tests for hybrid AI + NLMS enhancement in signal_processing/hybrid.py.

Uses a dummy enhancer object with identity enhance_spec signature (never copied module).
All synthetic tests use reproducible RNG seeded with SEED from common.constants.
"""

import numpy as np
import pytest
import torch

from ai_engine.models.stft import STFT
from common.constants import CLIP_SAMPLES, SAMPLE_RATE, SEED
from signal_processing.hybrid import enhance as hybrid_enhance
from signal_processing.nlms import enhance as nlms_enhance


class DummyIdentityAIEnhancer:
    """Dummy AI enhancer implementing enhance_spec with identity STFT transformation.

    Computes standard STFT of noisy audio and returns the complex spectrum untouched.
    """

    def __init__(self) -> None:
        self.stft = STFT()

    def enhance_spec(self, noisy: np.ndarray) -> np.ndarray:
        wave = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            spec = self.stft.forward(wave)
        return spec.squeeze(0).cpu().numpy().astype(np.complex64)


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixture providing reproducible PRNG."""
    return np.random.default_rng(SEED)


@pytest.fixture
def dummy_enhancer() -> DummyIdentityAIEnhancer:
    """Fixture providing a dummy identity AI enhancer."""
    return DummyIdentityAIEnhancer()


@pytest.fixture
def synthetic_audio(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Fixture providing synthetic noisy and reference signals."""
    t = np.linspace(0, CLIP_SAMPLES / SAMPLE_RATE, CLIP_SAMPLES, endpoint=False)
    clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = rng.normal(0, 0.3, CLIP_SAMPLES).astype(np.float32)
    ref = rng.normal(0, 0.3, CLIP_SAMPLES).astype(np.float32)
    noisy = clean + noise
    return noisy, ref


def test_hybrid_enhance_output_properties(
    dummy_enhancer: DummyIdentityAIEnhancer,
    synthetic_audio: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify hybrid enhancement output dtype, shape, and finiteness."""
    noisy, ref = synthetic_audio

    out = hybrid_enhance(noisy, reference=ref, ai_enhancer=dummy_enhancer)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == noisy.shape
    assert np.all(np.isfinite(out))


def test_hybrid_equals_nlms_when_dummy_is_identity(
    dummy_enhancer: DummyIdentityAIEnhancer,
    synthetic_audio: tuple[np.ndarray, np.ndarray],
) -> None:
    """When the dummy AI enhancer is identity STFT, hybrid output equals NLMS-only."""
    noisy, ref = synthetic_audio

    out_hybrid = hybrid_enhance(noisy, reference=ref, ai_enhancer=dummy_enhancer)
    out_nlms = nlms_enhance(noisy, reference=ref)

    # Since DummyIdentityAIEnhancer passes STFT(noisy) directly to NLMSFilter,
    # the output must be identical to NLMS-only enhancement.
    np.testing.assert_allclose(out_hybrid, out_nlms, atol=1e-5)


def test_hybrid_reference_none_returns_ai_output_without_nlms(
    dummy_enhancer: DummyIdentityAIEnhancer,
    synthetic_audio: tuple[np.ndarray, np.ndarray],
) -> None:
    """When reference is None, returns the AI reconstructed output without NLMS."""
    noisy, _ = synthetic_audio

    out_hybrid = hybrid_enhance(noisy, reference=None, ai_enhancer=dummy_enhancer)

    # With dummy identity enhancer, STFT followed directly by iSTFT reconstructs noisy
    np.testing.assert_allclose(out_hybrid, noisy, atol=1e-4)


def test_hybrid_raises_on_missing_enhancer(
    synthetic_audio: tuple[np.ndarray, np.ndarray],
) -> None:
    """When no enhancer is provided and AIEnhancer cannot be imported, raise RuntimeError."""
    noisy, ref = synthetic_audio

    with pytest.raises(RuntimeError, match="AIEnhancer is not available"):
        hybrid_enhance(noisy, reference=ref, ai_enhancer=None)
