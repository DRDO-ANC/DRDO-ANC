"""Unit tests for the STFT / iSTFT wrapper class."""

import pytest
import torch

from ai_engine.models.stft import STFT
from common.constants import CLIP_SAMPLES, HOP_LENGTH, N_FFT, N_FREQ, SEED


@pytest.fixture(autouse=True)
def set_torch_seed() -> None:
    """Ensure deterministic random seed across all tests."""
    torch.manual_seed(SEED)


def test_stft_output_shape() -> None:
    """Verify STFT forward output shape and complex64 dtype."""
    stft = STFT()
    batch_size = 2
    x = torch.randn(batch_size, CLIP_SAMPLES)
    spec = stft.forward(x)

    expected_frames = 1 + CLIP_SAMPLES // HOP_LENGTH  # 1 + 64000 // 256 = 251
    assert spec.shape == (batch_size, N_FREQ, expected_frames)
    assert spec.dtype == torch.complex64


@pytest.mark.parametrize("length", [CLIP_SAMPLES, 16000, 64001, 1000])
def test_stft_round_trip(length: int) -> None:
    """Verify that iSTFT reconstructs the input waveform within numerical tolerance."""
    stft = STFT()
    x = torch.randn(2, length)
    spec = stft.forward(x)
    recon = stft.inverse(spec, length=length)

    assert recon.shape == (2, length)
    assert recon.dtype == torch.float32
    max_abs_err = torch.max(torch.abs(x - recon)).item()
    assert max_abs_err < 1e-4


def test_stft_batch_consistency() -> None:
    """Verify that batched execution produces results identical to unbatched execution."""
    stft = STFT()
    x1 = torch.randn(1, 16000)
    x2 = torch.randn(1, 16000)
    x_batch = torch.cat([x1, x2], dim=0)

    spec_batch = stft.forward(x_batch)
    spec1 = stft.forward(x1)
    spec2 = stft.forward(x2)

    assert torch.allclose(spec_batch[0], spec1[0], atol=1e-6)
    assert torch.allclose(spec_batch[1], spec2[0], atol=1e-6)

    recon_batch = stft.inverse(spec_batch, length=16000)
    recon1 = stft.inverse(spec1, length=16000)
    recon2 = stft.inverse(spec2, length=16000)

    assert torch.allclose(recon_batch[0], recon1[0], atol=1e-6)
    assert torch.allclose(recon_batch[1], recon2[0], atol=1e-6)


def test_stft_zero_input() -> None:
    """Verify zero input produces exact zero output without NaNs."""
    stft = STFT()
    x = torch.zeros(1, 16000)
    spec = stft.forward(x)
    recon = stft.inverse(spec, length=16000)

    assert not torch.isnan(spec).any()
    assert not torch.isnan(recon).any()
    assert torch.all(recon == 0.0)


def test_stft_error_handling() -> None:
    """Verify ValueError is raised on invalid dimension, dtype, or signal length."""
    stft = STFT()

    # 1-D input [T]
    with pytest.raises(ValueError, match="2-D tensor"):
        stft.forward(torch.randn(16000))

    # Integer dtype
    with pytest.raises(ValueError, match="floating-point"):
        stft.forward(torch.randint(-100, 100, (1, 16000)))

    # T <= N_FFT // 2 (256)
    with pytest.raises(ValueError, match=r"N_FFT // 2"):
        stft.forward(torch.randn(1, N_FFT // 2))

    # Invalid inverse: real tensor
    with pytest.raises(ValueError, match="complex"):
        stft.inverse(torch.randn(1, N_FREQ, 10), length=16000)

    # Invalid inverse: wrong frequency dimension
    with pytest.raises(ValueError, match="frequency dimension"):
        stft.inverse(torch.zeros(1, N_FREQ + 1, 10, dtype=torch.complex64), length=16000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_stft_device_movement() -> None:
    """Verify window buffer follows the module when transferred to CUDA."""
    stft = STFT().cuda()
    assert stft.window.device.type == "cuda"

    x = torch.randn(1, 16000, device="cuda")
    spec = stft.forward(x)
    assert spec.device.type == "cuda"

    recon = stft.inverse(spec, length=16000)
    assert recon.device.type == "cuda"


def test_stft_gradient_flow() -> None:
    """Verify autograd backpropagation flows through forward and inverse passes."""
    stft = STFT()
    x = torch.randn(2, 4000, requires_grad=True)
    spec = stft.forward(x)
    recon = stft.inverse(spec, length=4000)

    loss = recon.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-3)
