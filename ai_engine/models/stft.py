"""Standardized STFT and iSTFT module wrapper for DRDO-ANC."""

import torch
import torch.nn as nn

from common.constants import HOP_LENGTH, N_FFT, N_FREQ, WIN_LENGTH


class STFT(nn.Module):
    """PyTorch module wrapper for STFT analysis and iSTFT synthesis."""

    def __init__(self) -> None:
        """Initialize the STFT module and register the analysis/synthesis window buffer."""
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN_LENGTH), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Compute the complex STFT representation of a batch of waveforms."""
        if not isinstance(wave, torch.Tensor):
            raise ValueError(f"wave must be a torch.Tensor, got {type(wave).__name__}")
        if wave.ndim != 2:
            raise ValueError(
                f"wave must be a 2-D tensor of shape [B, T], got shape {list(wave.shape)}. "
                "A 1-D tensor [T] must be unsqueezed to [1, T] by the caller."
            )
        if not wave.is_floating_point():
            raise ValueError(f"wave must have a floating-point dtype, got {wave.dtype}")
        min_length = N_FFT // 2 + 1
        if wave.shape[-1] <= N_FFT // 2:
            raise ValueError(
                f"wave length T ({wave.shape[-1]}) must be greater than N_FFT // 2 "
                f"({N_FFT // 2}) for reflect padding (minimum length {min_length})"
            )

        return torch.stft(
            wave,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Reconstruct time-domain waveforms from complex STFT spectra."""
        if not isinstance(spec, torch.Tensor):
            raise ValueError(f"spec must be a torch.Tensor, got {type(spec).__name__}")
        if not spec.is_complex():
            raise ValueError(f"spec must have a complex dtype, got {spec.dtype}")
        if spec.ndim != 3:
            raise ValueError(
                f"spec must be a 3-D tensor of shape [B, N_FREQ, F], got shape {list(spec.shape)}"
            )
        if spec.shape[1] != N_FREQ:
            raise ValueError(
                f"spec frequency dimension ({spec.shape[1]}) must match N_FREQ ({N_FREQ})"
            )

        return torch.istft(
            spec,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            length=length,
        )
