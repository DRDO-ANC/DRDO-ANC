"""Hybrid AI-adaptive speech enhancement module.

Conforms to DRDO-ANC AI Coding Contract v1.
Pipeline:
  1. AI Model: Computes intermediate complex spectrum D = enhance_spec(noisy) [257, F].
  2. If reference is None: resynthesizes AI output via iSTFT without the adaptive stage.
  3. If reference is provided: transforms reference to STFT spectrum X,
     and applies NLMSFilter.process(X, D) -> E.
  4. iSTFT: Reconstructs time-domain audio from E.
"""

from typing import Any

import numpy as np
import torch

from ai_engine.models.stft import STFT
from common.constants import N_FFT, N_FREQ, NLMS_EPS, NLMS_MU, NLMS_TAPS, SAMPLE_RATE
from signal_processing.nlms import NLMSFilter


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    *,
    ai_enhancer: Any | None = None,
    taps: int = NLMS_TAPS,
    mu: float = NLMS_MU,
    eps: float = NLMS_EPS,
) -> np.ndarray:
    """Enhance noisy audio using a hybrid AI + STFT-domain NLMS filter cascade.

    Args:
        noisy: 1D NumPy array [T] of primary noisy audio samples.
        reference: Optional 1D NumPy array [T] of reference noise samples.
            Must match `len(noisy)` when provided.
        sr: Sampling rate in Hz (defaults to SAMPLE_RATE = 16000).
        ai_enhancer: AI model instance with an `enhance_spec(noisy: np.ndarray) -> np.ndarray`
            method returning complex spectrum of shape [N_FREQ, F]. If None, attempts
            to instantiate AIEnhancer from ai_engine.models.enhancer.
        taps: NLMS filter taps in STFT frames (default: NLMS_TAPS = 4).
        mu: NLMS step size (default: NLMS_MU = 0.5).
        eps: NLMS regularization epsilon (default: NLMS_EPS = 1e-6).

    Returns:
        np.ndarray: Enhanced 1D audio array [T] with dtype np.float32.
    """
    del sr
    noisy_arr = np.asarray(noisy, dtype=np.float32)
    t_len = len(noisy_arr)

    min_length = N_FFT // 2 + 1
    if t_len < min_length or np.all(noisy_arr == 0):
        return noisy_arr.copy()

    # Resolve AI enhancer
    if ai_enhancer is None:
        try:
            from ai_engine.models.enhancer import AIEnhancer  # type: ignore

            ai_enhancer = AIEnhancer()
        except ImportError as exc:
            raise RuntimeError(
                "AIEnhancer is not available in ai_engine.models.enhancer. "
                "Pass an ai_enhancer instance with an enhance_spec method."
            ) from exc

    # Step 1: AI enhancement in STFT domain -> complex spectrum [257, F]
    D = ai_enhancer.enhance_spec(noisy_arr).astype(np.complex64)
    if D.shape[0] != N_FREQ:
        raise ValueError(f"AI enhanced spectrum shape {D.shape} must have {N_FREQ} frequency bins")

    stft_module = STFT()

    # Step 2: If reference is None, return AI output without NLMS filtering
    if reference is None:
        d_tensor = torch.from_numpy(D).unsqueeze(0)
        with torch.no_grad():
            recon_tensor = stft_module.inverse(d_tensor, length=t_len)
        return recon_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    # Step 3: Reference is provided -> validate and run NLMS
    ref_arr = np.asarray(reference, dtype=np.float32)
    if len(ref_arr) != t_len:
        raise ValueError(f"Reference length ({len(ref_arr)}) must match noisy length ({t_len})")

    # Forward STFT on reference
    with torch.no_grad():
        x_tensor = stft_module.forward(torch.from_numpy(ref_arr).unsqueeze(0))
    X = x_tensor.squeeze(0).cpu().numpy().astype(np.complex64)

    # Adaptive filtering: NLMS takes X (reference) and D (AI output as desired)
    nlms = NLMSFilter(n_freq=N_FREQ, taps=taps, mu=mu, eps=eps)
    E = nlms.process(X, D)

    # Step 4: iSTFT synthesis
    e_tensor = torch.from_numpy(E).unsqueeze(0)
    with torch.no_grad():
        recon_tensor = stft_module.inverse(e_tensor, length=t_len)

    enhanced = recon_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(enhanced)):
        enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)

    return enhanced
