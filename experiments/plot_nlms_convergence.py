"""Demonstration script for plotting NLMS adaptive filter error convergence.

Generates synthetic reference noise and primary microphone signals, runs
the STFT-domain NLMS filter, and outputs mean-squared error (MSE) per frame.
Does not commit any image to the repository.
"""

import sys
import tempfile
from pathlib import Path

# Ensure repository root is on sys.path when run as standalone script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from ai_engine.models.stft import STFT
from common.constants import N_FREQ, NLMS_EPS, NLMS_MU, NLMS_TAPS, SAMPLE_RATE, SEED
from signal_processing.nlms import NLMSFilter


def run_convergence_experiment() -> None:
    """Run synthetic NLMS adaptive filter convergence and save MSE plot."""
    rng = np.random.default_rng(SEED)
    duration_s = 4.0
    n_samples = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # 440 Hz tone + synthetic reference noise
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ref_noise = rng.normal(0, 0.4, n_samples).astype(np.float32)

    # Secondary acoustic path filter (8-tap FIR + 16 sample delay)
    fir = np.array([0.05, 0.1, 0.2, 0.35, 0.25, 0.15, 0.08, 0.04], dtype=np.float32)
    filtered = np.convolve(ref_noise, fir, mode="full")[:n_samples]
    delayed = np.pad(filtered, (16, 0))[:n_samples]
    noisy = tone + delayed

    stft_module = STFT()
    with torch.no_grad():
        D = stft_module.forward(torch.from_numpy(noisy).unsqueeze(0)).squeeze(0).numpy().astype(np.complex64)
        X = stft_module.forward(torch.from_numpy(ref_noise).unsqueeze(0)).squeeze(0).numpy().astype(np.complex64)

    nlms = NLMSFilter(n_freq=N_FREQ, taps=NLMS_TAPS, mu=NLMS_MU, eps=NLMS_EPS)
    E = nlms.process(X, D)

    # Calculate MSE per frame in dB
    mse_per_frame = np.mean(np.abs(E) ** 2, axis=0)
    mse_db = 10.0 * np.log10(mse_per_frame + 1e-12)

    output_dir = Path(tempfile.gettempdir())
    out_csv = output_dir / "nlms_convergence.csv"
    np.savetxt(out_csv, mse_db, delimiter=",", header="mse_db", comments="")
    print(f"Convergence MSE (dB) data saved to: {out_csv}")

    # Optional plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        out_img = output_dir / "nlms_convergence.png"
        plt.figure(figsize=(8, 4))
        plt.plot(mse_db, label="NLMS Error Power (dB)")
        plt.xlabel("STFT Frame Index")
        plt.ylabel("MSE (dB)")
        plt.title("STFT-Domain NLMS Filter Convergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_img)
        plt.close()
        print(f"Convergence plot saved to: {out_img}")
    except ImportError:
        pass


if __name__ == "__main__":
    run_convergence_experiment()
