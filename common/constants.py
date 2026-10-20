"""Global constants fixed by DRDO-ANC AI Coding Contract v1.

These values are fixed by the contract and change only through a "contract:" Issue.
"""

# Core pipeline and STFT constants (Section A)
SAMPLE_RATE: int = 16000
N_FFT: int = 512
HOP_LENGTH: int = 256
WIN_LENGTH: int = 512
N_FREQ: int = 257  # N_FFT // 2 + 1
CLIP_SAMPLES: int = 64000  # 4 s
SEED: int = 42

# Adaptive filter defaults (Section A, Section C)
NLMS_TAPS: int = 4
NLMS_MU: float = 0.5
NLMS_EPS: float = 1e-6

# Real-time frame length, 16 ms (Section A)
FRAME_SAMPLES: int = HOP_LENGTH

# Real-time latency budget in milliseconds (Section C)
LATENCY_BUDGET_MS: int = 16

# Mixing peak normalisation threshold (Section C)
PEAK_NORM: float = 0.99

# Standard mixing SNR levels in dB (Section C)
SNR_LEVELS_DB: tuple[int, ...] = (-5, 0, 5, 10, 15)

# Frozen dataset splits (Section D)
SPLITS: tuple[str, ...] = ("train", "val", "test", "test_unseen_noise")

# Standard noise classification taxonomy (Section D)
NOISE_TYPES: tuple[str, ...] = (
    "gunshot",
    "explosion",
    "helicopter",
    "vehicle",
    "drone",
    "siren",
    "machinery",
    "other",
)

# Benchmark method identifiers (Section D)
METHODS: tuple[str, ...] = ("noisy", "spectral", "wiener", "lms", "nlms", "ai", "hybrid")

# Manifest CSV ordered columns (Section D)
MANIFEST_COLUMNS: tuple[str, ...] = (
    "id",
    "speaker_id",
    "clean_path",
    "noise_path",
    "reference_path",
    "noise_type",
    "snr_db",
    "split",
    "mixed_path",
)

# Results JSON keys (Section D)
RESULT_KEYS: tuple[str, ...] = (
    "name",
    "method",
    "snr",
    "si_snr",
    "ssnr",
    "stoi",
    "pesq",
    "latency_ms",
    "rtf",
    "commit",
    "date",
)
"""Project-wide audio and training constants."""

from typing import Final

SAMPLE_RATE: Final[int] = 16_000
N_FFT: Final[int] = 512
HOP_LENGTH: Final[int] = 256
WIN_LENGTH: Final[int] = 512
N_FREQ: Final[int] = N_FFT // 2 + 1  # 257
CLIP_SAMPLES: Final[int] = 64_000  # 4 s at 16 kHz
SEED: Final[int] = 42

# NLMS adaptive-filter defaults
NLMS_TAPS: Final[int] = 4
NLMS_MU: Final[float] = 0.5
NLMS_EPS: Final[float] = 1e-6

# SNR levels for mixing clean + noise
MIX_SNRS_DB: Final[tuple[int, ...]] = (-5, 0, 5, 10, 15)

# Frame / normalisation
FRAME_SAMPLES: Final[int] = 256
PEAK_NORM: Final[float] = 0.99
