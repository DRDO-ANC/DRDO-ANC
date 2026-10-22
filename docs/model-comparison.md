
# AI Model Comparison

Preliminary comparison of deep learning baseline architectures for active noise cancellation and speech enhancement.

## Architecture Comparison

| Model | Parameters | Measured CPU Latency | Input | Output | Causal |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **CRN** | 1,820,369 (~1.82M) | 1.45 ms / spectral frame; 2.27 ms / 256-sample waveform frame (STFT → CRN → iSTFT); 10.3 ms / 1-second clip | Complex spectrum `[B, 257, F]` (magnitude used) | Complex spectrum `[B, 257, F]` (magnitude mask applied) | Yes, at inference (`eval()` mode) |
| **DCCRN** | 1,719,714 (~1.72M) | ~3.2–4.3 ms / spectral frame; ~37.4 ms / 1-second clip (63 frames) | Complex spectrum `[B, 257, F]` (real & imag used) | Complex spectrum `[B, 257, F]` (polar complex ratio mask applied) | Yes, at inference (`eval()` mode) |

### Notes

- **Measurement environment:** CPU measurements from the development machine (x86_64, PyTorch CPU). These figures are not Raspberry Pi or embedded device benchmarks. The single-spectral-frame measurement does not include carried LSTM state; Phase 4 must remeasure latency using the actual streaming path.
- **CRN:** Convolutional Recurrent Network with a causal 2D convolutional encoder, 2-layer unidirectional LSTM bottleneck, and causal 2D transposed-convolutional decoder with skip connections. Employs a real-valued sigmoid magnitude mask applied to the noisy complex spectrum while preserving the noisy phase.
- **CRN causality:** Causal at inference (eval mode); BatchNorm statistics are pooled across time during training.
- **DCCRN:** Deep Complex Convolution Recurrent Network with causal complex 2D convolutional encoder, 2-layer unidirectional ComplexLSTM bottleneck, and causal complex 2D transposed-convolutional decoder with skip connections. Employs a polar complex ratio mask with bounded magnitude and phase shift applied to the noisy complex spectrum.
- **DCCRN causality:** Causal at inference (eval mode); ComplexBatchNorm statistics are pooled across time during training.

## How to Decide

Final model selection between CRN and DCCRN must not be based on architectural intuition or personal preference. It must be determined empirically under strictly standardized conditions:

1. **Frozen Dataset Evaluation:** Both models must be trained and evaluated on the exact same frozen dataset (`dataset-v1`), using identical speaker splits and noise conditions to ensure leakage-safe, fair comparisons.
2. **Objective Audio Metrics:** Model performance must be evaluated quantitatively using standard speech enhancement and active noise cancellation metrics:
   - **SI-SNR** (Scale-Invariant Signal-to-Noise Ratio improvement in dB)
   - **SNR** (Standard Signal-to-Noise Ratio improvement in dB)
   - **PESQ** (Perceptual Evaluation of Speech Quality, ITU-T P.862)
   - **STOI** (Short-Time Objective Intelligibility)
3. **Hardware & Latency Constraints:** Latency must be measured on the target deployment hardware (e.g., embedded / real-time streaming harness) with streaming state management enabled.
4. **Trade-off Analysis:** A model with marginal PESQ/STOI gains must be evaluated against its latency and compute footprint within the real-time frame budget (e.g., DCCRN complex convolution overhead vs. CRN real-valued processing). The final candidate must satisfy both audio enhancement targets and real-time execution bounds.

