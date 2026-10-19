
# AI Model Comparison

Preliminary comparison of deep learning baseline architectures for active noise cancellation and speech enhancement.

## Architecture Comparison

| Model | Parameters | Measured CPU Latency | Input | Output | Causal |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **CRN** | 1,820,369 (~1.82M) | 1.45 ms / spectral frame; 2.27 ms / 256-sample waveform frame (STFT → CRN → iSTFT); 10.3 ms / 1-second clip | Complex spectrum `[B, 257, F]` (magnitude used) | Complex spectrum `[B, 257, F]` (magnitude mask applied) | Yes, at inference (`eval()` mode) |
| **DCCRN** | not measured | not measured | not measured | not measured | not measured |

### Notes

- **Measurement environment:** CPU measurements from the development machine. These figures are not Raspberry Pi benchmarks. The single-spectral-frame measurement does not include carried LSTM state; Phase 4 must remeasure latency using the actual streaming path.
- **CRN:** Convolutional Recurrent Network with a causal 2D convolutional encoder, 2-layer unidirectional LSTM bottleneck, and causal 2D transposed-convolutional decoder with skip connections. Employs a real-valued sigmoid magnitude mask applied to the noisy complex spectrum while preserving the noisy phase.
- **CRN causality:** Causal at inference (eval mode); BatchNorm statistics are pooled across time during training.
- **DCCRN:** Complex recurrent network operating on complex spectrogram representations (to be evaluated in a subsequent sprint).
