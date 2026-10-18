# Dual-Microphone Reference Signal Simulation

Conforms to DRDO-ANC AI Coding Contract v1 (Section D).

## Acoustic Formulation & Motivation
Dual-channel Active Noise Cancellation (ANC) architectures rely on an external or reference microphone placed in the acoustic field to detect the incoming background noise disturbance before or concurrent with its propagation to the primary (earpiece / communication) microphone.

In real-world tactical environments, the reference microphone receives:
1. A slightly time-advanced or delayed version of the noise wavefront relative to the primary sensor depending on geometric orientation.
2. Moderate multipath reflections and acoustic diffraction around the headset or vehicle structure.
3. Independent electronic and transducer sensor noise.
4. **Zero speech**: Clean speech must never leak into the reference sensor.

## Simulation Recipe
The function `simulate_reference(noise, rng)` in `ai_engine/data/reference_sim.py` implements this physical transfer model:

1. **Acoustic Delay**:
   - Sample range: `DELAY_MIN_SAMPLES = 0` to `DELAY_MAX_SAMPLES = 80` samples.
   - At $f_s = 16\,000\text{ Hz}$, this corresponds to $0$ to $5.0\text{ ms}$ (acoustic propagation path differences of up to $\approx 1.7\text{ m}$).
   - **Critical Design Constraint**: The maximum delay ($5\text{ ms}$) is kept strictly below the $32\text{ ms}$ STFT window ($N_{\text{FFT}} = 512$ samples). This guarantees that STFT-domain adaptive filters (such as NLMS) can model the cross-channel acoustic relationship without non-causal frame violations.

2. **Decaying Multipath FIR Filter**:
   - Filter length: `TAPS_MIN = 8` to `TAPS_MAX = 32` taps ($0.5$ to $2.0\text{ ms}$).
   - Impulse response envelope:
     $$h[n] = e^{-\gamma n} \cdot v[n], \quad v[n] \sim \mathcal{N}(0, 1), \quad \gamma \in [0.05, 0.2]$$
   - Normalized to unit energy ($\sum_{n} h[n]^2 = 1.0$) to maintain energy consistency.

3. **Sensor Noise Injection**:
   - Independent additive Gaussian sensor noise scaled between $20\text{ dB}$ and $30\text{ dB}$ below the reference noise power:
     $$\text{SNR}_{\text{sensor}} \sim \mathcal{U}(20.0, 30.0)\text{ dB}$$

4. **Speech Isolation**:
   - Clean speech arrays are strictly excluded from the reference simulation.

5. **Peak Ceiling Guard**:
   - The reference signal is rescaled by `PEAK_NORM / ref_peak` only when its maximum absolute amplitude exceeds `PEAK_NORM = 0.99`. This prevents digital clipping or distortion when written to 16-bit PCM WAV formats, which can otherwise occur due to constructive interference through the multi-tap FIR filter or additive sensor noise. When the unscaled reference peak is already $\le 0.99$, no rescaling is applied.

