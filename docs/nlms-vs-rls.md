# NLMS vs RLS Real-Time Feasibility and Convergence Analysis

Analysis of computational cost, real-time feasibility against the 20 ms hop budget, convergence rates, and parameter tuning for classical DSP filters in DRDO-ANC.

## Purpose

The primary objective of this investigation is to evaluate whether Recursive Least Squares (RLS) offers sufficient convergence benefits over Normalized Least Mean Squares (NLMS) to justify its higher computational complexity in real-time Active Noise Cancellation (ANC). Furthermore, this document reports empirical parameter tuning results for single-microphone (Spectral Subtraction, Wiener filter) and two-microphone (NLMS, LMS) classical baseline enhancers.

## Cost Model ($O(N)$ vs $O(N^2)$)

Let $N$ denote the adaptive filter order (number of taps). At each audio sample $n$, the algorithm processes an input sample, computes an output, updates the error, and adjusts its internal states.

### NLMS Cost Model: $O(N)$

At each sample iteration:
1. **Delay line shift & insert:** $N-1$ copy operations.
2. **Filter output calculation:** $y[n] = \mathbf{w}^T[n] \mathbf{x}[n]$ requires $N$ multiplications and $N-1$ additions ($N$ MACs).
3. **Error computation:** $e[n] = d[n] - y[n]$ requires 1 subtraction.
4. **Energy normalization:** $\|\mathbf{x}[n]\|^2 + \epsilon$ requires $N$ multiplications and $N$ additions.
5. **Weight adaptation:** $\mathbf{w}[n+1] = \mathbf{w}[n] + \frac{\mu}{\|\mathbf{x}[n]\|^2 + \epsilon} e[n] \mathbf{x}[n]$ requires 1 division, $N + 1$ multiplications, and $N$ additions.

**Total operations per sample:** $\approx 3N + 2$ multiplications, $3N$ additions, 1 division $\implies \mathcal{O}(N)$ complexity.

### RLS Cost Model: $O(N^2)$

Let $\lambda$ be the forgetting factor and $\mathbf{P}[n]$ be the $N \times N$ inverse covariance matrix.
1. **Delay line update:** $N-1$ copy operations.
2. **Filter output calculation:** $y[n] = \mathbf{w}^T[n] \mathbf{x}[n]$ requires $N$ MACs ($N$ mults, $N$ adds).
3. **Error calculation:** $e[n] = d[n] - y[n]$ requires 1 subtraction.
4. **Intermediate vector calculation:** $\boldsymbol{\pi}[n] = \mathbf{P}[n-1] \mathbf{x}[n]$ requires a full matrix-vector product: $N^2$ multiplications and $N(N-1)$ additions.
5. **Gain vector computation:** $\mathbf{k}[n] = \frac{\boldsymbol{\pi}[n]}{\lambda + \mathbf{x}^T[n] \boldsymbol{\pi}[n]}$ requires $N$ multiplications, $N$ additions, 1 scalar addition, 1 division, and $N$ divisions/multiplications.
6. **Weight vector update:** $\mathbf{w}[n] = \mathbf{w}[n-1] + \mathbf{k}[n] e[n]$ requires $N$ multiplications and $N$ additions.
7. **Inverse covariance matrix update (Riccati equation):**
   $$\mathbf{P}[n] = \frac{1}{\lambda} \left( \mathbf{P}[n-1] - \mathbf{k}[n] \boldsymbol{\pi}^T[n] \right)$$
   Outer product $\mathbf{k}[n] \boldsymbol{\pi}^T[n]$ requires $N^2$ multiplications. Matrix subtraction and scalar scaling by $\lambda^{-1}$ require $N^2$ subtractions and $N^2$ multiplications.

**Total operations per sample:** $\approx 3N^2 + 4N$ multiplications, $2N^2 + 2N$ additions, 1 division $\implies \mathcal{O}(N^2)$ complexity.

As filter order $N$ scales from 32 to 256, RLS operations per sample scale by $(256/32)^2 = 64\times$, whereas NLMS scales by only $256/32 = 8\times$.

## Method

All experiments were executed with reproducible CLI commands using synthetic test signals generated deterministically with fixed random seeds.

### Hardware and Environment
- **Platform**: Windows-10-10.0.26300-SP0
- **Python**: 3.11.9
- **NumPy**: 2.4.6
- **CPU Count**: 12 logical cores
- **Date**: 2026-10-27
- **Implementation Source**: Benchmark run against official `signal_processing` implementations temporarily staged in `/tmp/p3check` from branches `origin/dsp/phase1-p2` and `origin/dsp/phase2-p2`.

### Reproduction Commands
```bash
# Timing and convergence comparison
python -m scripts.time_nlms_vs_rls --seed 42 --seconds 1.0

# Parameter sweep and tuning
python -m scripts.tune_dsp --seed 42
```

## Measured Results

Execution timing benchmarks were conducted over 1.0 s of 16 kHz audio (16,000 samples) across filter lengths $N \in \{32, 64, 128, 256\}$. Each measurement records the best of 3 timed runs following an initial 1,000-sample warmup run. The per-hop processing time is evaluated against the 20 ms hop budget (320 samples at 16 kHz):

| Filter Length | Method | RTF    | us / sample | ms / 20 ms hop | Budget Status |
|---------------|--------|--------|-------------|----------------|---------------|
| 32            | NLMS   | 0.0388 | 2.42        | 0.775          | WITHIN        |
| 32            | RLS    | 0.1255 | 7.85        | 2.510          | WITHIN        |
| 64            | NLMS   | 0.0397 | 2.48        | 0.793          | WITHIN        |
| 64            | RLS    | 0.1949 | 12.18       | 3.898          | WITHIN        |
| 128           | NLMS   | 0.0424 | 2.65        | 0.847          | WITHIN        |
| 128           | RLS    | 0.3852 | 24.08       | 7.704          | WITHIN        |
| 256           | NLMS   | 0.0438 | 2.74        | 0.875          | WITHIN        |
| 256           | RLS    | 1.4872 | 92.95       | 29.744         | OVER          |

### Budget Status Analysis
- NLMS remains well within budget across all filter orders, using at most **0.875 ms per 20 ms hop** (4.38% of available frame budget at $N=256$).
- RLS remains within budget up to $N=128$ (7.704 ms per hop), but at $N=256$ it surges to **29.744 ms per 20 ms hop** (RTF = 1.4872), decisively **OVER** budget by 148.7%.

## Convergence

Convergence was evaluated on a 2-second acoustic ANC setup (16-tap FIR primary path with 3-sample pure delay). The metric reports the number of elapsed audio samples required for the 100 ms moving-average error power to drop 20 dB below its starting level:

| Filter Length | NLMS ($\mu=0.5$)      | RLS ($\lambda=0.999$, $\delta=1.0$) |
|---------------|-----------------------|-------------------------------------|
| 32            | 187 samples (11.7 ms) | 64 samples (4.0 ms)                 |
| 64            | 314 samples (19.6 ms) | 100 samples (6.2 ms)                |

RLS achieves 20 dB attenuation in approximately one-third of the time required by NLMS (4.0 ms vs 11.7 ms at order 32, and 6.2 ms vs 19.6 ms at order 64).

## Recommendation

We propose to investigate RLS exclusively as an optional or hybrid adaptation mode under strict filter order constraints ($\le 64$), but **the project default must remain NLMS-only**.

### Supporting Evidence and Rationale:
1. **Latency and Budget Risk:** NLMS uses less than 0.88 ms per 20 ms hop across all orders ($N \le 256$), leaving >95% headroom for STFT analysis, neural enhancement stages, and I/O buffer management. At order 256, unvectorized Python RLS exceeds the frame budget (29.744 ms > 20 ms). Even at order 64, RLS consumes 3.90 ms per hop (5x more CPU than NLMS).
2. **Diminishing Convergence Benefit:** While RLS converges in 4.0 ms (versus 11.7 ms for NLMS at $N=32$), both filters achieve 20 dB error power attenuation well within 20 ms (a single hop). For continuous speech enhancement and stationary/slowly-varying acoustic noise paths, a 7.7 ms faster initial convergence does not justify a $3\times$ to $5\times$ computational overhead.
3. **Numerical Stability:** The standard RLS Riccati update is susceptible to loss of positive definiteness under finite-precision arithmetic unless square-root or QR-decomposition forms (e.g. QR-RLS) are adopted, adding further implementation complexity.

## Tuned Defaults

The following tables report empirical SNR gain across input SNRs ($-5, 0, 5, 10$ dB) evaluated on synthetic signals (tone, two-tone, chirp) and noise profiles (white, coloured) with seeds 42 and 43.

### 1. Spectral Subtraction (`spectral`)

Sweep grid: $\alpha \in [1.0, 1.5, 2.0, 3.0, 4.0]$, $\beta \in [0.01, 0.02, 0.05, 0.10]$, `noise_mode` $\in$ `[initial, percentile, min_stats]`.

| $\alpha$ | $\beta$ | Noise Mode | -5 dB (dB) | 0 dB (dB) | 5 dB (dB) | 10 dB (dB) | Mean Gain (dB) | Worst Gain (dB) |
|----------|---------|------------|------------|-----------|-----------|------------|----------------|-----------------|
| 1.0      | 0.01    | initial    | 4.31       | 4.28      | 4.22      | 4.16       | 4.24           | 3.53            |
| 1.0      | 0.02    | initial    | 4.15       | 4.12      | 4.06      | 4.00       | 4.08           | 3.41            |
| 2.0      | 0.02    | initial    | 7.61       | 7.58      | 7.45      | 7.29       | 7.48           | 5.95            |
| 3.0      | 0.02    | initial    | 9.88       | 9.96      | 9.82      | 9.54       | 9.80           | 7.46            |
| 4.0      | 0.01    | initial    | 11.56      | 11.92     | 11.82     | 11.41      | **11.68**      | **8.27**        |
| 4.0      | 0.02    | initial    | 10.99      | 11.30     | 11.22     | 10.86      | 11.09          | 8.09            |
| 4.0      | 0.01    | percentile | 1.95       | 1.96      | 1.96      | 1.94       | 1.95           | 1.72            |
| 4.0      | 0.01    | min_stats  | 0.07       | 0.07      | 0.07      | 0.07       | 0.07           | 0.06            |

- **Best Per Input SNR:**
  - -5 dB: $\alpha=4.0, \beta=0.01, \text{mode}=\text{initial}$ (Gain: 11.56 dB)
  - 0 dB: $\alpha=4.0, \beta=0.01, \text{mode}=\text{initial}$ (Gain: 11.92 dB)
  - 5 dB: $\alpha=4.0, \beta=0.01, \text{mode}=\text{initial}$ (Gain: 11.82 dB)
  - 10 dB: $\alpha=4.0, \beta=0.01, \text{mode}=\text{initial}$ (Gain: 11.41 dB)
- **Note:** The best setting is consistent between low (-5 dB) and high (10 dB) SNR.
- **Recommended Default:** $\alpha = 4.0, \beta = 0.01, \text{noise\_mode} = \text{"initial"}$ (Mean Gain: 11.68 dB, Worst-Case Gain: 8.27 dB).

### 2. Wiener Filter (`wiener`)

Sweep grid: $\xi_{\min} \in [-35, -30, -25, -20, -15]$ dB, `decision_directed` $\in [\text{False}, \text{True}]$, `noise_mode` $\in$ `[initial, percentile, min_stats]`.

| $\xi_{\min}$ (dB) | Noise Mode | Decision Directed | -5 dB (dB) | 0 dB (dB) | 5 dB (dB) | 10 dB (dB) | Mean Gain (dB) | Worst Gain (dB) |
|-------------------|------------|-------------------|------------|-----------|-----------|------------|----------------|-----------------|
| -35.0             | initial    | False             | 6.48       | 6.43      | 6.33      | 6.22       | 6.36           | 5.16            |
| -25.0             | initial    | False             | 6.48       | 6.42      | 6.33      | 6.21       | 6.36           | 5.16            |
| -15.0             | initial    | False             | 6.43       | 6.37      | 6.28      | 6.17       | 6.31           | 5.13            |
| -35.0             | initial    | True              | 14.49      | 14.97     | 14.47     | 13.60      | **14.38**      | **9.95**        |
| -25.0             | initial    | True              | 14.49      | 14.96     | 14.47     | 13.59      | 14.38          | 9.95            |
| -15.0             | initial    | True              | 14.38      | 14.86     | 14.37     | 13.52      | 14.28          | 9.93            |
| -35.0             | percentile | True              | 3.01       | 2.99      | 2.97      | 2.94       | 2.98           | 2.69            |
| -35.0             | min_stats  | True              | 0.09       | 0.09      | 0.09      | 0.09       | 0.09           | 0.08            |

- **Best Per Input SNR:**
  - -5 dB: $\xi_{\min} = -35.0\text{ dB}, \text{mode}=\text{initial}, \text{decision\_directed}=\text{True}$ (Gain: 14.49 dB)
  - 0 dB: $\xi_{\min} = -35.0\text{ dB}, \text{mode}=\text{initial}, \text{decision\_directed}=\text{True}$ (Gain: 14.97 dB)
  - 5 dB: $\xi_{\min} = -35.0\text{ dB}, \text{mode}=\text{initial}, \text{decision\_directed}=\text{True}$ (Gain: 14.47 dB)
  - 10 dB: $\xi_{\min} = -35.0\text{ dB}, \text{mode}=\text{initial}, \text{decision\_directed}=\text{True}$ (Gain: 13.60 dB)
- **Note:** The best setting is consistent between low (-5 dB) and high (10 dB) SNR. Decision-directed tracking provides a substantial boost of +8 dB over static a priori SNR estimation.
- **Recommended Default:** $\xi_{\min} = -35.0\text{ dB}, \text{noise\_mode} = \text{"initial"}, \text{decision\_directed} = \text{True}$ (Mean Gain: 14.38 dB, Worst-Case Gain: 9.95 dB).

### 3. NLMS Filter (`nlms`)

Sweep grid: $\text{order} \in [16, 32, 64, 128]$, $\mu \in [0.01, 0.05, 0.1, 0.5, 1.0]$.

| Order | $\mu$ | -5 dB (dB) | 0 dB (dB) | 5 dB (dB) | 10 dB (dB) | Mean Gain (dB) | Worst Gain (dB) |
|-------|-------|------------|-----------|-----------|------------|----------------|-----------------|
| 16    | 0.01  | 14.87      | 13.85     | 11.80     | 8.59       | **12.28**      | **5.50**        |
| 16    | 0.05  | 14.88      | 11.61     | 7.40      | 2.69       | 9.14           | -0.42           |
| 16    | 0.10  | 13.24      | 9.24      | 4.63      | -0.23      | 6.72           | -3.22           |
| 16    | 0.50  | 7.50       | 2.75      | -2.17     | -7.14      | 0.24           | -8.74           |
| 32    | 0.01  | 14.28      | 13.39     | 11.57     | 8.66       | 11.98          | 5.89            |
| 32    | 0.05  | 16.37      | 12.53     | 8.02      | 3.19       | 10.03          | 0.06            |
| 32    | 0.50  | 8.19       | 3.20      | -1.80     | -6.80      | 0.70           | -8.31           |
| 64    | 0.01  | 11.70      | 11.20     | 10.03     | 7.87       | 10.20          | 5.53            |
| 64    | 0.05  | 15.41      | 12.18     | 8.04      | 3.38       | 9.75           | 0.13            |
| 128   | 0.01  | 9.06       | 8.77      | 8.02      | 6.46       | 8.08           | 4.48            |
| 128   | 0.05  | 14.04      | 11.42     | 7.76      | 3.36       | 9.14           | -0.09           |

- **Best Per Input SNR:**
  - -5 dB: $\text{order}=32, \mu=0.05$ (Gain: 16.37 dB)
  - 0 dB: $\text{order}=16, \mu=0.01$ (Gain: 13.85 dB)
  - 5 dB: $\text{order}=16, \mu=0.01$ (Gain: 11.80 dB)
  - 10 dB: $\text{order}=32, \mu=0.01$ (Gain: 8.66 dB)
- **Note:** Best setting differs between low (-5 dB) and high (10 dB) SNR: $[\text{order}=32, \mu=0.05]$ at -5 dB versus $[\text{order}=32, \mu=0.01]$ at 10 dB. At high input SNR, larger $\mu$ causes noticeable misadjustment noise that degrades output SNR.
- **Recommended Default:** $\text{order} = 16, \mu = 0.01$ (Mean Gain: 12.28 dB, Worst-Case Gain: 5.50 dB).

### 4. LMS Filter (`lms`)

Sweep grid: $\text{order} \in [16, 32, 64]$, $\mu \in [0.0005, 0.001, 0.002, 0.005]$.

| Order | $\mu$  | -5 dB (dB) | 0 dB (dB) | 5 dB (dB) | 10 dB (dB) | Mean Gain (dB) | Worst Gain (dB) |
|-------|--------|------------|-----------|-----------|------------|----------------|-----------------|
| 16    | 0.0005 | 12.17      | 8.15      | 4.26      | 1.72       | 6.58           | 0.88            |
| 16    | 0.0010 | 14.13      | 10.58     | 6.48      | 3.05       | 8.56           | 1.68            |
| 16    | 0.0020 | 15.44      | 12.64     | 8.89      | 4.93       | 10.48          | 3.11            |
| 16    | 0.0050 | 15.66      | 14.17     | 11.53     | 7.88       | 12.31          | 6.11            |
| 32    | 0.0005 | 13.06      | 8.51      | 4.40      | 1.76       | 6.93           | 0.88            |
| 32    | 0.0010 | 15.34      | 11.13     | 6.70      | 3.12       | 9.08           | 1.70            |
| 32    | 0.0020 | 16.63      | 13.29     | 9.19      | 5.05       | 11.04          | 3.13            |
| 32    | 0.0050 | 15.80      | 14.27     | 11.61     | 7.94       | **12.40**      | **6.03**        |
| 64    | 0.0005 | 12.84      | 8.44      | 4.37      | 1.76       | 6.85           | 0.88            |
| 64    | 0.0010 | 14.71      | 10.87     | 6.61      | 3.10       | 8.82           | 1.69            |
| 64    | 0.0020 | 15.19      | 12.54     | 8.87      | 4.93       | 10.38          | 3.10            |
| 64    | 0.0050 | 13.27      | 12.39     | 10.47     | 7.39       | 10.88          | 5.61            |

- **Best Per Input SNR:**
  - -5 dB: $\text{order}=32, \mu=0.0020$ (Gain: 16.63 dB)
  - 0 dB: $\text{order}=32, \mu=0.0050$ (Gain: 14.27 dB)
  - 5 dB: $\text{order}=32, \mu=0.0050$ (Gain: 11.61 dB)
  - 10 dB: $\text{order}=32, \mu=0.0050$ (Gain: 7.94 dB)
- **Note:** Best setting differs between low (-5 dB) and high (10 dB) SNR: $[\text{order}=32, \mu=0.002]$ vs $[\text{order}=32, \mu=0.005]$.
- **Recommended Default:** $\text{order} = 32, \mu = 0.0050$ (Mean Gain: 12.40 dB, Worst-Case Gain: 6.03 dB).

## Limitations

1. **Synthetic Noise vs Real Acoustic Paths:** All metrics reported in this document were derived exclusively from synthetic tone, two-tone, and chirp signals combined with white and coloured noise filtered through an artificial 16-tap FIR channel.
2. **TODO: re-run on P3's subset:** These tuned values serve as initial DSP baselines and must be re-evaluated and confirmed on the frozen speech dataset subset (`dataset-v1`) as soon as real voice recordings and non-stationary acoustic recordings are available from P3.
3. **No Voice Activity Detector (VAD):** Single-mic spectral subtraction and Wiener filtering rely on initial noise lead-in (`noise_mode="initial"`). In continuous streaming, an adaptive VAD or continuous noise floor tracker is required.
