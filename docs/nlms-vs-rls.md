# Architectural Analysis: STFT-Domain NLMS vs RLS for DRDO-ANC

Conforms to DRDO-ANC AI Coding Contract v1 (Section C & Section D).

## 1. Computational Complexity Comparison

In the subband STFT adaptive filtering architecture, an independent adaptive filter is evaluated for each of the $N_{\text{FREQ}} = 257$ frequency bins across discrete time frames $t$. Let $L$ denote the filter tap length ($L = \text{NLMS\_TAPS} = 4$ frames).

| Algorithm | Computational Complexity per Bin per Frame | Memory per Bin | Total Ops per Frame ($N_{\text{FREQ}} = 257, L = 4$) |
|---|---|---|---|
| **NLMS** | $\mathcal{O}(L)$ | $\mathcal{O}(L)$ (weights $W \in \mathbb{C}^L$) | $\approx 257 \times (2L + 2) \approx 2\,570$ FLOPs |
| **RLS**  | $\mathcal{O}(L^2)$ | $\mathcal{O}(L^2)$ (covariance matrix $P \in \mathbb{C}^{L \times L}$) | $\approx 257 \times (3L^2 + 4L) \approx 16\,448$ FLOPs |

- **NLMS**: Requires an inner product $\sum_{l=0}^{L-1} W[k, l] X[k, t-l]$, an energy sum $\sum_{l=0}^{L-1} |X[k, t-l]|^2$, and a single scaled vector addition.
- **RLS**: Incurs matrix-vector products $\mathbf{P} \mathbf{x}$, an outer product update $\mathbf{P} \leftarrow \lambda^{-1}(\mathbf{P} - \mathbf{k} \mathbf{x}^H \mathbf{P})$, and scalar normalization per frequency bin at each frame.

---

## 2. Empirical Benchmark & Timing Measurements

Measurements were conducted on synthetic complex STFT spectra across $N_{\text{FREQ}} = 257$ frequency bins over a 4.0-second audio stream (250 frames at the standard $16\text{ ms}$ hop length) using `time.perf_counter`:

- **NLMS**:
  - Total processing time for 250 frames (4.0 s audio): **7.114 ms**
  - Average execution time per frame: **0.0285 ms**
  - Frame budget utilization: **0.18%** of the $16.0\text{ ms}$ real-time latency budget (`LATENCY_BUDGET_MS = 16`).
- **RLS**:
  - Total processing time for 250 frames (4.0 s audio): **34.251 ms**
  - Average execution time per frame: **0.1370 ms**
  - Frame budget utilization: **0.86%** of the $16.0\text{ ms}$ real-time latency budget (`LATENCY_BUDGET_MS = 16`).

---

## 3. Engineering Analysis & Recommendations

1. **Default Algorithm**:
   - **We retain NLMS as the default adaptive filtering algorithm.**
   - At $L = 4$ taps, NLMS runs in $0.0285\text{ ms}$ per frame, consuming less than $0.2\%$ of our $16\text{ ms}$ real-time budget. It offers robust numerical stability without matrix conditioning risks.

2. **Feasibility of RLS**:
   - While RLS is approximately $4.8\times$ more computationally expensive than NLMS, its measured processing time of $0.137\text{ ms}$ per frame remains comfortably below the $16\text{ ms}$ real-time limit, leaving over $15.8\text{ ms}$ of headroom per frame for upstream neural networks and I/O.
   - However, recursive inverse covariance updates in complex arithmetic are prone to loss of positive definiteness under low excitation (silence / near-zero reference energy).

3. **Proposal for Future Exploration**:
   - Based on the measured compute headroom, **we propose to investigate** a stabilized square-root or QR-RLS formulation in future phases if fast-changing non-stationary noise environments (e.g. abrupt weapon fire or vehicle RPM transitions) require faster convergence rates than NLMS can achieve.
