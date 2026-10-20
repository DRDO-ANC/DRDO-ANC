# Datasets and Sourcing Specification

## 1. Defence Proxy Statement and Rationale

In this project, we propose to investigate hybrid AI and adaptive-filtering methods for speech enhancement in challenging defence acoustic environments. Real operational defence audio recordings are sensitive, classified, or unavailable. Therefore, we utilize curated open-source acoustic corpora as realistic proxies for operational noises (e.g., machinery, gunfire, explosions, rotorcraft, and engine sounds) alongside standard clean speech corpora.

No classified, proprietary, or real defence operational recordings are included or used in this repository. All proxy datasets are publicly accessible research datasets or open audio repositories.

---

## 2. Dataset Manifest Freeze Status

Per Section D of the AI Coding Contract v1, dataset splits are strictly partitioned by speaker ID and held-out noise classes. Once generated, verified, and tagged at `dataset-v1`, the manifest and split assignments are permanently frozen.

| Dataset Version | Git Tag | Manifest File | Manifest SHA-256 Hash | Freeze Date | Sign-off |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dataset-v1` | `dataset-v1` | `data/manifest.csv` | *(to be calculated at freeze)* | *(pending)* | *(pending)* |

---

## 3. Dataset Profiles

Allowed contract `noise_type` values (Section D):
`gunshot`, `explosion`, `helicopter`, `vehicle`, `drone`, `siren`, `machinery`, `other`.

---

### 3.1 LibriSpeech ASR Corpus

- **Role**: Clean Speech.
- **Contract Splits**:
  - `train-clean-100`: Partitioned by speaker ID for `train` and `val` splits.
  - `dev-clean`: Reserved for the `test` split.
- **Noise Classes / Mapped `noise_type`**: None (clean speech reference).
- **Source URL**: `https://www.openslr.org/12`
- **Licence**: `TO VERIFY` *(Note: OpenSLR lists LibriSpeech under CC BY 4.0; to verify against official OpenSLR distribution page).*
- **Version**: 1.0 (train-clean-100, dev-clean subsets).
- **Approximate Size**: ~6.3 GB (`train-clean-100.tar.gz`), ~337 MB (`dev-clean.tar.gz`).
- **Notes**: High-quality 16 kHz read English speech derived from LibriVox audiobooks. Utterance boundaries and speaker IDs are explicitly defined in file paths.

---

### 3.2 DEMAND (Diverse Environments Multi-channel Acoustic Noise Database)

- **Role**: Noise (Stationary & Quasi-Stationary Background).
- **Noise Classes & Mapped `noise_type`**:
  - `DKITCHEN`, `DLIVING`, `DWASHING`, `OHALL`, `OMEETING`, `OOFFICE`: mapped to `machinery` or `other`.
  - `NFIELD`, `NPARK`, `NRIVER`: mapped to `other`.
  - `TBUS`, `TCAR`, `TMETRO`: mapped to `vehicle`.
  - `SPSQUARE`, `STRAIN`, `SCAFE`: mapped to `other` or `machinery`.
- **Source URL**: `https://zenodo.org/records/1227121`
- **Licence**: `TO VERIFY` *(Note: Often distributed under CC BY-SA 3.0 / Creative Commons; verify against Zenodo archive).*
- **Version**: 16-channel, 16 kHz / 48 kHz releases.
- **Approximate Size**: ~1–2 GB per environment package (using single-channel 16 kHz downsampled extraction).
- **Notes**: Provides multi-channel real-world recordings across indoor, outdoor, and transport environments; channels will be selected/downmixed to 16 kHz mono.

---

### 3.3 MAD (Military Acoustic Dataset / Mil-Acoustic)

- **Role**: Noise (Military Vehicle & Heavy Weaponry Proxies).
- **Noise Classes & Mapped `noise_type`**:
  - Ground vehicles (tanks, tracked carriers, trucks): mapped to `vehicle`.
  - Aircraft & rotorcraft: mapped to `helicopter` or `vehicle`.
  - Heavy weapons, ballistic discharges: mapped to `gunshot`.
- **Source URL**: `TO VERIFY` *(Note: Sourced from academic repositories hosting military vehicle and acoustic classification benchmarks; verify institutional URL).*
- **Licence**: `TO VERIFY` *(Note: Verify individual academic/institutional usage terms and distribution restrictions).*
- **Version**: Research baseline release.
- **Approximate Size**: ~500 MB – 2 GB (depending on subset).
- **Notes**: Serves as specialized proxy acoustic data for engine tracks and heavy tracked equipment.

---

### 3.4 ESC-50 (Dataset for Environmental Sound Classification)

- **Role**: Noise (Environmental & Impulsive).
- **Noise Classes & Mapped `noise_type`**:
  - Gunshot (`gunshot`): mapped to `gunshot`.
  - Explosion (`fireworks`, `explosion`): mapped to `explosion`.
  - Siren (`siren`, `car_horn`): mapped to `siren`.
  - Helicopter (`helicopter`): mapped to `helicopter`.
  - Engine / machinery (`engine`, `chainsaw`): mapped to `machinery` / `vehicle`.
  - Wind, rain, crackling fire: mapped to `other`.
- **Source URL**: `https://github.com/karolpiczak/ESC-50`
- **Licence**: `TO VERIFY` *(Note: Published under CC BY-NC 3.0; verify terms regarding derivative works and non-commercial research).*
- **Version**: 2.0.0 (2,000 environmental recordings of 5 seconds each).
- **Approximate Size**: ~840 MB.
- **Notes**: Standardized 5-second clips arranged in 5 predefined folds. Excellent for impulsive and non-stationary acoustic events.

---

### 3.5 Freesound

- **Role**: Noise (Impulsive & Specialized Proxies: Explosions, Gunshots, UAV/Drone Propellers).
- **Noise Classes & Mapped `noise_type`**:
  - Artillery, blasts, shockwaves: mapped to `explosion`.
  - Small arms fire, weapon discharges: mapped to `gunshot`.
  - Multi-rotor UAVs, drone motor whine: mapped to `drone`.
  - Warning horns, alert alarms: mapped to `siren`.
  - Mechanical noise: mapped to `machinery`.
- **Source URL**: `https://freesound.org`
- **Licence**: `TO VERIFY` (Per-clip basis).
- **Licence Policy**:
  - Freesound hosts audio under multiple licensing tiers.
  - **Only CC0 (Public Domain Dedication) and CC-BY (Creative Commons Attribution) clips are permitted.**
  - Clips with non-commercial (`CC-BY-NC`) or share-alike (`CC-BY-SA`) clauses must be excluded unless authorized.
- **Tracking Schema**: Every downloaded Freesound clip must be documented in a tracking CSV file (`freesound_clips.csv`) prior to inclusion in the manifest.
  
  Columns for `freesound_clips.csv`:
  ```csv
  clip_id,original_name,license,author,source_url,noise_type
  ```
  Example entries:
  ```csv
  123456,explosion_debris.wav,CC0,user1,https://freesound.org/s/123456/,explosion
  234567,quadcopter_hover.wav,CC-BY,user2,https://freesound.org/s/234567/,drone
  ```

---

## 4. Contract Noise Taxonomy Mapping Matrix

| Contract `noise_type` | Primary Source Datasets | Notes |
| :--- | :--- | :--- |
| `gunshot` | ESC-50, MAD, Freesound (CC0/CC-BY) | Impulsive proxy for small arms / artillery fire |
| `explosion` | Freesound (CC0/CC-BY), ESC-50 | Blast wave, detonation, shockwave proxies |
| `helicopter` | ESC-50, MAD, Freesound (CC0/CC-BY) | Rotorcraft blade slap and turbine hum |
| `vehicle` | DEMAND, MAD, ESC-50 | Tracked vehicles, military trucks, transport |
| `drone` | Freesound (CC0/CC-BY) | Small UAV motor whine, high-frequency propeller hum |
| `siren` | ESC-50, Freesound (CC0/CC-BY) | Tonal alert, evacuation siren |
| `machinery` | DEMAND, ESC-50, Freesound (CC0/CC-BY) | Stationary generators, compressors, heavy tools |
| `other` | DEMAND, ESC-50, Freesound (CC0/CC-BY) | Ambient wind, environmental noise floor |

---

## 5. Speaker and Noise Split Strategy

To prevent data contamination and ensure valid evaluation:
1. **Speaker Disjointness**: `train` and `val` splits are partitioned strictly by speaker ID from LibriSpeech `train-clean-100`. No speaker appearing in `train` or `val` shall ever appear in `test`. The `test` split is populated exclusively from LibriSpeech `dev-clean`.
2. **Held-out Noise Classes**: At least two full noise classes (or designated subclasses) will be held out entirely from training to form the `test_unseen_noise` evaluation split, evaluating the generalization of models and adaptive filters to novel acoustic conditions.
