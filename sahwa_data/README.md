# sahwa Dataset
**Project:** sahwa — AI-Powered Neurological Wearable for Epilepsy, Parkinson's, and Fall Detection  
**Competition:** URIC ADU 2026  
**Authors:** Youssef Mohamed (1093155), Mazen Ahmed (1093429)  
**Advisor:** Dr. Rabah Al Abdi  
**Collection Date:** March 6–14, 2026  

---

## Hardware

- **MCU:** ESP32-C3
- **IMU:** MPU6500 (6-axis: 3-axis accelerometer + 3-axis gyroscope)
- **Sample Rate:** 50 Hz
- **Accelerometer Range:** ±16g
- **Gyroscope Range:** ±2000 dps
- **Interface:** I2C (SDA=5, SCL=6)

---

## CSV Format

All files share the same 9-column schema:

| Column | Type | Description |
|---|---|---|
| `timestamp_ms` | int | Device uptime in milliseconds |
| `ax` | float | Accelerometer X axis (g) |
| `ay` | float | Accelerometer Y axis (g) |
| `az` | float | Accelerometer Z axis (g) |
| `gx` | float | Gyroscope X axis (deg/s) |
| `gy` | float | Gyroscope Y axis (deg/s) |
| `gz` | float | Gyroscope Z axis (deg/s) |
| `label` | int | Numeric label ID |
| `label_name` | str | Human-readable label name |

---

## Dual-Model Architecture

The system uses **two separate models** due to different sensor placements for different conditions:

- **Wrist model** → Epilepsy mode (seizure + fall detection)
- **Ankle model** → Parkinson's mode (FoG detection)

Both models share common activity labels (Stand, Walk, FastWalk, Sit, SitStand) to provide context-aware baseline classification.

---

## Wrist Model Dataset (`/wrist/`)

Sensor worn on the dominant wrist. Subject: healthy adult male simulating target conditions.

| File | Label ID | Samples | Duration | Description |
|---|---|---|---|---|
| `Stand.csv` | 0 | 50,105 | 16.7 min | Standing still |
| `Walk.csv` | 1 | 54,204 | 18.1 min | Normal walking pace |
| `FastWalk.csv` | 2 | 50,356 | 16.8 min | Brisk/fast walking |
| `Sit.csv` | 3 | 50,038 | 16.7 min | Seated, minimal movement |
| `SitStand.csv` | 4 | 50,026 | 16.7 min | Repeated sit-to-stand transitions |
| `Fall.csv` | 6 | 50,278 | 16.8 min | Simulated forward/backward/sideways falls onto soft surface |
| `Seizure.csv` | 8 | 63,409 | 21.1 min | Simulated tonic-clonic seizure cycles |

**Total wrist samples: 368,416**

**Note:** Wrist Stairs data was not collected — IMU accelerometer clipping (>16g) occurred during heel-strike impact at the wrist. Stair detection is handled exclusively by the ankle model.

### Seizure Simulation Protocol
Each cycle (~2 min):
1. **Tonic phase (15s):** arm rigidly stiffened
2. **Clonic phase (90s):** rapid wrist jerking ~4×/sec, large amplitude, gradually slowing
3. **Postictal phase (20s):** arm completely limp

Signal characteristics: accel mean ~1.47g, gyro ~103 deg/s, ~20% samples >2g, max ~7.2g.

---

## Ankle Model Dataset (`/ankle/`)

Sensor taped firmly to the lateral side of the dominant ankle, sock worn over sensor.

| File | Label ID | Samples | Duration | Description |
|---|---|---|---|---|
| `Stand.csv` | 0 | 50,060 | 16.7 min | Standing still |
| `Walk.csv` | 1 | 52,367 | 17.5 min | Normal walking pace |
| `FastWalk.csv` | 2 | 52,516 | 17.5 min | Brisk/fast walking |
| `Sit.csv` | 3 | 51,885 | 17.3 min | Seated, minimal movement |
| `SitStand.csv` | 4 | 50,040 | 16.7 min | Repeated sit-to-stand transitions |
| `Stairs.csv` | 5 | 58,005 | 19.3 min | Stair climbing up and down, toe-first stepping |
| `FoG.csv` | 7 | 53,250 | 17.8 min | Simulated Freezing of Gait episodes |

**Total ankle samples: 368,123**

### FoG Simulation Protocol
Each cycle (~55s):
1. **Walk phase (30s):** normal walking
2. **Freeze phase (20s):** rapid tiny foot shuffles ~4–5×/sec, feet barely leaving ground
3. **Recovery (5s):** resuming normal walk

Signal characteristics: dominant frequency shifts from ~1.7 Hz (walk) to ~5.0–6.4 Hz (freeze) — a 3–4× frequency contrast that is the primary FoG detection feature.

---

## Signal Quality Summary

| Label | Placement | Mean g | Gyro °/s | Dom Hz | Clip % |
|---|---|---|---|---|---|
| Stand | Wrist | 0.992 | 9.3 | 0.56 | 0.00 |
| Walk | Wrist | 1.003 | 47.2 | 1.26 | 0.00 |
| FastWalk | Wrist | 1.208 | 140.6 | 1.84 | 0.00 |
| Sit | Wrist | 0.995 | 10.7 | 0.72 | 0.00 |
| SitStand | Wrist | 0.997 | 28.8 | 2.06 | 0.00 |
| Fall | Wrist | 1.008 | 43.0 | 0.86 | 0.00 |
| Seizure | Wrist | 1.474 | 103.2 | 0.02 | 0.00 |
| Stand | Ankle | 0.995 | 3.2 | 2.96 | 0.00 |
| Walk | Ankle | 1.462 | 143.9 | 1.74 | 0.00 |
| FastWalk | Ankle | 1.883 | 183.5 | 1.98 | 0.00 |
| Sit | Ankle | 1.006 | 3.6 | 10.28 | 0.00 |
| SitStand | Ankle | 1.007 | 17.0 | 2.90 | 0.00 |
| Stairs | Ankle | 1.137 | 80.8 | 1.32 | 0.00 |
| FoG | Ankle | 1.162 | 67.7 | 5.42 | 0.00 |

---

## Limitations

- Data collected from a single healthy subject simulating target conditions. Future work will validate on clinically confirmed patients.
- Wrist stairs label absent due to IMU range limitation during impact.
- FoG simulation may not fully capture all clinical presentations of Parkinson's freezing episodes.

---

## Intended ML Pipeline

- **Window size:** 256 samples (5.12s) with 50% overlap
- **Features:** mean, std, min, max, RMS, dominant frequency, spectral entropy, zero-crossing rate per axis + magnitude
- **Wrist model:** Random Forest (200 trees), 8-class (no Stairs)
- **Ankle model:** Gradient Boosting, 7-class (with Stairs, with FoG)
- **Validation:** 5-fold stratified cross-validation
- **Class imbalance:** SMOTE oversampling on training folds only
- **Target:** ≥85% macro F1-score
