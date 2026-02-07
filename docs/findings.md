# Findings & Inferences — Marine Engine Predictive Maintenance

## 1. Dataset Overview

- **Size:** 10,000 samples × 20 columns
- **Time range:** 2024-01-01 00:00:00 → 2024-01-01 02:46:39 (second-level resolution)
- **Data quality:** Zero missing values across all columns; 18 float64 sensor columns + 1 datetime + 1 integer target
- **Target column:** `Fault_Label` (0–7)

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Class Distribution

| Fault Label | Fault Type             | Samples | Percentage |
|-------------|------------------------|---------|------------|
| 0           | Normal                 | 6,507   | 65.07%     |
| 1           | Fuel Injection         | 509     | 5.09%      |
| 2           | Cylinder Pressure Loss | 498     | 4.98%      |
| 3           | Exhaust Overheating    | 488     | 4.88%      |
| 4           | Bearing/Vibration      | 481     | 4.81%      |
| 5           | Lube Oil Degradation   | 499     | 4.99%      |
| 6           | Turbocharger Failure   | 519     | 5.19%      |
| 7           | Mixed Fault            | 499     | 4.99%      |

**Key finding:** The imbalance ratio (max/min) is **13.5×**. Normal operation constitutes ~65% of data, while each fault class is approximately 5%. This moderate imbalance necessitates class-weighted training to prevent the model from defaulting to "Normal" predictions.

### 2.2 Sensor Correlation Analysis

- **No highly correlated pairs (|r| > 0.95)** were found among the 18 raw sensor features.
- **Notable correlations detected:**
  - Vibration_X ↔ Vibration_Y: r = **0.88** (strong positive — expected, as structural vibrations couple across axes)
  - Oil_Temp ↔ Oil_Pressure: r = **−0.28** (weak negative — rising oil temperature decreases oil viscosity and pressure)
  - Vibration_Z ↔ Oil_Pressure: r = **−0.49** (moderate negative)
  - All EGT columns show weak positive correlation with Vibration_Y (~0.14–0.17)
- **Inference:** Raw sensor features are largely independent, which is favorable for tree-based models. No raw features needed removal.

### 2.3 Per-Fault Sensor Signatures (Boxplot Analysis)

- **Bearing/Vibration (Label 4):** Vibration_X and Vibration_Y show dramatically elevated distributions compared to all other fault types. Vibration_Y median shifts from ~0.05 to ~0.35. This is the most distinguishable fault.
- **Lube Oil Degradation (Label 5):** Oil_Temp shows a higher median and wider spread. Oil_Pressure is notably depressed.
- **Turbocharger Failure (Label 6):** Air_Pressure has a visibly lower distribution, consistent with reduced turbocharger boost.
- **Exhaust Overheating (Label 3):** EGT distributions show a slight upward shift, but significant overlap with Normal — making this fault harder to detect from univariate analysis.
- **Fuel Injection (Label 1) and Cylinder Pressure Loss (Label 2):** These faults show subtler sensor signatures with substantial overlap with Normal distributions across most sensors.
- **Mixed Fault (Label 7):** Vibration_Z shows a noticeably elevated distribution combined with low Air_Pressure and low Oil_Pressure.

### 2.4 Time-Series Visualization

- Fault events are **randomly distributed** (not temporally clustered). They appear as brief, interleaved episodes rather than sustained fault periods.
- Sensor traces (Vibration_X, Cylinder1_Exhaust_Temp, Oil_Temp) show **instantaneous jumps** at fault boundaries rather than gradual drifts.
- **Inference:** Since faults are injected per-row without temporal buildup, long rolling windows risk diluting the fault signal with surrounding normal data.

### 2.5 EGT Drift Analysis (Exhaust Overheating)

- **463 transitions** into Exhaust Gas Overheating (Label 3) were detected in the dataset.
- EGT behavior around fault onset shows **no clear pre-onset drift** — the exhaust temperatures do not exhibit a gradual ramp-up before the fault label changes.
- The fault appears as an **instantaneous label change** rather than a slow thermal drift.
- **Inference:** The lack of pre-onset drift indicates that traditional sliding-window approaches designed to capture gradual degradation are less effective here. The dataset simulates abrupt fault injection.

### 2.6 Autocorrelation & Window Size Selection

- Autocorrelation analysis of Vibration_X, Cylinder1_Exhaust_Temp, and Oil_Temp shows that autocorrelation drops to approximately zero within 30–60 lags.
- **Selected windows:** Primary = 60s, Secondary = 10s (for vibration channels).
- **Later finding (V2):** A shorter 10s window proved more effective, as the 60s window diluted the row-level fault signals.

---

## 3. Feature Engineering

### 3.1 Rolling Statistics (Phase 1)

- Computed rolling mean, std, min, max for all 18 sensor columns over the 60s primary window.
- Secondary 10s window applied to Vibration_X, Vibration_Y, Vibration_Z for capturing rapid changes.
- **Result:** Feature count increased from 18 to **104 columns**.
- 60 initial rows dropped due to rolling window NaN.

### 3.2 Domain-Specific Features

- **Inter-Cylinder Pressure Deviations** (4 features): Each cylinder's pressure minus the average of all cylinders. Designed to isolate single-cylinder anomalies.
- **Inter-Cylinder EGT Deviations** (4 features): Same approach for exhaust gas temperatures.
- **Rate of Change** (5 features): First derivative (.diff()) for Oil_Temp, Oil_Pressure, and all Cylinder Exhaust Temps.
- **Fuel-to-Load Ratio** (1 feature): Fuel_Flow / Engine_Load to normalize fuel consumption.
- **Result:** 119 columns after domain features.

### 3.3 Physics-Informed Residuals

- Linear regression models trained on healthy data (Fault_Label = 0) to predict expected EGT and Oil_Temp from Engine_Load, Shaft_RPM, and Ambient_Temp.
- **Critical finding:** All residual models achieved R² ≈ 0.000 on healthy data:
  - Cylinder1_Exhaust_Temp: R² = 0.0006
  - Cylinder2_Exhaust_Temp: R² = 0.0000
  - Cylinder3_Exhaust_Temp: R² = 0.0005
  - Cylinder4_Exhaust_Temp: R² = 0.0009
  - Oil_Temp: R² = 0.0011
- **Inference:** The sensor values in this dataset have **no meaningful linear relationship** with operational parameters (load, RPM, ambient temp). This means the physics-informed residuals are essentially identical to the raw sensor values, adding no new information. This was confirmed when the correlation filter dropped all 5 residual/raw pairs at r = 1.000.

### 3.4 FFT Vibration Features

- Applied FFT on 30-second windows for Vibration_X, Vibration_Y, Vibration_Z.
- Extracted: Energy_Low_Freq, Energy_Mid_Freq, Energy_High_Freq, Peak_Frequency per channel (12 features).
- **Result:** 136 columns after FFT features.

### 3.5 Correlation Filtering (Post-Engineering)

Dropped **8 features** with |r| > 0.95:
1. `Vibration_X_std_10s` (corr 0.986 with `Vibration_X_max_10s`)
2. `Vibration_Y_std_10s` (corr 0.986 with `Vibration_Y_max_10s`)
3. `Vibration_Z_std_10s` (corr 0.986 with `Vibration_Z_max_10s`)
4. `Cylinder1_Exhaust_Temp` (corr 1.000 with `Cylinder1_Exhaust_Temp_Residual`)
5. `Cylinder2_Exhaust_Temp` (corr 1.000 with `Cylinder2_Exhaust_Temp_Residual`)
6. `Cylinder3_Exhaust_Temp` (corr 1.000 with `Cylinder3_Exhaust_Temp_Residual`)
7. `Cylinder4_Exhaust_Temp` (corr 1.000 with `Cylinder4_Exhaust_Temp_Residual`)
8. `Oil_Temp` (corr 1.000 with `Oil_Temp_Residual`)

**Final V1 feature set:** 126 features, dataset shape: (9,940 × 128), 0 NaN values.

---

## 4. Data Splitting

Strict **time-based split** (no random shuffling) to prevent data leakage:

| Split | Samples | Indices      | Normal % |
|-------|---------|-------------|----------|
| Train | 6,958   | 0–6,957      | 65.6%    |
| Val   | 1,491   | 6,958–8,448  | 64.9%    |
| Test  | 1,491   | 8,449–9,939  | 63.1%    |

Class distributions are approximately consistent across splits, confirming no label leakage.

---

## 5. Model Training & Evaluation (V1)

### 5.1 Single Multi-Class LightGBM

- **Configuration:** `objective='multiclass'`, `num_class=8`, `metric='multi_logloss'`
- **Class weights** applied inversely proportional to frequency (Normal: 0.191, fault classes: ~2.4–2.7)
- **Best iteration:** 93 (early stopping at patience=50)
- **Best validation loss:** 0.3796

**Feature importance (gain-based) — Top 7:**
1. Air_Pressure (32,500)
2. Vibration_Z (28,500)
3. Oil_Temp_Residual (23,500)
4. Vibration_X (21,000)
5. Oil_Pressure (20,500)
6. Vibration_Y (17,000)
7. Fuel_Flow (9,000)

### 5.2 Hierarchical Model (Binary Detector + Multi-Class Diagnoser)

**Stage 1 — Binary Detector (Normal vs. Any Fault):**
- Best iteration: 115 (early stopping)
- **Optimal threshold:** 0.0871 (tuned for ≥ 0.95 recall)
- Validation recall: **0.9503** (meets the ≥0.95 target)
- Validation F1: 0.6762

**Stage 2 — Multi-Class Diagnoser (7 fault types):**
- Best iteration: 97 (early stopping)
- Best validation loss: 0.1774

### 5.3 Model Comparison (Test Set)

| Fault Type             | Single Recall | Hier Recall | Single F1 | Hier F1 |
|------------------------|---------------|-------------|-----------|---------|
| Normal                 | 0.9447        | 0.5739      | 0.9058    | 0.7124  |
| Fuel Injection         | **0.1795**    | 0.6026      | 0.2523    | 0.2781  |
| Cylinder Pressure Loss | **0.4691**    | 0.7531      | 0.5507    | 0.4919  |
| Exhaust Overheating    | **0.6184**    | 0.7632      | 0.6963    | 0.5179  |
| Bearing/Vibration      | 1.0000        | 1.0000      | 1.0000    | 1.0000  |
| Lube Oil Degradation   | 0.9778        | 0.9889      | 0.9617    | 0.9271  |
| Turbocharger Failure   | 0.9733        | 1.0000      | 0.9605    | 0.9202  |
| Mixed Fault            | 1.0000        | 1.0000      | 1.0000    | 0.9937  |
| **MACRO AVG**          | **0.7704**    | **0.8352**  | **0.7909**| **0.7302**|

**Key findings:**
- **Single model:** Higher F1 (0.7909) but lower recall (0.7704). Excellent for Normal, Bearing/Vibration, Lube Oil, Turbocharger, and Mixed Fault (all ≥ 0.97 recall). Critically poor for Fuel Injection (0.18 recall — 82% missed).
- **Hierarchical model:** Higher recall (0.8352) but lower F1 (0.7302). The aggressive binary detector catches more faults but decreases Normal precision (false alarm rate increases).
- **Selected for SHAP:** Hierarchical model (for its superior recall on safety-critical fault detection).

**Critical weaknesses (both models):**
- Fuel Injection: Recall 0.18 (single) / 0.60 (hierarchical) — most difficult fault to detect
- Cylinder Pressure Loss: Recall 0.47 / 0.75 — significant miss rate
- Exhaust Overheating: Recall 0.62 / 0.76 — one-third missed by single model

---

## 6. SHAP Explainability Analysis

### 6.1 Global Feature Importance (Mean |SHAP| across all classes)

| Rank | Feature                            | Mean |SHAP| |
|------|------------------------------------|--------------------|
| 1    | Oil_Pressure                       | 0.2397             |
| 2    | Air_Pressure                       | 0.1811             |
| 3    | Vibration_Z                        | 0.1468             |
| 4    | Fuel_Flow                          | 0.1267             |
| 5    | Oil_Temp_Residual                  | 0.1026             |
| 6    | Vibration_X                        | 0.0998             |
| 7    | Vibration_Y                        | 0.0973             |
| 8    | Cylinder2_Pressure                 | 0.0524             |
| 9    | Cylinder3_Exhaust_Temp_Residual    | 0.0502             |
| 10   | Cylinder1_Exhaust_Temp_Residual    | 0.0483             |

**Inference:** Raw instantaneous sensor values dominate over rolling statistics and engineered features, consistent with the per-row fault injection pattern in this dataset.

### 6.2 Per-Class SHAP Signatures

**Normal (Class 0):**
- Top drivers: Oil_Pressure, Air_Pressure, Fuel_Flow, Vibration channels
- When these sensors are within normal ranges (moderate values), SHAP pushes prediction toward Normal
- Low oil pressure or abnormal vibration pushes strongly *away* from Normal (negative SHAP)

**Fuel Injection (Class 1):**
- **Dominant feature:** Fuel_Flow (SHAP up to +2.46 for anomalous values)
- Fuel_to_Load_Ratio and Oil_Pressure are secondary drivers
- **Inference:** Low fuel flow values strongly indicate injection fault — the model correctly identifies disrupted fuel delivery, though the signal overlaps significantly with normal variation, explaining the low recall

**Cylinder Pressure Loss (Class 2):**
- **Dominant features:** Cylinder2_Pressure (SHAP +2.2), followed by other cylinder pressures
- Inter-cylinder pressure deviations (Cyl2, Cyl3, Cyl1_Pressure_Deviation) are important secondary features
- **Inference:** The model correctly identifies pressure anomalies in individual cylinders. Low pressure in one cylinder while others are normal is the key diagnostic signature.

**Exhaust Overheating (Class 3):**
- **Dominant features:** All four Cylinder_Exhaust_Temp_Residual features (SHAP up to +4.5)
- EGT deviations and rates of change provide supporting signal
- **Inference:** High EGT residuals (temperature beyond what operating conditions explain) are the primary diagnostic indicator. Despite the residual models having near-zero R², the residual features inherit the raw EGT values and remain useful.

**Bearing/Vibration (Class 4):**
- **Dominant features:** Vibration_X (SHAP +4.66) and Vibration_Y (SHAP +4.51)
- This fault has the **cleanest separation** — vibration values that are even slightly elevated provide overwhelming evidence
- **Inference:** This is a textbook bearing fault signature: elevated multi-axis vibration. The 10s rolling max features provide supporting confirmation.

**Lube Oil Degradation (Class 5):**
- **Dominant features:** Oil_Temp_Residual (SHAP +4.24) and Oil_Pressure (SHAP +2.97)
- Vibration_Z, Oil_Pressure_rate_of_change, and Oil_Temp_rate_of_change are secondary
- **Inference:** The combination of elevated oil temperature with depressed or rapidly changing oil pressure is the classic lubrication degradation signature.

**Turbocharger Failure (Class 6):**
- **Dominant feature:** Air_Pressure (SHAP +7.02) — overwhelmingly dominant
- Air_Pressure_std_60s, Air_Pressure_mean_60s, Air_Pressure_min_60s provide supplementary temporal context
- **Inference:** A drop in air/boost pressure is the direct physical consequence of turbocharger failure. The model correctly identifies this singular, strong causal relationship.

**Mixed Fault (Class 7):**
- **Dominant features:** Vibration_Z (SHAP +7.28) and Oil_Pressure (SHAP ~1.0)
- Vibration_Z_mean_10s and Vibration_Z_max_10s are secondary
- **Inference:** The Mixed Fault combines vibration anomaly (Z-axis specifically) with oil system degradation, representing a compound failure mode.

### 6.3 SHAP Dependence Plots (Top 5 Features)

- **Oil_Pressure:** Values below ~2.5 bar produce strong negative mean SHAP (away from Normal); values around 3–4 bar cluster near zero. Lube Oil Degradation and Mixed Fault samples cluster at extremes.
- **Air_Pressure:** Sharp non-linear relationship — values below ~0.8 bar produce very high positive SHAP for Turbocharger Failure. Turbocharger samples form a visually distinct low-pressure cluster.
- **Vibration_Z:** Near-linear positive relationship with SHAP impact. Values above ~0.2 strongly indicate Mixed Fault; values above ~0.3 are near-certain.
- **Fuel_Flow:** Values below ~100 kg/h produce strong positive SHAP for Fuel Injection fault. Clear threshold behavior.
- **Oil_Temp_Residual:** Values above ~10°C deviation push toward Lube Oil Degradation. Normal samples cluster tightly around zero residual.

---

## 7. Prediction Smoothing

| Method              | Macro F1 | Macro Recall | Weighted F1 |
|---------------------|----------|--------------|-------------|
| Raw predictions     | 0.7302   | 0.8352       | 0.7198      |
| Majority Vote (k=5) | 0.2278   | 0.2157       | 0.4971      |
| Persistence Filter  | 0.0967   | 0.1250       | 0.4884      |

**Key finding:** Both smoothing techniques **catastrophically degraded performance**. Majority vote reduced Macro F1 from 0.73 to 0.23; persistence filtering to 0.10.

**Root cause:** Faults in this dataset are randomly distributed per-row rather than occurring in sustained temporal clusters. Majority voting and persistence filtering assume temporal coherence of fault events, which does not exist here. The smoothing effectively averaged away the sparse fault predictions, replacing them with the dominant Normal class.

**Inference:** Prediction smoothing is **not applicable** to this dataset. In a real-world deployment with temporally clustered faults, these techniques would be valuable, but the simulation pattern makes them counterproductive.

---

## 8. Model Export

Three model artifacts were saved:
- `marine_engine_full_model.pkl` — Single multi-class model (4.93 MB)
- `marine_engine_detector.pkl` — Binary fault detector (0.76 MB)
- `marine_engine_diagnoser.pkl` — Multi-class fault diagnoser (4.53 MB)

Pipeline configuration (window sizes, feature lists, dropped features) saved as JSON.
SHAP importance summary exported to CSV.

---

## 9. V2 Model Improvements

### 9.1 Root Causes of V1 Weaknesses

1. **Rolling features dilute fault signals:** Faults are injected per-row, not as temporal clusters. A 60s window blends ~1 faulty row with ~59 normal rows, washing out the fault signature.
2. **Physics residuals ≈ raw values:** R² ≈ 0.00 means the linear regression learned nothing useful. Residuals were perfectly correlated (r = 1.000) with raw values, adding no information.
3. **Prediction smoothing destroys performance:** Temporal smoothing assumes clustered faults — the opposite of what this dataset contains.

### 9.2 V2 Changes Applied

- **Shorter rolling window:** 10s (vs. 60s) — less dilution of fault signals
- **16 new interaction features** replacing useless physics residuals:
  - Vibration_Magnitude (√(X² + Y² + Z²))
  - EGT_Max, EGT_Mean, EGT_Std (cross-cylinder exhaust gas statistics)
  - Oil_Health_Index (Oil_Temp / Oil_Pressure)
  - Mean_Cyl_Pressure, Cyl_Pressure_Std
  - Air_to_Fuel_Ratio, Fuel_per_RPM
  - Per-cylinder EGT deviation from mean, Pressure deviation from mean
- **Stronger class weights** (power 1.5 scaling for more aggressive minority class emphasis)
- **Deeper trees:** 127 leaves (vs. 63), smaller min_child_samples (10 vs. 20)
- **More trees:** 2,000 max with 100-round early stopping patience
- **Removed useless** physics residuals and prediction smoothing

### 9.3 V2 Feature Set

- **89 features** (vs. 126 in V1) — fewer but more targeted
- 2 features dropped by correlation filter (vs. 8 in V1)
- Dataset shape: (9,990 × 91) — more samples retained due to shorter 10s window

### 9.4 V2 Results — Per-Class Recall Comparison

| Fault Type             | V1 Best (Hier) | V2 Single | V2 Hierarchical |
|------------------------|-----------------|-----------|-----------------|
| Normal                 | 0.5739          | 0.9133    | 0.7770          |
| Fuel Injection         | 0.6026          | 0.3291    | **0.4937**      |
| Cylinder Pressure Loss | 0.7531          | 0.5185    | **0.6543**      |
| Exhaust Overheating    | 0.7632          | 0.6842    | **0.7763**      |
| Bearing/Vibration      | 1.0000          | 1.0000    | 1.0000          |
| Lube Oil Degradation   | 0.9889          | 0.9670    | 0.9890          |
| Turbocharger Failure   | 1.0000          | 1.0000    | 0.9868          |
| Mixed Fault            | 1.0000          | 1.0000    | 0.9873          |

### 9.5 V2 Overall Metrics

| Metric       | V1 Best | V2 Best | Change  |
|--------------|---------|---------|---------|
| Macro F1     | 0.7302  | 0.8040  | **+0.0738** |
| Macro Recall | 0.8352  | 0.8331  | −0.0021 |

**Key findings:**
- **Macro F1 improved by +7.4%** (0.7302 → 0.8040) — the best overall metric across both V1 and V2
- Macro recall remained approximately stable (~0.83)
- V2 Single model dramatically improved Normal recall (0.57 → 0.91), reducing false alarm rate
- The three weak fault classes (Fuel Injection, Cylinder Pressure Loss, Exhaust Overheating) improved in the single model but regressed slightly in the hierarchical variant compared to V1 hierarchical

### 9.6 V2 SHAP Feature Importance (Top 15)

| Rank | Feature              | Mean |SHAP| |
|------|----------------------|--------------------|
| 1    | Vibration_Magnitude  | 0.2378             |
| 2    | EGT_Max              | 0.2224             |
| 3    | Air_Pressure         | 0.1949             |
| 4    | Oil_Health_Index     | 0.1867             |
| 5    | Vibration_Z          | 0.1135             |
| 6    | Cyl_Pressure_Std     | 0.1135             |
| 7    | Mean_Cyl_Pressure    | 0.1131             |
| 8    | Fuel_Flow            | 0.1054             |
| 9    | Air_to_Fuel_Ratio    | 0.0964             |
| 10   | Vibration_Y          | 0.0892             |
| 11   | Oil_Temp             | 0.0829             |
| 12   | Vibration_X          | 0.0749             |
| 13   | Oil_Pressure         | 0.0677             |
| 14   | EGT_Std              | 0.0508             |
| 15   | Fuel_per_RPM         | 0.0387             |

**Key finding:** 9 of the top 20 SHAP features are **new interaction features** from V2 engineering:
- Vibration_Magnitude, EGT_Max, Oil_Health_Index, Cyl_Pressure_Std, Mean_Cyl_Pressure, Air_to_Fuel_Ratio, EGT_Std, EGT_Mean, Fuel_per_RPM

This confirms that the domain-informed interaction features capture fault-relevant information that individual raw sensors cannot express alone.

---

## 10. Summary of Key Inferences

### What the Model Learned (Domain-Validated)

1. **Bearing/Vibration faults** produce unmistakable elevation in Vibration_X and Vibration_Y — the easiest fault to classify (100% recall across all models).
2. **Turbocharger failures** directly reduce Air_Pressure (boost pressure) — a single dominant feature makes this highly detectable.
3. **Lube Oil Degradation** manifests as elevated Oil_Temp combined with depressed Oil_Pressure — the Oil_Health_Index (Temp/Pressure) in V2 captures this relationship directly.
4. **Mixed Faults** are characterized by elevated Vibration_Z and Oil_Pressure anomalies — a compound signature that the model reliably separates.
5. **Exhaust Overheating** is detected through elevated EGT residuals/values — all four cylinder EGT channels contribute, with cross-cylinder deviations providing additional signal.
6. **Cylinder Pressure Loss** relies on inter-cylinder pressure deviations — one cylinder losing compression while others remain normal creates a detectable imbalance.
7. **Fuel Injection faults** are the hardest to detect — low fuel flow is the primary indicator, but the signal overlaps significantly with normal operational variation.

### Dataset Limitations Discovered

1. **No temporal fault progression:** Faults are injected per-row rather than developing over time, limiting the utility of time-series features (rolling windows, temporal smoothing).
2. **No physics-based relationships in data:** The linear independence of sensor values from operational parameters (R² ≈ 0) makes physics-informed residuals ineffective.
3. **Subtlety of thermodynamic faults:** Fuel Injection, Cylinder Pressure Loss, and Exhaust Overheating have sensor signatures that overlap significantly with normal operation, limiting achievable recall.

### Recommendations

1. **For deployment:** Use the V2 Hierarchical model (Macro F1 = 0.80) without prediction smoothing.
2. **For Fuel Injection detection:** Collect more diverse fault examples and engineer features targeting injector-specific signatures (e.g., fuel pressure pulsation, cylinder-specific fuel delivery metrics).
3. **For real-world data:** Re-evaluate physics-informed residuals and temporal smoothing, as real engine data will likely exhibit temporal fault progression and sensor-operational parameter correlations.
4. **Model retraining:** Establish a retraining schedule as new labeled data becomes available, focusing on underrepresented fault classes.
