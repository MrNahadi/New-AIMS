# Context Document

I would like to build an Explainable ML for Predictive maintenance of Marine Diesel Engines(LightGBM & SHAP), Here is a description of the dataset I have:

## About Dataset

This dataset contains sensor data generated for marine engine performance monitoring and fault diagnosis. Marine engines are the backbone of maritime transportation, and their operational reliability is crucial for safety and efficiency. The dataset simulates real-world onboard monitoring systems by capturing engine parameters under both normal and faulty operating conditions.

### Dataset Highlights

- **10,000+ samples** (scalable to larger datasets)
- **Time-series format** with second-level resolution
- **Multiple sensor features** covering combustion, exhaust, vibration, lubrication, and environment
- **Target column** (Fault_Label) indicating health state
- **Normal operation and 7 fault categories:**
    - Fuel Injection Fault
    - Cylinder Pressure Loss
    - Exhaust Gas Overheating
    - Bearing/Vibration Fault
    - Lubrication Oil Degradation
    - Turbocharger Failure
    - Mixed Fault
- **Key data fields:**
    - Timestamp → Sequential time index
    - Cylinder Pressure (per cylinder) → Combustion performance
    - Exhaust Gas Temperature (per cylinder) → Thermal condition
    - Vibration (X, Y, Z) → Structural vibration monitoring
    - Lubrication Oil (Temp, Pressure) → Oil health indicators
    - Shaft RPM, Engine Load, Torque → Operational parameters
    - Fuel Flow & Air Pressure → Engine efficiency
    - Ambient Conditions → Environmental factors
    - Fault_Label → Target column (0 = Normal, 1–7 = Fault categories)
- **Applications:** Marine engine fault diagnosis, predictive maintenance systems, maritime big data analytics, and health monitoring research

## Plan

I would Like to go about building the Model in this way:

### Phase 1: Exploratory Data Analysis (EDA)

Before training, you must understand the "physics" inside your data.

* **Check Class Imbalance:** In real-world data, "Normal" (Label 0) usually makes up 90%+ of the data, while faults (1-7) are rare. If you don't handle this, your model will just guess "Normal" every time and achieve 90% accuracy but 0% utility.
* **Sensor Correlation Heatmap:** Check which sensors move together. For example, *Exhaust Gas Temperature (EGT)* often correlates with *Cylinder Pressure*.
* **Fault Signatures:** Plot the sensor data for specific fault windows.
* *Question to ask the data:* When `Fault_Label = 3` (Exhaust Gas Overheating), does the EGT sensor spike immediately, or is there a slow drift? This tells you how large your "observation window" needs to be.



### Phase 2: Preprocessing & Feature Engineering

This is the most critical phase for time-series data. You cannot simply feed raw rows into a model because a single second of data rarely tells the whole story; the **trend** matters.

**1. Windowing (Lag Features)**
Instead of using a single timestamp to predict a fault, use a "window" of past data (e.g., the last 60 seconds).

* **Sliding Window:** Create 3D arrays of shape `(Samples, Time_Steps, Features)` if using Deep Learning, or flatten them for Tree models.
* **Rolling Statistics:** Calculate the Mean, Standard Deviation, Min, and Max over the last  seconds for each sensor. High variance in vibration usually indicates a `Bearing/Vibration Fault`.

**2. Domain-Specific Features**

* **Inter-Cylinder Deviation:** For multi-cylinder engines, the *difference* between one cylinder's pressure and the average of all others is often more predictive than the raw pressure value.


* **Rate of Change:** Calculate the first derivative (slope) of temperatures. A rapid rise in lubrication oil temperature is more alarming than a high but stable temperature.

**3. Data Splitting (Crucial Warning)**
**Do not do a random shuffle split.**
Since this is time-series data, random shuffling causes "Data Leakage" (the model sees the future during training). You must use a **Time-based Split**:

* *Train:* First 70% of time.
* *Validation:* Next 15%.
* *Test:* Last 15%.

---

### Phase 3: Model Selection

Given your dataset size (10k+ samples) and nature, here are the two best approaches:

#### Option A: The Baseline (Gradient Boosting)

Start here. Models like **XGBoost**, **LightGBM**, or **Random Forest** are excellent at handling tabular sensor data and provide feature importance (interpretability).

* **Input:** Flattened windows with rolling statistics features.
* **Pros:** Fast to train, handles class imbalance well (using `class_weight`), interpretable.

#### Option B: The Advanced Approach (Deep Learning)

If the patterns are complex temporal dependencies (e.g., a vibration flutter that happens 10 seconds before a failure), use **LSTMs (Long Short-Term Memory)** or **1D-CNNs**.

* **Input:** Raw time-series sequences (e.g., 60-second windows).
* **Pros:** Can learn raw signal patterns without manual feature engineering.
* **Cons:** Harder to tune, "black box" (hard to explain to marine engineers).

---

### Phase 4: Evaluation Metrics

Accuracy is a bad metric here due to class imbalance. Use these instead:

1. **Confusion Matrix:** You need to know *exactly* which faults are being confused. (e.g., Is the model confusing "Fuel Injection Fault" with "Cylinder Pressure Loss"?)
2. **Recall (Sensitivity):** For maintenance, False Negatives are dangerous. You want high Recall. It is better to flag a healthy engine as faulty (False Alarm) than to miss a Turbocharger Failure (Missed Detection).
3. **F1-Score:** The harmonic mean of Precision and Recall, useful for imbalanced datasets.

---

### Phase 5: Explainability with SHAP

#### Feature Engineering for SHAP Interpretability

**The Core Principle:** SHAP explanations are only as useful as your feature names.

- **Don't:** Feed raw sensor values → SHAP tells you "Sensor X at timestamp T matters" (vague)
- **Do:** Engineer aggregated features → SHAP tells you "60-second Rolling Variance of Vibration is high" (actionable)

**Recommended Feature Naming Conventions:**
```
EGT_mean_60s, EGT_std_60s, Vibration_max_10s, Oil_Temp_rate_of_change, etc.
```

#### Multi-Class SHAP Explanation

Your dataset has 8 classes (Normal + 7 Fault Types). SHAP handles this by returning a matrix of values, one set per class.

**Workflow:**
1. Get model's predicted class (e.g., Class 3 = Exhaust Overheating)
2. Inspect SHAP values **only for that predicted class**
3. Example output: "Exhaust_Temp contributed +40% probability; Oil_Pressure had negligible effect"

#### Handling Correlated Features

Marine sensors are naturally correlated (physics-driven). Engine Load ↑ → Fuel Flow ↑ → Exhaust Temp ↑

**The Problem:** LightGBM may randomly split importance between correlated features, and SHAP may misattribute causality.

**The Solution:**
- During Phase 1 (EDA), identify correlations > 0.95
- Either drop one feature or engineer a ratio (e.g., `Fuel_to_Load_Ratio`)
- This ensures SHAP explanations reflect true physical causality


### Summary Checklist

| Step | Action Item |
| --- | --- |
| **1** | **Clean:** Handle missing values (forward fill is usually best for sensors). |
| **2** | **Engineer:** Create rolling means, std devs, and inter-cylinder deviations. |
| **3** | **Split:** strict time-based split (Train on past, test on future). |
| **4** | **Train:** Start with XGBoost or Random Forest. |
| **5** | **Evaluate:** Optimize for Recall (catch as many faults as possible). |

## Phase 6: Refinements & Advanced Strategies (Brainstormed)

### 1. Physics-Informed Residuals
Tree models struggle to learn linear dependencies like "Temperature threshold increases with Load".
*   **Action:** Create "Residual Features".
    *   Train a simple regression: `Expected_EGT = f(Load, RPM)` on healthy data.
    *   Feature: `EGT_Residual = Actual_EGT - Expected_EGT`.
    *   This isolates the fault signature from operational changes.

### 2. Frequency Domain Features (Vibration)
Rolling statistics (mean/std) miss periodic faults (e.g., shaft misalignment).
*   **Action:** Apply Fast Fourier Transform (FFT) on vibration windows.
*   **Features:** Energy in Low/Mid/High frequency bands, or Peak Frequency.

### 3. Prediction Smoothing
Raw model predictions can be "jittery" (e.g., 0, 0, 1, 0, 1).
*   **Action:** Apply a **Rolling Majority Vote** or **Persistence Filter** (e.g., "Alarm only if 5 consecutive windows are predicted faulty").

### 4. Hierarchical Modeling
Class imbalance (90% Normal) is extreme.
*   **Action:** Split into two models:
    1.  **Detector (Binary):** Is there a fault? (Optimized for Recall).
    2.  **Diagnoser (Multi-class):** Which fault is it? (Optimized for Accuracy).

## Implementation Log

### Update: Addressing Poor F1 Scores in Thermodynamic Faults
**Status:** Strategy Defined

*   **What:** Apply **Physics-Informed Residuals** to normalize sensor data against operating conditions.
*   **Why:** Initial model runs yielded poor F1 scores (~0.30) for thermodynamic faults (e.g., Fuel Injection). The model was confusing high operational stress (e.g., high load/ambient temp) with actual fault conditions.
*   **How:**
    1.  **Baseline:** Trained a Linear Regression model on **Healthy Data only** (`Fault_Label == 0`).
    2.  **Mapping:** Modeled the relationship: `Expected_Value = f(Engine_Load, RPM, Ambient_Temp)`.
    3.  **Feature Engineering:** Created new features based on `Residual = Actual_Value - Expected_Value`.
    4.  **Impact:** This isolates the fault signature by removing the bias caused by environmental factors and engine load.
