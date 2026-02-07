# Marine Engine Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-02569B?style=for-the-badge&logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6F00?style=for-the-badge)](https://shap.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

An **Explainable Machine Learning** system for predictive maintenance of marine diesel engines, combining **LightGBM** gradient boosting with **SHAP** explainability to detect and diagnose 7 distinct engine fault types from streaming sensor data.

> **Best Model Performance:** Macro F1 = **0.8040** | Macro Recall = **0.8331** | 5 of 7 faults detected at ≥96% recall

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Explainability](#explainability)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [License](#license)

---

## Overview

Marine engines are the backbone of maritime transportation — their operational reliability is critical for safety and efficiency. This project builds a fault classification system that ingests 18 sensor channels from a marine diesel engine and classifies each observation into **Normal** operation or one of **7 fault types**.

```mermaid
flowchart LR
    A["18 Sensor\nChannels"] --> B["Feature\nEngineering\n89 features"]
    B --> C["LightGBM\nClassifier"]
    C --> D["SHAP\nExplainability"]
    D --> E["Fault\nDiagnosis"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#000
    style B fill:#fff3e0,stroke:#e65100,color:#000
    style C fill:#e8f5e9,stroke:#2e7d32,color:#000
    style D fill:#fce4ec,stroke:#c62828,color:#000
    style E fill:#f3e5f5,stroke:#6a1b9a,color:#000
```

### Key Design Principles

- **Safety-first:** Optimized for **high recall** — a missed fault (false negative) is far more costly than a false alarm
- **Explainable:** SHAP-based feature attribution provides physically interpretable diagnostic signatures
- **Iterative:** Two development versions (V1 → V2) driven by root-cause analysis of failures

---

## Architecture

Two model architectures were developed and compared:

```mermaid
flowchart TD
    subgraph Single["Single Multi-Class Model — Best F1"]
        S1[Input Features] --> S2["LightGBM\n8-class classifier"]
        S2 --> S3["Output: Class 0–7"]
    end

    subgraph Hierarchical["Hierarchical Two-Stage Model — Best Recall"]
        H1[Input Features] --> H2["Stage 1: Binary Detector\nNormal vs. Any Fault"]
        H2 -->|"P(fault) < τ"| H3["Predict: Normal"]
        H2 -->|"P(fault) ≥ τ"| H4["Stage 2: 7-Class Diagnoser\nWhich fault?"]
        H4 --> H5["Output: Fault 1–7"]
    end

    style Single fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Hierarchical fill:#fff3e0,stroke:#e65100,color:#000
```

| Architecture | Macro F1 | Macro Recall | Best For |
|---|---|---|---|
| **V2 Single** | **0.8040** | 0.8331 | Overall balanced performance |
| **V2 Hierarchical** | 0.7302 | **0.8331** | Safety-critical deployments |

---

## Dataset

**10,000** time-stamped sensor observations from a marine diesel engine at 1-second resolution (~2h 47m).

```mermaid
pie title Class Distribution
    "Normal (65.1%)" : 6507
    "Fuel Injection (5.1%)" : 509
    "Cylinder Pressure Loss (5.0%)" : 498
    "Exhaust Overheating (4.9%)" : 488
    "Bearing/Vibration (4.8%)" : 481
    "Lube Oil Degradation (5.0%)" : 499
    "Turbocharger Failure (5.2%)" : 519
    "Mixed Fault (5.0%)" : 499
```

### Sensor Channels (18 features)

| Category | Sensors |
|---|---|
| **Combustion** | Cylinder Pressure (×4) |
| **Exhaust** | Exhaust Gas Temperature (×4) |
| **Vibration** | Vibration X, Y, Z |
| **Lubrication** | Oil Temperature, Oil Pressure |
| **Operational** | Shaft RPM, Engine Load, Torque |
| **Efficiency** | Fuel Flow, Air Pressure |
| **Environment** | Ambient Temperature |

---

## Feature Engineering

The raw 18 sensors are expanded into **89 engineered features** through four pipelines:

```mermaid
flowchart TD
    A["Raw Sensors\n18 features"] --> B["Rolling Statistics\nmean, std, min, max\n(10s window)"]
    A --> C["Domain Features\ninter-cylinder deviations\nrate of change, fuel-to-load ratio"]
    A --> D["FFT Spectral Features\nlow/mid/high energy bands\npeak frequency"]
    A --> E["Interaction Features\nVibration Magnitude, Oil Health Index\nEGT stats, Pressure stats"]

    B --> F["Correlation Filter\ndrop |r| > 0.95"]
    C --> F
    D --> F
    E --> F
    F --> G["Final: 89 Features"]

    style E fill:#c8e6c9,stroke:#2e7d32,color:#000
    style G fill:#e3f2fd,stroke:#1565c0,color:#000
```

### Highlight: V2 Interaction Features

| Feature | Formula | Physical Meaning |
|---|---|---|
| Vibration Magnitude | √(Vx² + Vy² + Vz²) | Total vibration energy |
| Oil Health Index | T_oil / P_oil | Oil degradation proxy |
| EGT Max | max(T₁, T₂, T₃, T₄) | Hottest cylinder exhaust |
| EGT Std | σ(T₁, T₂, T₃, T₄) | Cross-cylinder temperature imbalance |
| Cyl Pressure Std | σ(P₁, P₂, P₃, P₄) | Cross-cylinder pressure imbalance |
| Air-to-Fuel Ratio | P_air / F_flow | Combustion efficiency indicator |
| Fuel per RPM | F_flow / RPM | Specific fuel consumption |

**6 of the top 10 most important features (by SHAP) are V2 interaction features** — validating the engineering decisions.

---

## Results

### Per-Fault Detection Performance (V2 Single Model)

```mermaid
xychart-beta
    title "Per-Fault Recall — V2 Single Model"
    x-axis ["Normal", "Fuel Inj", "Cyl Press", "Exh Heat", "Bearing", "Lube Oil", "Turbo", "Mixed"]
    y-axis "Recall" 0 --> 1.05
    bar [0.91, 0.33, 0.52, 0.68, 1.0, 0.97, 1.0, 1.0]
```

### Fault Detectability Hierarchy

```mermaid
graph LR
    subgraph Easy["Easy (≥96% Recall)"]
        E1["Bearing/Vibration\n100%"]
        E2["Turbocharger\n100%"]
        E3["Mixed Fault\n100%"]
        E4["Lube Oil\n97%"]
    end
    subgraph Medium["Medium (65–77%)"]
        M1["Exhaust Overheating\n68%"]
        M2["Cylinder Pressure Loss\n52%"]
    end
    subgraph Hard["Hard (<50%)"]
        H1["Fuel Injection\n33%"]
    end

    Easy ~~~ Medium ~~~ Hard

    style Easy fill:#c8e6c9,stroke:#2e7d32,color:#000
    style Medium fill:#fff9c4,stroke:#f9a825,color:#000
    style Hard fill:#ffcdd2,stroke:#c62828,color:#000
```

### V1 → V2 Improvement

| Metric | V1 Best | V2 Best | Improvement |
|---|---|---|---|
| **Macro F1** | 0.7302 | **0.8040** | **+10.1%** |
| Normal Recall | 0.5739 | **0.9133** | **+59.1%** |
| Macro Recall | 0.8352 | 0.8331 | −0.25% |

---

## Explainability

SHAP (SHapley Additive exPlanations) provides transparent, physically interpretable explanations for every prediction.

### Top 10 Global Feature Importances

```mermaid
mindmap
  root(("SHAP\nTop Features"))
    Vibration Magnitude
      0.238
      Bearing faults
    EGT Max
      0.222
      Exhaust faults
    Air Pressure
      0.195
      Turbocharger faults
    Oil Health Index
      0.187
      Lube oil faults
    Vibration Z
      0.114
      Mixed faults
    Cyl Pressure Std
      0.114
      Pressure loss
    Mean Cyl Pressure
      0.113
      Combustion health
    Fuel Flow
      0.105
      Fuel injection
    Air-to-Fuel Ratio
      0.096
      Combustion efficiency
    Vibration Y
      0.089
      Structural vibration
```

### Diagnostic Signatures

Each fault type has a clear, physics-aligned SHAP signature:

| Fault | Primary SHAP Driver | Physical Explanation |
|---|---|---|
| **Bearing/Vibration** | Vibration X, Y ↑↑ | Degraded bearings cause elevated vibration |
| **Turbocharger** | Air Pressure ↓↓ | Failed turbo can't generate boost pressure |
| **Mixed Fault** | Vibration Z ↑↑ | Compound failure with dominant Z-axis vibration |
| **Lube Oil** | Oil Health Index ↑↑ | High temp + low pressure = degraded lubrication |
| **Exhaust Overheating** | EGT Max, EGT channels ↑ | Elevated exhaust temps across cylinders |
| **Cyl Pressure Loss** | Cyl Pressure Std ↑ | One cylinder loses compression |
| **Fuel Injection** | Fuel Flow ↓ | Disrupted fuel delivery; overlaps with normal variation |

---

## Project Structure

```
├── data/
│   └── marine_engine_fault_dataset.csv   # Raw dataset (10,000 × 20)
├── docs/
│   ├── context.md                        # Project context & planning
│   ├── documentation.md                  # Full technical documentation
│   ├── findings.md                       # Key findings & insights
│   └── report.md                         # Comprehensive project report
├── notebook/
│   └── marine_engine_predictive_maintenance.ipynb  # Main analysis notebook
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook / JupyterLab

### Installation

```bash
git clone https://github.com/MrNahadi/New-AIMS.git
cd New-AIMS
pip install lightgbm shap pandas numpy scipy matplotlib seaborn scikit-learn
```

### Run

```bash
jupyter notebook notebook/marine_engine_predictive_maintenance.ipynb
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built for safer seas through explainable AI</i>
</p>
