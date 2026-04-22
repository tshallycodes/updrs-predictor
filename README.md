# 🧠 UPDRS Predictor — Parkinson's Disease Severity Assessment Tool

> A machine learning system that analyses finger tapping signals to assess Parkinson's Disease severity using the Unified Parkinson's Disease Rating Scale (UPDRS).

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Clinical Background](#clinical-background)
- [System Versions](#system-versions)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Models](#models)
- [Results](#results)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Team](#team)

---

## Project Overview

This project develops an AI-powered clinical assessment tool that classifies Parkinson's Disease severity from finger tapping movement data. By converting raw amplitude and time signals into clinically meaningful features, the system provides an objective, data-driven alternative to subjective clinician scoring.

The tool was developed in **two versions**:

| Version | Task | Classes | Purpose |
|---------|------|---------|---------|
| V1 | Binary Classification | Healthy / Parkinson's | Detect presence or absence of disease |
| V2 | Multi-class Classification | Mild / Moderate / Severe | Assess severity for treatment planning |

---

## Clinical Background

Parkinson's Disease is a progressive neurological disorder characterised by motor symptoms including **bradykinesia** (slowness of movement), **tremors**, and **amplitude decrement** (progressive reduction in movement size). In clinical settings, severity is measured using the **Unified Parkinson's Disease Rating Scale (UPDRS)**, which scores patients from 0 (normal) to 4 (severe impairment).

The **finger tapping test (UPDRS item 3.4)** is a standardised motor assessment where patients rapidly tap their index finger against their thumb. This test captures:

- **Speed** → bradykinesia
- **Rhythm** → motor consistency
- **Amplitude** → movement magnitude over time
- **Pauses** → motor hesitation or freezing

---

## System Versions

### Version 1 — Binary Classification (Presence / Absence)

Establishes a proof-of-concept baseline, classifying patients as either **Healthy** or **Parkinson's**.

**Label encoding:**

| UPDRS Score | Class |
|-------------|-------|
| 0 – 2 | Healthy (0) |
| 3 – 4 | Parkinson's (1) |

**Model trained:** SVC (RBF kernel, `C=0.5`, `class_weight='balanced'`), validated via 5-fold Stratified K-Fold cross-validation, trained independently for left and right hand.

---

### Version 2 — Multi-class Severity Classification

Reformulates the problem into a clinically meaningful 3-class severity task, enabling treatment planning and disease progression monitoring.

**Label encoding:**

| UPDRS Score | Class | Interpretation |
|-------------|-------|----------------|
| 0 – 1 | Mild (0) | Normal to slight impairment |
| 2 – 3 | Moderate (1) | Clear motor dysfunction |
| 4 | Severe (2) | Significant impairment |

**Key improvements over V1:**

- Multi-class severity prediction aligned with UPDRS scoring
- Clinically grounded feature engineering (amplitude decrement, ITI variability, pause detection)
- Savitzky-Golay smoothing filter applied before feature extraction
- Mean-based signal padding (replaces unstable edge/constant padding from V1)
- Mutual information feature selection — top features selected per hand
- Hyperparameter tuning via `RandomizedSearchCV` (sklearn) and Optuna (PyTorch MLP)
- Separate left and right hand modelling to respect Parkinson's motor asymmetry

---

## Dataset

**Source:** Private dataset provided by a University of Bradford researcher specialising in Parkinson's Disease motor analysis  
**Subjects:** ~130 patients (healthy controls and Parkinson's patients)  
**Signals per patient:** Amplitude and time recordings for **left hand** and **right hand** independently  
**Label:** UPDRS score (0–4) per hand

Each patient's data consists of:
- `Amplitude.txt` — finger movement magnitude over time
- `Time.txt` — timestamps of movement

> ⚠️ The dataset is private and is not included in this repository. To request access, please contact the project team directly.

---

## Repository Structure

```
updrs-predictor/
│
├── saved_models/
│   ├── rf_left.pkl                  # Random Forest — left hand
│   ├── rf_right.pkl                 # Random Forest — right hand
│   ├── svc_left.pkl                 # SVC — left hand
│   ├── svc_right.pkl                # SVC — right hand
│   ├── svr_left.pkl                 # SVR — left hand
│   ├── svr_right.pkl                # SVR — right hand
│   ├── cnn_left.pth                 # PyTorch MLP weights — left hand
│   ├── cnn_right.pth                # PyTorch MLP weights — right hand
│   ├── cnn_left_params.json         # Optuna best hyperparameters — left hand
│   ├── cnn_right_params.json        # Optuna best hyperparameters — right hand
│   ├── sklearn_features_left.json   # MI-selected features — left hand
│   └── sklearn_features_right.json  # MI-selected features — right hand
│
├── preprocessing.py                 # Signal loading, padding, smoothing, feature extraction
├── streamlit.py                     # Main Streamlit web application (V2 — severity)
├── signal_store.py                  # Thread-safe singleton for live signal capture
├── hand_landmarker.task             # MediaPipe hand landmark model
├── requirements.txt
├── packages.txt
├── .python-version
└── README.md
```

---

## Pipeline Overview

### 1. Signal Loading
Raw amplitude and time `.txt` files are loaded per patient, per hand.

### 2. Preprocessing
- Signals standardised to **1800 samples** using **mean-based padding**
- **Savitzky-Golay filter** applied (window=7, poly order=2) for noise reduction while preserving peak structure

### 3. Feature Extraction
Nine clinically grounded features extracted from each signal:

| Feature | Clinical Relevance |
|---------|-------------------|
| `num_taps` | Captures bradykinesia — reduced tapping rate |
| `mean_peak_amp` | Average movement amplitude |
| `std_peak_amp` | Amplitude variability |
| `amp_decrement` | Progressive amplitude reduction — hallmark of Parkinson's |
| `mean_iti` | Inter-tap interval — slower = more severe |
| `std_iti` | Rhythm consistency |
| `cv_iti` | Coefficient of variation of tap timing |
| `num_long_pauses` | Count of motor hesitation events |
| `prop_long_pauses` | Proportion of hesitation in total tapping duration |

### 4. Feature Selection
Mutual Information (`SelectKBest`) applied to select the most predictive features per hand, reducing noise and improving generalisation.

### 5. Model Training
Four models trained independently for left and right hand with `StandardScaler` normalisation and hyperparameter tuning.

### 6. Prediction
The Streamlit app accepts amplitude and time values and outputs a severity prediction across all four models simultaneously.

---

## Models

| Model | Type | Tuning Method |
|-------|------|--------------|
| **SVC** | Support Vector Classifier | `RandomizedSearchCV` |
| **SVR** | Support Vector Regressor | `RandomizedSearchCV` |
| **Random Forest** | Ensemble classifier | `RandomizedSearchCV` |
| **PyTorch MLP** | Fully connected neural network | Optuna |

All models trained separately for left and right hand.

---

## Results

### Version 1 — Binary Classification

Evaluated via 5-fold Stratified K-Fold cross-validation:

| Metric | Left Hand | Right Hand |
|--------|-----------|------------|
| Accuracy | 0.82 | 0.79 |
| Weighted F1 | 0.81 | 0.78 |
| ROC-AUC | 0.87 | 0.84 |
| Uncertainty Rate | 0.11 | 0.13 |

---

### Version 2 — Multi-class Severity Classification

Evaluated on held-out test set (20% split):

#### Left Hand

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| SVC | 0.8386 | 0.8884 |
| SVR | 0.9242 | nan |
| Random Forest | 0.9394 | 0.9105 |
| PyTorch MLP | 0.9286 | 0.8942 |

#### Right Hand

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| SVC | 0.9242 | 0.8878 |
| SVR | 0.9242 | nan |
| Random Forest | 0.9242 | 0.8878 |
| PyTorch MLP | 0.9286 | 0.8942 |


**Key findings:**

- Traditional ML (SVC, SVR, RF) consistently outperforms the MLP — the dataset is too small for deep learning to be effective
- The Severe class (UPDRS 4) is never predicted by any model due to extreme class imbalance — very few Severe cases exist in the dataset
- Performance gains in V2 came primarily from **feature engineering and feature selection**, not model complexity
- Bilateral modelling (separate left/right) captures Parkinson's motor asymmetry, which a combined model would obscure

---

## Installation & Setup

### Prerequisites
- Python 3.12
- pip

### Install Dependencies

```bash
git clone https://github.com/tshallycodes/updrs-predictor.git
cd updrs-predictor
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run streamlit.py
```

Once running, navigate to `http://localhost:8501` in your browser.

**How to use:**
1. Select which **hand** to assess — Left or Right
2. Paste your **amplitude values** (comma separated) into the amplitude field
3. Paste your **time values** (comma separated) into the time field
4. Click **Predict** — all four models run simultaneously and return severity predictions with confidence scores and probability breakdowns per class

---

## Limitations

- **Small dataset (~130 samples):** Limits model generalisation, particularly for the neural network
- **Class imbalance:** Severe cases (UPDRS 4) are heavily underrepresented — no model reliably predicts this class
- **Webcam integration not deployed:** A live video capture version using MediaPipe hand tracking was developed locally but could not be deployed on Streamlit Cloud due to native library constraints
- **No left-right fusion:** Predictions are made independently per hand; no combined bilateral decision strategy is implemented
- **Not clinically validated:** This tool is a research prototype and is not intended for clinical use

---

## Future Work

- Address class imbalance using SMOTE or class-weighted loss functions
- Collect more Severe (UPDRS 4) cases to improve rare class prediction
- Implement bilateral fusion — combine left and right hand predictions into a single severity score
- Deploy the webcam-based live signal capture version on a dedicated server environment
- Explore frequency-domain features (FFT) for richer signal representation
- Validate on an external dataset to assess generalisation

---

## Supporting Research

- Iosa et al. (2020) — [Hand Resting Tremor Assessment Using SVC, RF, and Bilateral Design](https://pmc.ncbi.nlm.nih.gov/articles/PMC7381229/) — *Frontiers in Bioengineering and Biotechnology*
- Kodali et al. (2023) — [Automatic Classification of Parkinson's Disease Severity Level](https://www.sciencedirect.com/science/article/pii/S0885230823000670) — *Computer Speech and Language*
- Benmalek et al. (2018) — [UPDRS-Based Multiclass Classification with Small Datasets](https://dl.acm.org/doi/10.1007/s10772-017-9401-9) — *International Journal of Speech Technology*

---

## Team

Developed as part of the Discipline-Specific AI Project module at the University of Bradford.

| Name |
|------|
| Chukwuebuka Tshally-Okeke |
| Avyandra Shahi |
| Elvis Odinkor |
| Ryan Kioko |

---

> **Disclaimer:** This tool is a university research project. It is not a validated medical device and should not be used for clinical diagnosis or treatment decisions.
