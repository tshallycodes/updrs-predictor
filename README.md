# 🧠 UPDRS Predictor — Parkinson's Disease Severity Assessment Tool
#To DO- add images, update repo structure and check again

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
- [Team](#team)

---

## Project Overview

This project develops an AI-powered clinical assessment tool that classifies Parkinson's Disease severity from finger tapping movement data. By converting raw amplitude signals into clinically meaningful features, the system provides an objective, data-driven alternative to subjective clinician scoring.

The tool was developed in **two versions**:

| Version | Task | Classes | Purpose |
|---------|------|---------|---------|
| V1 | Binary Classification | Healthy / Parkinson's | Detect presence or absence of disease |
| V2 | Multi-class Classification | Mild / Moderate / Severe | Assess severity for treatment planning |

---

## Clinical Background

Parkinson's Disease is a progressive neurological disorder characterised by motor symptoms including **bradykinesia** (slowness), **tremors**, and **amplitude decrement** (progressive reduction in movement size). In clinical settings, severity is measured using the **Unified Parkinson's Disease Rating Scale (UPDRS)**, which scores patients from 0 (normal) to 4 (severe impairment).

The **finger tapping test (UPDRS item 3.4)** is a standardised motor assessment where patients rapidly tap their index finger against their thumb. This test captures:

- **Speed** → bradykinesia
- **Rhythm** → motor consistency
- **Amplitude** → movement magnitude over time
- **Pauses** → motor hesitation or freezing

![alt text](IMG)
> *Caption: "Healthy individuals show consistent, high-amplitude tapping while Parkinson's patients exhibit progressive amplitude reduction and irregular inter-tap intervals"*

---

## System Versions

### Version 1 — Binary Classification (Presence / Absence)

The first version establishes a proof-of-concept baseline, classifying patients as either **Healthy** or **Parkinson's**.

**Label encoding:**

| UPDRS Score | Class |
|-------------|-------|
| 0 – 2 | Healthy (0) |
| 3 – 4 | Parkinson's (1) |

**Models trained:** SVC, Random Forest, SVR, PyTorch MLP — independently for left and right hand.

---

### Version 2 — Multi-class Severity Classification

Version 2 reformulates the problem into a clinically meaningful 3-class severity task, enabling treatment planning and disease progression monitoring.

**Label encoding:**

| UPDRS Score | Class | Interpretation |
|-------------|-------|----------------|
| 0 – 1 | Mild (0) | Normal to slight impairment |
| 2 – 3 | Moderate (1) | Clear motor dysfunction |
| 4 | Severe (2) | Significant impairment |

> 📸 *[Image: Bar chart of UPDRS class distribution across the dataset]*
> *Caption: "Class distribution showing the imbalance across Mild, Moderate, and Severe categories — Severe cases are underrepresented in the dataset"*

**Key improvements over V1:**

- Multi-class severity prediction aligned with UPDRS scoring
- Clinically grounded feature engineering (amplitude decrement, ITI variability, pause detection)
- Savitzky-Golay smoothing filter applied before feature extraction
- Improved mean-based signal padding (replaces unstable edge/constant padding from V1)
- Separate left and right hand modelling to respect Parkinson's motor asymmetry

> 📸 *[Image: V1 → V2 pipeline evolution diagram]*
> *Caption: "Evolution from binary presence/absence detection (V1) to three-class severity prediction (V2)"*

---

## Dataset

**Source:** Private dataset provided by our project client (a University of Bradford researcher with a PhD specialising in Parkinson's Disease motor analysis)  
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
├── data/                        # Raw dataset (not included — see Dataset section)
│   ├── left/
│   └── right/
│
├── preprocessing/
│   └── preprocessing.py         # Signal loading, padding, smoothing, feature extraction
│
├── models/
│   └── model.py                 # SVC, SVR, Random Forest, PyTorch MLP training
│
├── app/
│   └── streamlit_app.py         # Streamlit web application
│
├── outputs/
│   ├── left.csv                 # Extracted features — left hand
│   └── right.csv                # Extracted features — right hand
│
├── docs/                        # Project documentation
│
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

> 📸 *[Image: End-to-end system pipeline diagram]*
> *Caption: "Full ML pipeline from raw signal input through preprocessing, feature extraction, model training, and Streamlit-based prediction output"*

### 1. Signal Loading
Raw amplitude and time `.txt` files are loaded per patient, per hand.

### 2. Preprocessing
- Signals standardised to **1800 samples** using **mean-based padding**
- **Savitzky-Golay filter** applied (window=7, poly order=2) for noise reduction while preserving peak structure

### 3. Feature Extraction
Nine clinically grounded features are extracted from each signal:

| Feature | Clinical Relevance |
|---------|-------------------|
| `num_taps` | Captures bradykinesia (reduced tapping rate) |
| `mean_peak_amp` | Reflects average movement size |
| `std_peak_amp` | Measures amplitude variability |
| `amp_decrement` | Hallmark of Parkinson's — progressive amplitude reduction |
| `mean_iti` | Inter-tap interval — slower = more severe |
| `std_iti` | Rhythm consistency |
| `cv_iti` | Coefficient of variation of tap timing |
| `num_long_pauses` | Motor hesitation events |
| `prop_long_pauses` | Proportion of hesitation in total tapping |

### 4. Model Training
Four models trained independently on left and right hand data with an 80/20 train-test split and StandardScaler normalisation.

### 5. Prediction
The Streamlit app accepts an uploaded signal file and outputs a severity prediction.

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| **SVC** | Support Vector Classifier | Strong performance on small datasets |
| **SVR** | Support Vector Regressor | Adapted for ordinal severity output |
| **Random Forest** | Ensemble classifier | Robust, interpretable feature importance |
| **PyTorch MLP** | Fully connected neural network | Underperformed due to small dataset size |

All models are trained separately for **left hand** and **right hand** to capture Parkinson's motor asymmetry.

---

## Results

### Version 2 Model Performance (Multi-class)

| Model | Accuracy | Notes |
|-------|----------|-------|
| SVC | ~0.71 | Best on left hand |
| SVR | ~0.71 | Comparable to SVC |
| Random Forest | ~0.71 | Best on right hand |
| PyTorch MLP | ~0.57 | Underperformed — insufficient data for deep learning |

> 📸 *[Image: Model performance comparison bar chart]*
> *Caption: "Comparison of model accuracy across four models — traditional ML consistently outperforms the neural network on this small dataset"*

> 📸 *[Image: Confusion matrices for best models — left and right hand]*
> *Caption: "Confusion matrices showing model predictions biased towards Mild and Moderate classes, with Severe cases absent from the evaluation set due to class imbalance"*

**Key finding:** Performance gains in V2 came primarily from **feature engineering**, not model complexity. The MLP underperformed because the dataset is too small for deep learning and features were already pre-engineered, removing the advantage of raw signal learning.

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies

```bash
git clone https://github.com/tshallycodes/updrs-predictor.git
cd updrs-predictor
pip install -r requirements.txt
```

### Requirements

```
streamlit
scikit-learn
torch
numpy
pandas
scipy
matplotlib
seaborn
```

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

**How to use:**
1. Select **Version 1** (binary: Healthy/Parkinson's) or **Version 2** (severity: Mild/Moderate/Severe)
2. Select which **hand** to assess (left or right)
3. Upload your signal files (Amplitude and Time data — 1800 values each). The app validates inputs and rejects malformed or incorrectly sized files
4. Click **Predict** to receive your result — the app outputs the predicted severity class alongside a **confidence score** and **probability breakdown** across all classes

> 📸 *[Image: Screenshot of the Streamlit app interface]*
> *Caption: "UPDRS Predictor Streamlit interface — users upload finger tapping signal files and receive an instant severity classification"*

---

## Limitations

- **Small dataset (~130 samples):** Limits model generalisation and reliability, particularly for the neural network
- **Class imbalance:** Severe cases (UPDRS 4) are underrepresented, meaning the model cannot reliably predict advanced Parkinson's
- **No cross-validation in V2:** Results depend on a single 80/20 split
- **No left-right fusion:** Predictions are made independently per hand; no combined decision strategy is implemented
- **Not clinically validated:** This tool is a research prototype and is not intended for clinical use

---

## Team

Developed as part of Discipline-Specific AI Project module at the University of Bradford.

| Name |
|------|
| Chukwuebuka Tshally-Okeke |
| Avyandra Shahi |
| Elvis Odinkor |
| Ryan Kioko |

---

> **Disclaimer:** This tool is a university research project. It is not a validated medical device and should not be used for clinical diagnosis or treatment decisions.
