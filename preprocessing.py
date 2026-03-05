#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# In[111]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ast
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


# # DATA RETRIEVAL

# In[112]:


BASE_DIR = "dataset/"
MODEL_DIR = "models/"
LEFT_HAND_DIR = os.path.join(BASE_DIR, "left")
RIGHT_HAND_DIR = os.path.join(BASE_DIR, "right")
#UPDRS_DIR = os.path.join(BASE_DIR, "csv")
UPDRS_FILE = os.path.join(BASE_DIR, "updrs_scores.csv")


# In[113]:


#Load UPDRS Scores
updrs_df = pd.read_csv(UPDRS_FILE)
updrs_df.head()


# In[114]:


#Helper Function to Read Signal Files
def read_signal_file(file_path):
    with open(file_path, "r") as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return values


# # DATA WRANGLING

#  LEFT HAND DATA PREPARATION

# In[115]:


#Process Left Hand Data
left_records = []

for patient_folder in os.listdir(LEFT_HAND_DIR):
    patient_path = os.path.join(LEFT_HAND_DIR, patient_folder)
    # print(patient_path)

    if not os.path.isdir(patient_path):
        continue

    patient_id = patient_folder

    file_text_path = os.path.join(patient_path, "Text files")
    amplitude_path = os.path.join(file_text_path, "Amplitude.txt")
    time_path = os.path.join(file_text_path, "Time.txt")

    if not os.path.exists(amplitude_path) or not os.path.exists(time_path):
        continue

    amplitude_values = read_signal_file(amplitude_path)
    time_values = read_signal_file(time_path)

    updrs_value = updrs_df.loc[
        updrs_df["patient"] == patient_id, "updrs_left"
    ].values

    # if len(updrs_value) == 0:
    #     continue

    updrs_value = updrs_value[0]

    left_records.append({
        "patient_id": patient_id,
        "updrs_left": updrs_value,
        "amplitude": amplitude_values,
        "time": time_values
    })


# In[116]:


#Create Left Hand DataFrame, Pad to 1800 values and Save CSV
left_df = pd.DataFrame(left_records)

# Create target column: 1 = Parkinson's, 0 = Healthy
left_df["target"] = left_df["patient_id"].apply(lambda x: 1 if x.startswith("PD") else 0)

TARGET_LEN = 1800

def pad_signal(amplitude, time, target_len=TARGET_LEN):

    amplitude = np.array(amplitude, dtype=float)
    time = np.array(time, dtype=float)

    # Pad amplitude with its mean
    if len(amplitude) < target_len:
        pad_width = target_len - len(amplitude)
        amp_mean = np.mean(amplitude) if len(amplitude) > 0 else 0.0
        amplitude = np.pad(
            amplitude,
            (0, pad_width),
            mode='constant',
            constant_values=amp_mean
        )

    # Pad time with its mean
    if len(time) < target_len:
        pad_width = target_len - len(time)
        time_mean = np.mean(time) if len(time) > 0 else 0.0
        time = np.pad(
            time,
            (0, pad_width),
            mode='constant',
            constant_values=time_mean
        )

    return amplitude.tolist(), time.tolist()



# Apply to each row
for idx, row in left_df.iterrows():
    amplitude_padded, time_padded = pad_signal(row['amplitude'], row['time'])
    left_df.at[idx, 'amplitude'] = amplitude_padded
    left_df.at[idx, 'time'] = time_padded


left_csv_path = os.path.join(BASE_DIR, "processed/left.csv")

left_df.to_csv(left_csv_path, index=False)


# In[117]:


# Check first row
first_row = left_df.iloc[0]

amplitude_len = len(first_row['amplitude'])
time_len = len(first_row['time'])

print(f"First row amplitude length: {amplitude_len}")
print(f"First row time length: {time_len}")



# LEFT HAND EXPLORATORY DATA ANALYSIS.

# In[118]:


#Basic Statistics
left_df.head()


# In[119]:


left_df.describe()


# In[120]:


fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# -------------------------------
# 1. UPDRS Class Distribution
# -------------------------------
sns.countplot(x="updrs_left", data=left_df, ax=ax[0, 0])
ax[0, 0].set_title("UPDRS Class Distribution (Left Hand)")
ax[0, 0].set_xlabel("UPDRS Label")
ax[0, 0].set_ylabel("Count")

# -------------------------------
# 2. Mean Amplitude vs UPDRS
# -------------------------------
left_amp_means = left_df["amplitude"].apply(np.mean)
sns.boxplot(x=left_df["updrs_left"], y=left_amp_means, ax=ax[0, 1])
ax[0, 1].set_title("Mean Amplitude vs UPDRS")
ax[0, 1].set_xlabel("UPDRS Label")
ax[0, 1].set_ylabel("Mean Amplitude")

# -------------------------------
# 3. Mean Time vs UPDRS
# -------------------------------
left_time_means = left_df["time"].apply(np.mean)
sns.boxplot(x=left_df["updrs_left"], y=left_time_means, ax=ax[0, 2])
ax[0, 2].set_title("Mean Time vs UPDRS")
ax[0, 2].set_xlabel("UPDRS Label")
ax[0, 2].set_ylabel("Mean Time")

# -------------------------------
# 4. Amplitude over Time by UPDRS
# -------------------------------
amp_0 = np.stack(left_df[left_df["updrs_left"] == 0]["amplitude"].values)
amp_1 = np.stack(left_df[left_df["updrs_left"] == 1]["amplitude"].values)

mean_0, std_0 = amp_0.mean(axis=0), amp_0.std(axis=0)
mean_1, std_1 = amp_1.mean(axis=0), amp_1.std(axis=0)

time_axis = np.arange(len(mean_0))

ax[1, 0].plot(time_axis, mean_0, label="Healthy")
ax[1, 0].fill_between(time_axis, mean_0-std_0, mean_0+std_0, alpha=0.2)

ax[1, 0].plot(time_axis, mean_1, label="Parkinson’s")
ax[1, 0].fill_between(time_axis, mean_1-std_1, mean_1+std_1, alpha=0.2)

ax[1, 0].set_title("Mean Amplitude Over Time by UPDRS")
ax[1, 0].set_xlabel("Time Index")
ax[1, 0].set_ylabel("Amplitude")
ax[1, 0].legend()

# -------------------------------
# 5. Amplitude Variability vs UPDRS
# -------------------------------
left_df["amp_std"] = left_df["amplitude"].apply(np.std)
sns.boxplot(x="updrs_left", y="amp_std", data=left_df, ax=ax[1, 1])
ax[1, 1].set_title("Amplitude Variability vs UPDRS")
ax[1, 1].set_xlabel("UPDRS Label")
ax[1, 1].set_ylabel("Amplitude Std")

# -------------------------------
# 6. Tap Timing Variability vs UPDRS
# -------------------------------
def inter_tap_std(time_seq):
    return np.std(np.diff(np.array(time_seq)))

left_df["tap_interval_std"] = left_df["time"].apply(inter_tap_std)
sns.boxplot(x="updrs_left", y="tap_interval_std", data=left_df, ax=ax[1, 2])
ax[1, 2].set_title("Tap Timing Variability vs UPDRS")
ax[1, 2].set_xlabel("UPDRS Label")
ax[1, 2].set_ylabel("Inter-Tap Std")

plt.tight_layout()
plt.show()


# RIGHT HAND DATA PREPARATION

# In[121]:


#Process Right Hand Data
right_records = []

for patient_folder in os.listdir(RIGHT_HAND_DIR):
    patient_path = os.path.join(RIGHT_HAND_DIR, patient_folder)
    # print(patient_path)

    if not os.path.isdir(patient_path):
        continue

    patient_id = patient_folder

    file_text_path = os.path.join(patient_path, "Text files")
    amplitude_path = os.path.join(file_text_path, "Amplitude.txt")
    time_path = os.path.join(file_text_path, "Time.txt")

    if not os.path.exists(amplitude_path) or not os.path.exists(time_path):
        continue

    amplitude_values = read_signal_file(amplitude_path)
    time_values = read_signal_file(time_path)

    updrs_value = updrs_df.loc[
        updrs_df["patient"] == patient_id, "updrs_right"
    ].values

    updrs_value = updrs_value[0]

    right_records.append({
        "patient_id": patient_id,
        "updrs_right": updrs_value,
        "amplitude": amplitude_values,
        "time": time_values
    })



#Create Right Hand DataFrame and Save CSV
right_df = pd.DataFrame(right_records)
right_df["target"] = right_df["patient_id"].apply(lambda x: 1 if x.startswith("PD") else 0)

for idx, row in right_df.iterrows():
    amplitude_padded, time_padded = pad_signal(row['amplitude'], row['time'])
    right_df.at[idx, 'amplitude'] = amplitude_padded
    right_df.at[idx, 'time'] = time_padded

right_csv_path = os.path.join(BASE_DIR, "processed/right.csv")
right_df.to_csv(right_csv_path, index=False)

first_row = right_df.iloc[0]
print(len(first_row['amplitude']), len(first_row['time']))



# **📊 RIGHT HAND Exploratory Data Analysis.**

# In[122]:


right_df.head()


# In[123]:


right_df.describe()


# In[124]:


fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# -------------------------------
# 1. UPDRS Class Distribution
# -------------------------------
sns.countplot(x="updrs_right", data=right_df, ax=ax[0, 0])
ax[0, 0].set_title("UPDRS Class Distribution (Right Hand)")
ax[0, 0].set_xlabel("UPDRS Label")
ax[0, 0].set_ylabel("Count")

# -------------------------------
# 2. Mean Amplitude vs UPDRS
# -------------------------------
right_amp_means = right_df["amplitude"].apply(np.mean)
sns.boxplot(x=right_df["updrs_right"], y=right_amp_means, ax=ax[0, 1])
ax[0, 1].set_title("Mean Amplitude vs UPDRS (Right Hand)")
ax[0, 1].set_xlabel("UPDRS Label")
ax[0, 1].set_ylabel("Mean Amplitude")

# -------------------------------
# 3. Mean Time vs UPDRS
# -------------------------------
right_time_means = right_df["time"].apply(np.mean)
sns.boxplot(x=right_df["updrs_right"], y=right_time_means, ax=ax[0, 2])
ax[0, 2].set_title("Mean Time vs UPDRS (Right Hand)")
ax[0, 2].set_xlabel("UPDRS Label")
ax[0, 2].set_ylabel("Mean Time")

# -------------------------------
# 4. Amplitude over Time by UPDRS
# -------------------------------
amp_0 = np.stack(right_df[right_df["updrs_right"] == 0]["amplitude"].values)
amp_1 = np.stack(right_df[right_df["updrs_right"] == 1]["amplitude"].values)

mean_0, std_0 = amp_0.mean(axis=0), amp_0.std(axis=0)
mean_1, std_1 = amp_1.mean(axis=0), amp_1.std(axis=0)

time_axis = np.arange(len(mean_0))

ax[1, 0].plot(time_axis, mean_0, label="Healthy")
ax[1, 0].fill_between(time_axis, mean_0 - std_0, mean_0 + std_0, alpha=0.2)

ax[1, 0].plot(time_axis, mean_1, label="Parkinson’s")
ax[1, 0].fill_between(time_axis, mean_1 - std_1, mean_1 + std_1, alpha=0.2)

ax[1, 0].set_title("Mean Amplitude Over Time by UPDRS (Right Hand)")
ax[1, 0].set_xlabel("Time Index")
ax[1, 0].set_ylabel("Amplitude")
ax[1, 0].legend()

# -------------------------------
# 5. Amplitude Variability vs UPDRS
# -------------------------------
right_df["amp_std"] = right_df["amplitude"].apply(np.std)
sns.boxplot(x="updrs_right", y="amp_std", data=right_df, ax=ax[1, 1])
ax[1, 1].set_title("Amplitude Variability vs UPDRS (Right Hand)")
ax[1, 1].set_xlabel("UPDRS Label")
ax[1, 1].set_ylabel("Amplitude Std")

# -------------------------------
# 6. Tap Timing Variability vs UPDRS
# -------------------------------
def inter_tap_std(time_seq):
    return np.std(np.diff(np.array(time_seq)))

right_df["tap_interval_std"] = right_df["time"].apply(inter_tap_std)
sns.boxplot(x="updrs_right", y="tap_interval_std", data=right_df, ax=ax[1, 2])
ax[1, 2].set_title("Tap Timing Variability vs UPDRS (Right Hand)")
ax[1, 2].set_xlabel("UPDRS Label")
ax[1, 2].set_ylabel("Inter-Tap Std")

plt.tight_layout()
plt.show()


# **🔄 Comparative Exploratory Data Analysis.**
# 
# 

# In[125]:


# ---------- Prepare data ----------
left_df = left_df.copy()
right_df = right_df.copy()

left_df["hand"] = "Left"
right_df["hand"] = "Right"

# Mean amplitude per patient
left_df["mean_amplitude"] = left_df["amplitude"].apply(np.mean)
right_df["mean_amplitude"] = right_df["amplitude"].apply(np.mean)

combined_df = pd.concat([left_df, right_df], ignore_index=True)

# ---------- Mean amplitude over time ----------
def mean_amplitude_over_time(df):
    # stack padded signals: (n_patients, 1800)
    stacked = np.vstack(df["amplitude"].values)
    return np.mean(stacked, axis=0)

left_mean_curve = mean_amplitude_over_time(left_df)
right_mean_curve = mean_amplitude_over_time(right_df)

# Time axis (ends at 30s as you observed)
time_axis = np.linspace(0, 30, len(left_mean_curve))

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (1) Mean amplitude by hand and target
sns.boxplot(
    data=combined_df,
    x="hand",
    y="mean_amplitude",
    hue="target",
    ax=axes[0]
)
axes[0].set_title("Mean Amplitude by Hand and Target")
axes[0].set_xlabel("Hand")
axes[0].set_ylabel("Mean Amplitude")
axes[0].legend(title="Target (1 = Parkinson’s)")

# (2) Mean amplitude over time by hand
axes[1].plot(time_axis, left_mean_curve, label="Left Hand")
axes[1].plot(time_axis, right_mean_curve, label="Right Hand")
axes[1].set_title("Mean Amplitude Over Time by Hand")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()

plt.tight_layout()
plt.show()


# # Feature Engineering
# 

# In[126]:


# Cell 1 — Load + parse data (binary labels already encoded)
LEFT_CSV_PATH  = "dataset/processed/left.csv"
RIGHT_CSV_PATH = "dataset/processed/right.csv"

left_df = pd.read_csv(LEFT_CSV_PATH)
right_df = pd.read_csv(RIGHT_CSV_PATH)

def _ensure_list(x):
    """
    CSV saves lists as strings, e.g. "[0.1, 0.2, ...]".
    Convert string -> Python list safely.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    return list(x)

for df in [left_df, right_df]:
    df["amplitude"] = df["amplitude"].apply(_ensure_list)
    df["time"] = df["time"].apply(_ensure_list)


# Quick dataset sanity checks
def summarize_lengths(df, name):
    amp_lens = df["amplitude"].apply(len)
    time_lens = df["time"].apply(len)
    print(f"\n{name} — rows: {len(df)}")
    print("Amplitude length summary:\n", amp_lens.describe())
    print("Time length summary:\n", time_lens.describe())

summarize_lengths(left_df, "LEFT")
summarize_lengths(right_df, "RIGHT")

print("\nClass balance (binary labels from rating sheet):")
print("LEFT target counts:\n", left_df["target"].value_counts())
print("RIGHT target counts:\n", right_df["target"].value_counts())



# In[127]:


# Signal preprocessing

def preprocess_signal(
    amplitude,
    window_length=7,
    polyorder=2,
    apply_filter=True
):
    """
    Light preprocessing for finger tapping amplitude signals.

    Parameters:
    - amplitude: list or np.array of raw amplitude values
    - window_length: Savitzky–Golay window (must be odd)
    - polyorder: polynomial order for smoothing
    - apply_filter: whether to apply smoothing

    Returns:
    - processed_amplitude: np.array
    """
    amplitude = np.asarray(amplitude, dtype=float)

    if apply_filter:
        # Safety: window must be < signal length and odd
        if window_length >= len(amplitude):
            window_length = len(amplitude) - 1
        if window_length % 2 == 0:
            window_length += 1

        amplitude = savgol_filter(
            amplitude,
            window_length=window_length,
            polyorder=polyorder
        )

    return amplitude


# In[128]:


# Checking results of singal processing

raw = left_df.loc[0, "amplitude"]
filtered = preprocess_signal(raw)

print("Raw first 10:", raw[:10])
print("Filtered first 10:", filtered[:10])


# In[129]:


# Robust peak detection (ignores padded time issues)

def extract_peaks(
    amplitude,
    time,
    prominence_ratio=0.1,
    min_interval_sec=0.15,
    fallback_duration_sec=30.0
):
    """
    Adaptive peak detection, robust to time arrays that have repeated values
    (e.g., padding). Uses inferred dt from time if valid; otherwise falls back
    to duration/len(amplitude).
    """
    amplitude = np.asarray(amplitude, dtype=float)
    time = np.asarray(time, dtype=float)

    # Try to infer dt from time, but guard against duplicates/padding
    diffs = np.diff(time)
    diffs = diffs[diffs > 0]  # keep only strictly positive steps

    if len(diffs) > 0:
        dt = np.median(diffs)
    else:
        # Fallback: assume fixed duration across 1800 samples
        dt = fallback_duration_sec / len(amplitude)

    # Convert min interval to samples; clamp to at least 1
    min_distance_samples = max(1, int(min_interval_sec / dt))

    # Adaptive prominence
    signal_range = np.max(amplitude) - np.min(amplitude)
    prominence = prominence_ratio * signal_range

    peaks, properties = find_peaks(
        amplitude,
        prominence=prominence,
        distance=min_distance_samples
    )

    peak_amplitudes = amplitude[peaks]
    peak_times = time[peaks]  # OK even if time has repeats; we handle ITIs later safely

    return peaks, peak_amplitudes, peak_times


# In[130]:


# Checking the results of peak detection

idx = 0  # try a few different indices later
amp = preprocess_signal(left_df.loc[idx, "amplitude"])
t = np.array(left_df.loc[idx, "time"])

peaks, peak_amps, peak_times = extract_peaks(amp, t)

print("Number of taps detected:", len(peaks))
print("First 5 peak amplitudes:", peak_amps[:5])
print("First 5 ITIs:", np.diff(peak_times)[:5])


# In[131]:


# Robust feature extraction (ITI from indices if needed)

def extract_features_from_signal(
    amplitude,
    time,
    early_frac=0.2,
    min_early_peaks=3,
    fallback_duration_sec=30.0
):
    amp = preprocess_signal(amplitude)
    t = np.asarray(time, dtype=float)

    peaks, peak_amps, peak_times = extract_peaks(amp, t, fallback_duration_sec=fallback_duration_sec)

    if len(peak_amps) < 5:
        return {
            "num_taps": np.nan,
            "mean_peak_amp": np.nan,
            "std_peak_amp": np.nan,
            "amp_decrement": np.nan,
            "mean_iti": np.nan,
            "std_iti": np.nan,
            "cv_iti": np.nan,
            "num_long_pauses": np.nan,
            "prop_long_pauses": np.nan,
        }

    features = {}
    features["num_taps"] = len(peak_amps)

    features["mean_peak_amp"] = np.mean(peak_amps)
    features["std_peak_amp"] = np.std(peak_amps)

    n_early = max(min_early_peaks, int(early_frac * len(peak_amps)))
    early_mean = np.mean(peak_amps[:n_early])
    late_mean = np.mean(peak_amps[-n_early:])
    features["amp_decrement"] = early_mean - late_mean

    # Robust dt inference
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if len(diffs) > 0:
        dt = np.median(diffs)
    else:
        dt = fallback_duration_sec / len(amp)

    # ITI from peak index differences (more reliable than time if time has repeats)
    peak_index_diffs = np.diff(peaks)
    itis = peak_index_diffs * dt

    features["mean_iti"] = np.mean(itis)
    features["std_iti"] = np.std(itis)
    features["cv_iti"] = features["std_iti"] / features["mean_iti"] if features["mean_iti"] > 0 else np.nan

    median_iti = np.median(itis)
    long_pauses = itis > (1.5 * median_iti)
    features["num_long_pauses"] = int(np.sum(long_pauses))
    features["prop_long_pauses"] = float(np.mean(long_pauses))

    return features


# In[132]:


# Checking the results of feature extraction

feat_test = extract_features_from_signal(
    left_df.loc[0, "amplitude"],
    left_df.loc[0, "time"]
)

feat_test


# In[133]:


# Cell 5 — Build feature tables (binary labels already encoded)

def build_feature_table(df, hand="left"):
    """
    Build a feature table from a dataframe containing
    amplitude, time, updrs_left/updrs_right and target columns.
    """
    feature_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {hand} features"):
        features = extract_features_from_signal(
            row["amplitude"],
            row["time"]
        )

        # Attach identifiers and labels
        features["patient_id"] = row["patient_id"]
        features["hand"] = hand

        if hand == "left":
            features["updrs"] = int(row["updrs_left"])   # binary label from rating sheet
            features["target"] = int(row["target"])
        else:
            features["updrs"] = int(row["updrs_right"])  # binary label from rating sheet
            features["target"] = int(row["target"])

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)

# Build feature tables
left_features_df = build_feature_table(left_df, hand="left")
right_features_df = build_feature_table(right_df, hand="right")

# Sanity checks
print("LEFT features shape:", left_features_df.shape)
print("RIGHT features shape:", right_features_df.shape)

print("\nLEFT feature-table class balance:")
print(left_features_df["target"].value_counts())

print("\nRIGHT feature-table class balance:")
print(right_features_df["target"].value_counts())

left_features_df.tail(15)


# In[134]:


# Find rows where time has duplicate steps / padding artifacts

'''This diagnostic identifies recordings where the time vector is not strictly increasing due to padding with repeated end values (e.g. 30.0s).
These signals are not corrupted; padding occurs after the movement window.
Feature extraction therefore avoids relying on raw time differences and instead uses robust index-based timing.'''

def time_has_issues(t):
    t = np.asarray(t, dtype=float)
    return np.any(np.diff(t) <= 0)

bad_left = left_df[left_df["time"].apply(time_has_issues)]
bad_right = right_df[right_df["time"].apply(time_has_issues)]

print("Bad LEFT rows:", len(bad_left))
print("Bad RIGHT rows:", len(bad_right))


# In[135]:


# FIXED validation plots (ensure both classes show)

features_df = pd.concat([left_features_df, right_features_df], ignore_index=True)

# Make sure target is numeric
features_df["target"] = features_df["target"].astype(int)

key_features = ["num_taps", "amp_decrement", "cv_iti", "mean_peak_amp"]

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, feature in zip(axes, key_features):
    sns.boxplot(
        data=features_df,
        x="target",
        y=feature,
        ax=ax
    )
    ax.set_xticklabels(["Healthy (UPDRS 0–1)", "PD (UPDRS 2–4)"])
    ax.set_title(feature)

plt.tight_layout()
plt.show()


# In[136]:


### UPDRS Compression


# In[137]:


# Reduce the UPDRS Scores from 0 to 4 to 0, 1 = 0, 2, 3 = 1, 4 = 2

left_features_df['updrs'] = left_features_df['updrs'].apply(lambda x: 0 if x == 0 or x == 1 else (1 if x == 2 or x == 3 else 2))
right_features_df['updrs'] = right_features_df['updrs'].apply(lambda x: 0 if x == 0 or x == 1 else (1 if x == 2 or x == 3 else 2))


# In[138]:


right_features_df.head(20)


# In[153]:


# Export feature tables to CSV (for modelling)


LEFT_OUT_PATH  = "dataset/processed/left_hand_features.csv"
RIGHT_OUT_PATH = "dataset/processed/right_hand_features.csv"

left_features_df.to_csv(LEFT_OUT_PATH, index=False)
right_features_df.to_csv(RIGHT_OUT_PATH, index=False)

print("Saved feature tables:")
print(" -", LEFT_OUT_PATH)
print(" -", RIGHT_OUT_PATH)

