# =========================
# MULTI-FILE EEG PIPELINE (IMPROVED)
# =========================

import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

summary_file = "chb01-summary.txt"

eeg_files = [
    "chb01_03.edf",
    "chb01_04.edf",
    "chb01_15.edf",
    "chb01_16.edf",
    "chb01_18.edf",
    "chb01_21.edf",
    "chb01_26.edf"
]

# =========================
# LOADING SEIZURE TIMES
# =========================

def load_all_seizure_times(summary_path):
    all_times = {}
    current_file = None
    num_seizures = 0
    start = None

    with open(summary_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("File Name:"):
                current_file = line.split(":", 1)[1].strip()
                num_seizures = 0
                start = None
                continue

            if current_file is None:
                continue

            if line.startswith("Number of Seizures in File:"):
                num_seizures = int(line.split(":", 1)[1].strip())
                continue

            if num_seizures == 0:
                continue

            if "Seizure Start Time:" in line:
                start = float(line.split(":", 1)[1].replace("seconds", "").strip())
                continue

            if "Seizure End Time:" in line and start is not None:
                end = float(line.split(":", 1)[1].replace("seconds", "").strip())
                all_times.setdefault(current_file, []).append((start, end))
                start = None

    return all_times


all_seizure_times = load_all_seizure_times(summary_file)

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(segment):
    features = []
    for channel in segment:
        features.append(np.mean(channel))
        features.append(np.var(channel))
        features.append(np.sqrt(np.mean(channel ** 2)))
        features.append(np.ptp(channel))
        zc = np.sum(np.diff(np.sign(channel)) != 0) / len(channel)
        features.append(zc)
    return features


# =========================
# DATA COLLECTION
# =========================

all_segments = []
all_labels = []

for file_path in eeg_files:

    print(f"\nProcessing {file_path}...")

    if not os.path.exists(file_path):
        print(f"⚠️ Missing: {file_path}")
        continue

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Fix duplicate channels
    seen = {}
    rename_map = {}
    for ch in raw.ch_names:
        if ch in seen:
            seen[ch] += 1
            rename_map[ch] = f"{ch}_{seen[ch]}"
        else:
            seen[ch] = 0
    if rename_map:
        raw.rename_channels(rename_map)

    raw.filter(1., 40., verbose=False)

    data = raw.get_data()
    sfreq = raw.info['sfreq']

    seizure_times = all_seizure_times.get(file_path, [])

    window_size = int(2 * sfreq)

    for i in range(0, data.shape[1] - window_size, window_size):

        segment = data[:, i:i + window_size]

        start_t = i / sfreq
        end_t = start_t + 2.0

        label = 0
        for (sz_start, sz_end) in seizure_times:
            if end_t > sz_start and start_t < sz_end:
                label = 1
                break

        all_segments.append(segment)
        all_labels.append(label)


print("\nTOTAL DATASET")
print(f"Total segments   : {len(all_segments)}")
print(f"Seizure segments : {sum(all_labels)}")

# =========================
# FEATURES
# =========================

X = np.array([extract_features(seg) for seg in all_segments])
y = np.array(all_labels)

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# SMOTE: Synthetic Minority Over-sampling Technique
# =========================

if sum(y_train) > 3:
    smote = SMOTE(random_state=42, k_neighbors=min(3, sum(y_train) - 1))
    X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_proba = model.predict_proba(X_test)[:, 1]

# 🔥 smoothing (IMPORTANT)
window = 5
y_proba_smooth = np.convolve(y_proba, np.ones(window)/window, mode='same')

threshold = 0.2
y_pred = (y_proba_smooth >= threshold).astype(int)

# =========================
# RESULTS
# =========================

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# PLOTS 
# =========================

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# TRUE vs PREDICTED
axes[0].plot(y_test[:200], label="True", alpha=0.7)
axes[0].plot(y_pred[:200], label="Predicted", linestyle="--")
axes[0].set_title("True vs Predicted Labels")
axes[0].legend()

# PROBABILITY
axes[1].plot(y_proba_smooth, label="Smoothed Probability")
axes[1].axhline(threshold, color='red', linestyle='--', label="Threshold")

# Highlight real seizures
for i, val in enumerate(y_test):
    if val == 1:
        axes[1].axvspan(i-0.5, i+0.5, color='orange', alpha=0.3)

axes[1].set_title("Seizure Probability (orange = true seizure)")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/seizure_plot.png")

plt.show()