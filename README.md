# eeg-seizure-detection
EEG seizure detection using machine learning (PhysioNet dataset)
# EEG Seizure Detection using Machine Learning

##  Overview

This project develops a machine learning pipeline to detect epileptic seizures from multi-channel EEG recordings using the CHB-MIT dataset (PhysioNet).

The goal is to transform raw neurophysiological signals into clinically meaningful predictions, addressing the challenge of identifying seizure events from noisy EEG data.

---

## STEPS TAKEN

### 1. EEG Signal Processing

* Loaded multi-channel EEG data (.edf format) using MNE
* Applied bandpass filtering (1–40 Hz) to remove noise and artefacts

### 2. Segmentation

* Divided continuous EEG signals into 2-second non-overlapping windows
* Each window treated as an independent sample

### 3. Feature Extraction

Extracted statistical features for each channel:

* Mean
* Variance
* Root Mean Square (RMS)
* Peak-to-peak amplitude
* Zero-crossing rate

→ Total: 115 features per segment (23 channels × 5 features)

### 4. Labelling

* Parsed seizure start and end times from clinical summary files
* Labelled segments based on overlap with seizure intervals

### 5. Handling Class Imbalance

* Seizure data represents ~1–2% of dataset
* Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance classes

### 6. Machine Learning Model

* Random Forest classifier (200 trees)
* Used class weighting to prioritise seizure detection
* Lowered decision threshold to improve sensitivity

### 7. Post-processing

* Applied moving average smoothing to predicted probabilities
* Used threshold-based classification to detect seizure events

---

##  Results

* Accuracy: ~96%
* Seizure Recall: ~40%
* Precision: ~22%

---

##  Model Output

<img width="1400" height="700" alt="image" src="https://github.com/user-attachments/assets/14b672d9-baad-4bfb-8262-e826eb2df7ed" />


---

##  Interpretation

The top plot compares true seizure labels with predicted labels. The model successfully detects some seizure events but fails to identify **all** occurrences, indicating moderate sensitivity.

The bottom plot shows the predicted seizure probability over time, with orange lines indicating true seizure intervals. The model produces higher probability spikes during some seizure events, demonstrating that it is capturing meaningful signal patterns.

However, there are also multiple high-probability spikes outside seizure periods, indicating a** high false positive rate**. This reflects a key challenge in EEG-based seizure detection: balancing sensitivity (detecting seizures) with specificity (avoiding false alarms).

Overall, the model demonstrates that simple statistical features can capture some seizure-related patterns, but more advanced approaches are needed for reliable clinical performance.

---

##  Future Improvements

* Frequency-domain features (e.g. band power in alpha, beta bands)
* Temporal models (LSTM / sequence-based learning)
* Patient-specific modelling
* Real-time seizure detection systems

---

##  Dataset

CHB-MIT Scalp EEG Database (PhysioNet)

---

##  Technologies Used

* Python
* MNE (EEG processing)
* NumPy / SciPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib

---


