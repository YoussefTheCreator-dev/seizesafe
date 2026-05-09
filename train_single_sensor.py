# ============================================================
#  URIC Research: Clinical LOSO Validation FoG Training
#  Optimized with LOSO CV, Jerk, and Tuned Gradient Boosting
# ============================================================

import numpy as np
import pandas as pd
import glob
import os
import joblib
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
DATASET_PATH = r"dataset_fog_release\dataset"
WINDOW_SIZE = 256
WINDOW_OVERLAP = 128
SENSOR_COLS = ['ankle_x', 'ankle_y', 'ankle_z'] 

def extract_axis_features(sig):
    sig_detrend = sig - np.mean(sig)
    f = [np.mean(sig), np.std(sig), np.sqrt(np.mean(sig**2)), np.max(sig) - np.min(sig)]
    jerk = np.diff(sig)
    f.extend([np.mean(jerk), np.std(jerk)])
    freq_values = np.abs(fft(sig_detrend))
    low_band = np.sum(freq_values[3:15]**2)
    high_band = np.sum(freq_values[15:41]**2)
    fi = high_band / low_band if low_band > 0 else 0
    f.extend([fi, np.sum(freq_values**2)])
    autocorr1 = np.corrcoef(sig[1:], sig[:-1])[0, 1] if np.std(sig) > 1e-6 else 0
    f.append(autocorr1)
    return f

def extract_features_per_axis(window):
    all_features = []
    for col in SENSOR_COLS:
        all_features.extend(extract_axis_features(window[col].values))
    mag = np.sqrt(window['ankle_x']**2 + window['ankle_y']**2 + window['ankle_z']**2).values
    all_features.extend(extract_axis_features(mag))
    corr_xy = np.corrcoef(window['ankle_x'], window['ankle_y'])[0, 1] if np.std(window['ankle_x']) > 1e-6 and np.std(window['ankle_y']) > 1e-6 else 0
    all_features.append(corr_xy)
    peaks, _ = find_peaks(mag, distance=20, prominence=200)
    all_features.append(len(peaks))
    return np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

# --- LOAD BY SUBJECT ---
print("🔍 Loading Ankle data from DAPHNET (Subject-Aware)...")
all_files = glob.glob(os.path.join(DATASET_PATH, "*.txt"))
cols = ['time', 'ankle_x', 'ankle_y', 'ankle_z', 'l_x', 'l_y', 'l_z', 't_x', 't_y', 't_z', 'label']

X, y, groups = [], [], []

for f in sorted(all_files):
    # Subject ID from filename (e.g., S01R01.txt -> 1)
    subject_id = int(os.path.basename(f)[1:3])
    tmp = pd.read_csv(f, sep=' ', header=None, names=cols)
    tmp = tmp[tmp['label'] != 0] # Remove label 0
    
    if len(tmp) < WINDOW_SIZE: continue
    
    i = 0
    while i + WINDOW_SIZE <= len(tmp):
        win = tmp.iloc[i : i + WINDOW_SIZE]
        labels_in_window = win['label'].values
        fog_ratio = np.sum(labels_in_window == 2) / len(labels_in_window)
        label = 2 if fog_ratio >= 0.3 else 1
        
        X.append(extract_features_per_axis(win[SENSOR_COLS]))
        y.append(label)
        groups.append(subject_id)
        i += WINDOW_OVERLAP
    print(f"   Processed Subject {subject_id:02d} ({os.path.basename(f)})")

X, y, groups = np.array(X), np.array(y), np.array(groups)
print(f"\n📈 Total Windows: {len(X)} | Subjects: {len(np.unique(groups))}")

# --- LEAVE-ONE-SUBJECT-OUT CROSS VALIDATION ---
logo = LeaveOneGroupOut()
scaler = StandardScaler()
f1_scores = []

print("\n🧪 Starting Leave-One-Subject-Out (LOSO) Validation...")

# We'll use the requested tuned parameters
gbc = GradientBoostingClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=5, 
    subsample=0.8,
    random_state=42
)

# For the final model save, we'll use a standard split but LOSO tells us the TRUTH
for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    current_test_subject = groups[test_idx][0]
    
    # Check if we have both classes in the test set
    if len(np.unique(y_test)) < 2:
        print(f"   ⚠️ Subject {current_test_subject:02d} has no FoG events. Skipping CV fold.")
        continue

    # SMOTE on training only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Scale
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc = scaler.transform(X_test)
    
    # Train
    gbc.fit(X_train_sc, y_train_res)
    
    # Predict
    y_pred = gbc.predict(X_test_sc)
    f1 = f1_score(y_test, y_pred, pos_label=2)
    f1_scores.append(f1)
    print(f"   ✅ Subject {current_test_subject:02d} Tested | F1 (FoG): {f1*100:.2f}%")

avg_f1 = np.mean(f1_scores)
print(f"\n📊 AVERAGE CLINICAL LOSO F1 SCORE: {avg_f1*100:.2f}%")

# --- FINAL TRAIN ON ALL DATA (For Live Use) ---
print("\n🌲 Training Final Production Model on all subjects...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
X_sc = scaler.fit_transform(X_res)
gbc.fit(X_sc, y_res)

joblib.dump(gbc, 'fog_rf_model_single.joblib')
joblib.dump(scaler, 'fog_scaler_single.joblib')
print("\n💾 Clinical-Grade Model Saved: fog_rf_model_single.joblib")
