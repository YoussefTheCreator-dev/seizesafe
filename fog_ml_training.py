# ============================================================
#  FoG Detection ML Pipeline
#  Using DAPHNET Dataset + Random Forest & CNN
#  For URIC ADU Research Project
# ============================================================

# ── STEP 0: Install dependencies ────────────────────────────
# Run this in terminal first:
# pip install ucimlrepo pandas numpy scikit-learn matplotlib seaborn tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import glob
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("✅ All imports successful")

# ============================================================
#  STEP 1: LOAD DAPHNET DATASET FROM LOCAL FILES
# ============================================================
print("\n📥 Loading DAPHNET dataset from local files...")

# ── UPDATE THIS PATH if your folder is different ────────────
DATASET_PATH = r"daphnet+freezing+of+gait.zip\dataset_fog_release\dataset"

# Column names based on DAPHNET documentation
COL_NAMES = [
    'time',
    'ankle_x', 'ankle_y', 'ankle_z',
    'leg_x',   'leg_y',   'leg_z',
    'trunk_x', 'trunk_y', 'trunk_z',
    'label'
]

# Load all .txt files
all_files = glob.glob(os.path.join(DATASET_PATH, "*.txt"))
if len(all_files) == 0:
    # Try without zip path
    all_files = glob.glob(os.path.join("dataset_fog_release", "dataset", "*.txt"))

print(f"   Found {len(all_files)} data files")
if len(all_files) == 0:
    print("   ❌ No files found! Update DATASET_PATH to where your .txt files are")
    exit()

dfs = []
for f in sorted(all_files):
    try:
        tmp = pd.read_csv(f, sep=' ', header=None, names=COL_NAMES)
        dfs.append(tmp)
        print(f"   Loaded: {os.path.basename(f)} — {len(tmp)} rows")
    except Exception as e:
        print(f"   ⚠️  Skipped {os.path.basename(f)}: {e}")

df_all = pd.concat(dfs, ignore_index=True)
print(f"\n   Total rows: {len(df_all)}")
print(f"   Label distribution: {df_all['label'].value_counts().to_dict()}")

SENSOR_COLS = ['ankle_x','ankle_y','ankle_z','leg_x','leg_y','leg_z',
               'trunk_x','trunk_y','trunk_z']

# Remove label 0 (not part of experiment)
df = df_all[df_all['label'] != 0].reset_index(drop=True)
print(f"\n   After removing non-experiment samples: {len(df)} rows")
print(f"   Class distribution: {Counter(df['label'].values)}")
print(f"\n   Label meanings:")
print(f"   1 = Walking/Standing (no freeze)")
print(f"   2 = Freezing of Gait (FoG)")

# ============================================================
#  STEP 2: VISUALIZE RAW DATA
# ============================================================
print("\n📊 Plotting raw sensor data sample...")

fig, axes = plt.subplots(3, 1, figsize=(14, 8))
fig.suptitle('DAPHNET Raw Sensor Data — First 1000 Samples', fontsize=14)

# Use ankle acceleration channels (most similar to your wrist device)
sample = df.head(1000)
colors = {1: 'green', 2: 'red'}
label_colors = [colors.get(l, 'gray') for l in sample['label']]

axes[0].plot(sample.iloc[:, 0].values, color='steelblue', linewidth=0.8)
axes[0].set_title('Ankle — Horizontal Forward Acceleration')
axes[0].set_ylabel('Acceleration (mg)')

axes[1].plot(sample.iloc[:, 1].values, color='darkorange', linewidth=0.8)
axes[1].set_title('Ankle — Vertical Acceleration')
axes[1].set_ylabel('Acceleration (mg)')

axes[2].plot(sample['label'].values, color='purple', linewidth=1.2)
axes[2].set_title('Label (1=Normal, 2=FoG)')
axes[2].set_ylabel('Label')
axes[2].set_xlabel('Sample index')

plt.tight_layout()
plt.savefig('raw_data_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: raw_data_visualization.png")

# ============================================================
#  STEP 3: FEATURE EXTRACTION (SLIDING WINDOW)
# ============================================================
print("\n🔧 Extracting features with sliding window...")

WINDOW_SIZE    = 128   # 2.56s at 50Hz — matches your device
WINDOW_OVERLAP = 64    # 50% overlap
SENSOR_COLS = ['ankle_x','ankle_y','ankle_z','leg_x','leg_y','leg_z','trunk_x','trunk_y','trunk_z']

def extract_features(window):
    """Extract features from a single window of IMU data."""
    features = []

    for col in SENSOR_COLS:
        sig = window[col].values.astype(float)

        # Time domain features
        features.append(np.mean(sig))           # mean
        features.append(np.std(sig))            # std dev
        features.append(np.sqrt(np.mean(sig**2)))  # RMS
        features.append(np.max(sig) - np.min(sig)) # range
        features.append(np.max(np.abs(sig)))    # peak

        # Zero crossing rate
        zc = np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0)
        features.append(zc / len(sig))

        # Signal energy
        features.append(np.sum(sig**2) / len(sig))

        # Freeze Index (ratio of high-freq to low-freq energy)
        half = len(sig) // 2
        low_energy  = np.sum(sig[:half]**2)
        high_energy = np.sum(sig[half:]**2)
        fi = high_energy / low_energy if low_energy > 0 else 0
        features.append(fi)

    # Cross-channel: total acceleration magnitude (ankle)
    ankle_mag = np.sqrt(
        window[SENSOR_COLS[0]]**2 +
        window[SENSOR_COLS[1]]**2 +
        window[SENSOR_COLS[2]]**2
    ).values
    features.append(np.mean(ankle_mag))
    features.append(np.std(ankle_mag))
    features.append(np.sqrt(np.mean(ankle_mag**2)))

    # Total acceleration magnitude (trunk)
    trunk_mag = np.sqrt(
        window[SENSOR_COLS[6]]**2 +
        window[SENSOR_COLS[7]]**2 +
        window[SENSOR_COLS[8]]**2
    ).values
    features.append(np.mean(trunk_mag))
    features.append(np.std(trunk_mag))
    features.append(np.sqrt(np.mean(trunk_mag**2)))

    return features

# Generate feature names
feat_names = []
for col in SENSOR_COLS:
    for f in ['mean','std','rms','range','peak','zcr','energy','freeze_idx']:
        feat_names.append(f"{col}_{f}")
feat_names += ['ankle_mag_mean','ankle_mag_std','ankle_mag_rms',
               'trunk_mag_mean','trunk_mag_std','trunk_mag_rms']

# Slide window over data
feature_rows = []
label_rows   = []

i = 0
while i + WINDOW_SIZE <= len(df):
    window = df.iloc[i : i + WINDOW_SIZE]

    # Label: majority vote in window (if >30% is FoG, label as FoG)
    labels_in_window = window['label'].values
    fog_ratio = np.sum(labels_in_window == 2) / len(labels_in_window)
    window_label = 2 if fog_ratio >= 0.3 else 1

    feats = extract_features(window[SENSOR_COLS])
    feature_rows.append(feats)
    label_rows.append(window_label)

    i += WINDOW_OVERLAP  # slide by overlap

X = np.array(feature_rows)
y = np.array(label_rows)

print(f"   Windows extracted: {len(X)}")
print(f"   Features per window: {X.shape[1]}")
print(f"   Class distribution: {Counter(y)}")
fog_pct = np.sum(y==2) / len(y) * 100
print(f"   FoG percentage: {fog_pct:.1f}%")

# ============================================================
#  STEP 4: TRAIN/TEST SPLIT + SCALING
# ============================================================
print("\n✂️  Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Train: {X_train.shape[0]} windows")
print(f"   Test:  {X_test.shape[0]} windows")

# ============================================================
#  STEP 5: RANDOM FOREST MODEL
# ============================================================
print("\n🌲 Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',  # handles class imbalance
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sc, y_train)

y_pred_rf = rf.predict(X_test_sc)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1  = f1_score(y_test, y_pred_rf, pos_label=2)

print(f"\n   ✅ Random Forest Results:")
print(f"   Accuracy : {rf_acc*100:.2f}%")
print(f"   F1 (FoG) : {rf_f1*100:.2f}%")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_rf,
      target_names=['Normal (1)', 'FoG (2)']))

# Cross validation
cv_scores = cross_val_score(rf, X_train_sc, y_train, cv=5, scoring='f1')
print(f"   5-Fold CV F1: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ============================================================
#  STEP 6: CNN MODEL
# ============================================================
print("\n🧠 Training CNN...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten,
                                         Dense, Dropout, BatchNormalization)
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical

    # Reshape for CNN: (samples, timesteps, channels)
    # Use raw windowed data instead of handcrafted features
    print("   Preparing raw windows for CNN...")

    X_cnn = []
    y_cnn = []

    i = 0
    while i + WINDOW_SIZE <= len(df):
        window = df.iloc[i : i + WINDOW_SIZE]
        labels_in_window = window['label'].values
        fog_ratio = np.sum(labels_in_window == 2) / len(labels_in_window)
        window_label = 1 if fog_ratio >= 0.3 else 0  # binary: 0=normal, 1=FoG

        X_cnn.append(window[SENSOR_COLS].values)
        y_cnn.append(window_label)
        i += WINDOW_OVERLAP

    X_cnn = np.array(X_cnn, dtype=np.float32)
    y_cnn = np.array(y_cnn)

    # Normalize
    X_cnn = (X_cnn - X_cnn.mean()) / (X_cnn.std() + 1e-8)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn
    )

    y_tr_cat = to_categorical(y_tr, 2)
    y_te_cat = to_categorical(y_te, 2)

    # Build CNN
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu',
               input_shape=(WINDOW_SIZE, len(SENSOR_COLS))),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"   Model parameters: {model.count_params():,}")

    early_stop = EarlyStopping(monitor='val_loss', patience=5,
                               restore_best_weights=True)

    history = model.fit(
        X_tr, y_tr_cat,
        epochs=30,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_pred_cnn_prob = model.predict(X_te)
    y_pred_cnn = np.argmax(y_pred_cnn_prob, axis=1)

    cnn_acc = accuracy_score(y_te, y_pred_cnn)
    cnn_f1  = f1_score(y_te, y_pred_cnn, pos_label=1)

    print(f"\n   ✅ CNN Results:")
    print(f"   Accuracy : {cnn_acc*100:.2f}%")
    print(f"   F1 (FoG) : {cnn_f1*100:.2f}%")
    print(f"\n   Classification Report:")
    print(classification_report(y_te, y_pred_cnn,
          target_names=['Normal', 'FoG']))

    # Save CNN model
    model.save('fog_cnn_model.h5')
    print("   Saved: fog_cnn_model.h5")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('CNN Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('CNN Loss'); ax2.legend()
    plt.savefig('cnn_training_history.png', dpi=150)
    plt.show()
    print("   Saved: cnn_training_history.png")

    cnn_available = True

except ImportError:
    print("   ⚠️  TensorFlow not installed. Skipping CNN.")
    print("   Run: pip install tensorflow")
    cnn_available = False

# ============================================================
#  STEP 7: CONFUSION MATRICES
# ============================================================
print("\n📊 Plotting confusion matrices...")

fig, axes = plt.subplots(1, 2 if cnn_available else 1,
                         figsize=(12 if cnn_available else 6, 5))

if not cnn_available:
    axes = [axes]

# Random Forest confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','FoG'],
            yticklabels=['Normal','FoG'], ax=axes[0])
axes[0].set_title(f'Random Forest\nAcc={rf_acc*100:.1f}% F1={rf_f1*100:.1f}%')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

if cnn_available:
    cm_cnn = confusion_matrix(y_te, y_pred_cnn)
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Normal','FoG'],
                yticklabels=['Normal','FoG'], ax=axes[1])
    axes[1].set_title(f'CNN\nAcc={cnn_acc*100:.1f}% F1={cnn_f1*100:.1f}%')
    axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print("   Saved: confusion_matrices.png")

# ============================================================
#  STEP 8: FEATURE IMPORTANCE (Random Forest)
# ============================================================
print("\n🔍 Top 15 most important features...")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(12, 5))
plt.bar(range(15), importances[indices], color='steelblue')
plt.xticks(range(15), [feat_names[i] for i in indices], rotation=45, ha='right')
plt.title('Top 15 Feature Importances — Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("   Saved: feature_importance.png")

# ============================================================
#  STEP 9: SAVE MODELS + SCALER
# ============================================================
import joblib

joblib.dump(rf,     'fog_rf_model.joblib')
joblib.dump(scaler, 'fog_scaler.joblib')
joblib.dump(feat_names, 'fog_feature_names.joblib')

print("\n💾 Saved:")
print("   fog_rf_model.joblib")
print("   fog_scaler.joblib")
print("   fog_feature_names.joblib")

# ============================================================
#  STEP 10: SUMMARY
# ============================================================
print("\n" + "="*50)
print("  RESULTS SUMMARY")
print("="*50)
print(f"\n  Dataset   : DAPHNET (UCI ML Repository)")
print(f"  Patients  : Parkinson's disease patients")
print(f"  Samples   : {len(df):,} raw samples")
print(f"  Windows   : {len(X):,} (size={WINDOW_SIZE}, overlap={WINDOW_OVERLAP})")
print(f"  Features  : {X.shape[1]} per window")
print(f"\n  Random Forest:")
print(f"    Accuracy : {rf_acc*100:.2f}%")
print(f"    F1 (FoG) : {rf_f1*100:.2f}%")
if cnn_available:
    print(f"\n  CNN:")
    print(f"    Accuracy : {cnn_acc*100:.2f}%")
    print(f"    F1 (FoG) : {cnn_f1*100:.2f}%")
print(f"\n  Next steps:")
print(f"    1. Collect your own device data")
print(f"    2. Validate model on your data")
print(f"    3. Convert RF model to run on ESP32")
print("="*50)
