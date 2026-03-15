import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.stats import mode
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
from imblearn.over_sampling import SMOTE

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. CNN-LSTM model will be skipped.")

# Configuration
DATA_DIR = os.path.join("data", "sahwa_data", "ankle")
FILES = ["Stand.csv", "Walk.csv", "FastWalk.csv", "Sit.csv", "SitStand.csv", "Stairs.csv", "FoG.csv"]
SAMPLING_RATE = 50  # Hz
WINDOW_SIZE = 256
STRIDE = 128  # 50% overlap

# Label mapping for Ankle
ORIGINAL_LABELS = [0, 1, 2, 3, 4, 5, 7]
LABEL_NAMES = ["Stand", "Walk", "FastWalk", "Sit", "SitStand", "Stairs", "FoG"]
LABEL_MAPPING = {orig: i for i, orig in enumerate(ORIGINAL_LABELS)}
INV_LABEL_MAPPING = {i: name for i, name in enumerate(LABEL_NAMES)}

def compute_spectral_entropy(psd):
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]
    if len(psd_norm) == 0:
        return 0
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return entropy / np.log2(len(psd))

def extract_features(window_data):
    features = {}
    features['mean'] = np.mean(window_data)
    features['std'] = np.std(window_data)
    features['min'] = np.min(window_data)
    features['max'] = np.max(window_data)
    features['rms'] = np.sqrt(np.mean(window_data**2))
    
    centered = window_data - np.mean(window_data)
    zcr = np.where(np.diff(np.sign(centered)))[0]
    features['zcr'] = len(zcr) / len(window_data)
    
    n = len(window_data)
    yf = fft(window_data)
    xf_mag = np.abs(yf[1:n//2])
    freqs = fftfreq(n, 1/SAMPLING_RATE)[1:n//2]
    
    if len(xf_mag) > 0:
        dom_idx = np.argmax(xf_mag)
        features['dom_freq'] = freqs[dom_idx]
        features['spec_entropy'] = compute_spectral_entropy(xf_mag**2)
    else:
        features['dom_freq'] = 0
        features['spec_entropy'] = 0
        
    return features

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_feature_importance(model, feature_names, title, filename):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = 20
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def build_cnn_lstm(input_shape, n_classes):
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("=== Loading Data & Feature Extraction (Ankle) ===")
    all_dfs = []
    for f in FILES:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_dfs.append(df)
        else:
            print(f"Warning: File {f} not found at {path}")
    
    if not all_dfs:
        print("No data found. Exiting.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['accel_magnitude'] = np.sqrt(full_df['ax']**2 + full_df['ay']**2 + full_df['az']**2)
    full_df['gyro_magnitude'] = np.sqrt(full_df['gx']**2 + full_df['gy']**2 + full_df['gz']**2)
    
    signals = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'accel_magnitude', 'gyro_magnitude']
    
    X_handcrafted = []
    X_raw = []
    y = []
    feature_names = []
    
    for start in range(0, len(full_df) - WINDOW_SIZE + 1, STRIDE):
        window = full_df.iloc[start : start + WINDOW_SIZE]
        window_labels = window['label'].values
        m = mode(window_labels, keepdims=True)
        majority_label = m.mode[0]
        
        mapped_label = LABEL_MAPPING.get(majority_label, -1)
        if mapped_label == -1:
            continue
            
        window_features = []
        raw_window = []
        
        is_first = (len(X_handcrafted) == 0)
        
        for sig in signals:
            sig_data = window[sig].values
            raw_window.append(sig_data)
            feats = extract_features(sig_data)
            for feat_name, val in feats.items():
                window_features.append(val)
                if is_first:
                    feature_names.append(f"{sig}_{feat_name}")
        
        X_handcrafted.append(window_features)
        X_raw.append(np.column_stack(raw_window))
        y.append(mapped_label)
        
    X_handcrafted = np.array(X_handcrafted)
    X_raw = np.array(X_raw)
    y = np.array(y)
    
    print(f"Extracted {X_handcrafted.shape[0]} windows.")

    with open('ankle_label_mapping.json', 'w') as f:
        json.dump(LABEL_MAPPING, f, indent=4)
    joblib.dump(LABEL_MAPPING, 'ankle_label_encoder.pkl')
        
    print("=== Splitting Data ===")
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.20, stratify=y, random_state=42
    )
    
    X_hc_train, X_hc_test = X_handcrafted[idx_train], X_handcrafted[idx_test]
    X_raw_train, X_raw_test = X_raw[idx_train], X_raw[idx_test]
    
    hc_scaler = StandardScaler()
    X_hc_train_scaled = hc_scaler.fit_transform(X_hc_train)
    X_hc_test_scaled = hc_scaler.transform(X_hc_test)
    
    raw_scaler = StandardScaler()
    n_train, seq_len, n_channels = X_raw_train.shape
    X_raw_train_flat = X_raw_train.reshape(-1, n_channels)
    raw_scaler.fit(X_raw_train_flat)
    X_raw_train_scaled = raw_scaler.transform(X_raw_train_flat).reshape(n_train, seq_len, n_channels)
    n_test = X_raw_test.shape[0]
    X_raw_test_scaled = raw_scaler.transform(X_raw_test.reshape(-1, n_channels)).reshape(n_test, seq_len, n_channels)

    smote = SMOTE(random_state=42)
    X_hc_train_smote, y_train_smote = smote.fit_resample(X_hc_train_scaled, y_train)
    
    print("=== Training and Evaluation (Ankle) ===")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, 
                                     subsample=0.8, colsample_bytree=0.8, n_jobs=-1, eval_metric='mlogloss')
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        cv_acc, cv_f1 = [], []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_hc_train_scaled, y_train)):
            X_fold_train, X_fold_val = X_hc_train_scaled[train_idx], X_hc_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            X_fold_train_smote, y_fold_train_smote = smote.fit_resample(X_fold_train, y_fold_train)
            model_clone = clone(model)
            model_clone.fit(X_fold_train_smote, y_fold_train_smote)
            preds = model_clone.predict(X_fold_val)
            acc = accuracy_score(y_fold_val, preds)
            f1 = f1_score(y_fold_val, preds, average='macro')
            cv_acc.append(acc); cv_f1.append(f1)
            print(f"  Fold {fold+1} - Acc: {acc:.4f}, Macro F1: {f1:.4f}")
            
        mean_acc = np.mean(cv_acc); mean_f1 = np.mean(cv_f1)
        print(f"  Mean CV Acc: {mean_acc:.4f}, Mean CV Macro F1: {mean_f1:.4f}")
        
        model.fit(X_hc_train_smote, y_train_smote)
        test_preds = model.predict(X_hc_test_scaled)
        
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='macro')
        results[model_name] = {"cv_acc": mean_acc, "test_acc": test_acc, "test_f1": test_f1, "preds": test_preds, "model": model}
        
        print("\n  Test Set Classification Report:")
        print(classification_report(y_test, test_preds, target_names=LABEL_NAMES))

    if TF_AVAILABLE:
        print("\n--- CNN-LSTM ---")
        cv_acc, cv_f1 = [], []
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_dict = {c: w for c, w in zip(classes, weights)}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw_train_scaled, y_train)):
            X_fold_train, X_fold_val = X_raw_train_scaled[train_idx], X_raw_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            cnn_lstm = build_cnn_lstm((seq_len, n_channels), len(LABEL_NAMES))
            cnn_lstm.fit(X_fold_train, y_fold_train, validation_data=(X_fold_val, y_fold_val),
                         epochs=100, batch_size=32, class_weight=class_weights_dict,
                         callbacks=[EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)], verbose=0)
            preds = np.argmax(cnn_lstm.predict(X_fold_val, verbose=0), axis=1)
            acc = accuracy_score(y_fold_val, preds); f1 = f1_score(y_fold_val, preds, average='macro')
            cv_acc.append(acc); cv_f1.append(f1)
            print(f"  Fold {fold+1} - Acc: {acc:.4f}, Macro F1: {f1:.4f}")
            
        mean_acc = np.mean(cv_acc); mean_f1 = np.mean(cv_f1)
        print(f"  Mean CV Acc: {mean_acc:.4f}, Mean CV Macro F1: {mean_f1:.4f}")
        
        final_cnn_lstm = build_cnn_lstm((seq_len, n_channels), len(LABEL_NAMES))
        history = final_cnn_lstm.fit(X_raw_train_scaled, y_train, validation_data=(X_raw_test_scaled, y_test),
                                     epochs=100, batch_size=32, class_weight=class_weights_dict,
                                     callbacks=[EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5)], verbose=1)
        
        plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss'); plt.legend(); plt.title('Loss')
        plt.subplot(1, 2, 2); plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc'); plt.legend(); plt.title('Accuracy')
        plt.tight_layout(); plt.savefig('ankle_training_history.png'); plt.close()
        
        preds = np.argmax(final_cnn_lstm.predict(X_raw_test_scaled), axis=1)
        test_acc = accuracy_score(y_test, preds); test_f1 = f1_score(y_test, preds, average='macro')
        results["CNN-LSTM"] = {"cv_acc": mean_acc, "test_acc": test_acc, "test_f1": test_f1, "preds": preds, "model": final_cnn_lstm}
        
        print("\n  Test Set Classification Report:"); print(classification_report(y_test, preds, target_names=LABEL_NAMES))

    print("\n=== Model Comparison (Ankle) ===")
    print(f"{'Model':<15} | {'CV Acc':<8} | {'Test Acc':<8} | {'Test F1':<8}"); print("-" * 47)
    best_model_name = None; best_f1 = -1
    for name, res in results.items():
        print(f"{name:<15} | {res['cv_acc']:.4f}   | {res['test_acc']:.4f}   | {res['test_f1']:.4f}")
        if res['test_f1'] > best_f1:
            best_f1 = res['test_f1']; best_model_name = name
            
    print(f"\nBest Model: {best_model_name}")
    
    joblib.dump(results["Random Forest"]["model"], 'ankle_rf_model.pkl'); joblib.dump(hc_scaler, 'ankle_rf_scaler.pkl')
    plot_feature_importance(results["Random Forest"]["model"], feature_names, "Top 20 Features - RF (Ankle)", 'ankle_rf_feature_importance.png')

    joblib.dump(results["XGBoost"]["model"], 'ankle_xgb_model.pkl'); joblib.dump(hc_scaler, 'ankle_xgb_scaler.pkl')
    plot_feature_importance(results["XGBoost"]["model"], feature_names, "Top 20 Features - XGBoost (Ankle)", 'ankle_xgb_feature_importance.png')
    
    if TF_AVAILABLE and "CNN-LSTM" in results:
        results["CNN-LSTM"]["model"].save('ankle_cnn_lstm_model.h5')
        joblib.dump(raw_scaler, 'ankle_cnn_lstm_scaler.pkl')
        
    plot_confusion_matrix(y_test, results[best_model_name]["preds"], f"Confusion Matrix - {best_model_name} (Ankle)", 'ankle_confusion_matrix.png')
    print("Done!")

if __name__ == "__main__":
    main()
