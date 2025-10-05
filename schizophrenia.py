import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# reproducibility (best-effort)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
normal_path = r"C:\Users\HP\Downloads\norm"        # files ending with .eea
schizo_path = r"C:\Users\HP\Downloads\sch"

# quick print
print("Normal files:", len([f for f in os.listdir(normal_path) if f.endswith('.eea')]))
print("Schizo files:", len([f for f in os.listdir(schizo_path) if f.endswith('.eea')]))
def read_signal(filepath):
    return np.loadtxt(filepath)   # your files are plain numeric text

def get_lengths(folder):
    lengths = []
    files = [f for f in os.listdir(folder) if f.endswith('.eea')]
    for f in files:
        s = read_signal(os.path.join(folder, f))
        lengths.append(len(s))
    return np.array(lengths), files

lens_norm, files_norm = get_lengths(normal_path)
lens_scz, files_scz = get_lengths(schizo_path)

print("Normal lengths summary:", np.min(lens_norm), np.median(lens_norm), np.max(lens_norm))
print("Schizophrenic lengths summary:", np.min(lens_scz), np.median(lens_scz), np.max(lens_scz))
FIXED_LEN = 500  # choose (e.g., 500). You can also set FIXED_LEN = min(all_lengths) after inspecting.

def load_eeg_fixed(folder_path, label, fixed_len=FIXED_LEN):
    X, y, ids = [], [], []
    for fname in os.listdir(folder_path):
        if not fname.endswith('.eea'):
            continue
        path = os.path.join(folder_path, fname)
        sig = read_signal(path).astype(np.float32)
        # normalize per-signal (z-score)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        # pad/trim
        if len(sig) > fixed_len:
            sig = sig[:fixed_len]
        elif len(sig) < fixed_len:
            sig = np.pad(sig, (0, fixed_len - len(sig)), 'constant')
        X.append(sig)
        y.append(label)
        ids.append(fname)  # file/subject id
    return np.array(X), np.array(y), np.array(ids)

Xn, yn, idn = load_eeg_fixed(normal_path, 0)
Xs, ys, ids = load_eeg_fixed(schizo_path, 1)
X = np.vstack([Xn, Xs])
y = np.hstack([yn, ys])
ids_all = np.hstack([idn, ids])

# reshape for keras: (samples, timesteps, channels)
X = X.reshape((X.shape[0], X.shape[1], 1))
print("X shape:", X.shape, "y shape:", y.shape)
WINDOW = 256        # window length in samples
STRIDE = 64         # step between windows (overlap = WINDOW - STRIDE)

def windows_from_file(sig, window=WINDOW, stride=STRIDE):
    segs = []
    for start in range(0, len(sig) - window + 1, stride):
        seg = sig[start:start + window]
        segs.append(seg)
    return np.array(segs)  # shape = (num_windows, window)

def load_eeg_windows(folder_path, label, window=WINDOW, stride=STRIDE):
    X_list, y_list, subj_list = [], [], []
    for fname in os.listdir(folder_path):
        if not fname.endswith('.eea'):
            continue
        path = os.path.join(folder_path, fname)
        sig = read_signal(path).astype(np.float32)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        segs = windows_from_file(sig, window, stride)
        if len(segs) == 0:
            # If signal shorter than window, pad to window once
            padded = np.pad(sig, (0, max(0, window - len(sig))), 'constant')[:window]
            segs = np.expand_dims(padded, axis=0)
        X_list.append(segs)
        y_list.append(np.ones(segs.shape[0], dtype=int) * label)
        subj_list.extend([fname]*segs.shape[0])
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    subj_ids = np.array(subj_list)
    return X, y, subj_ids

# load windows for both classes
Xn_w, yn_w, idn_w = load_eeg_windows(normal_path, 0)
Xs_w, ys_w, ids_w = load_eeg_windows(schizo_path, 1)
X_w = np.vstack([Xn_w, Xs_w])         # (N_samples, WINDOW)
y_w = np.hstack([yn_w, ys_w])
ids_w = np.hstack([idn_w, ids_w])

# reshape for keras
X_w = X_w.reshape((X_w.shape[0], X_w.shape[1], 1))
print("Windowed X shape:", X_w.shape, "y shape:", y_w.shape)
es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
# model checkpoint will be added inside training loop per-model
USE_WINDOWS = True

if USE_WINDOWS:
    X_data, y_data = X_w, y_w
else:
    X_data, y_data = X, y

# stratified split (note: if using windows, windows from same subject may leak -> do group-split instead)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, stratify=y_data, random_state=seed
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# build model and train
input_shape = X_train.shape[1:]
model = build_cnn(input_shape)   # or build_lstm(input_shape)

checkpoint_path = "best_model.h5"
mc = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=[es, reduce_lr, mc],
    verbose=2
),
