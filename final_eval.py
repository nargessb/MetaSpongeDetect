# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:17:10 2025

@author: narge
"""


import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import skew
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def dynamic_normalization(data):
    """
    Dynamically normalizes the data based on its shape and statistical properties.
    """
    if len(data.shape) == 3:  # If data is 3D (samples, features, time-series/images)
        print("[INFO] Applying channel-wise normalization for 3D data.")
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[1]):  # Normalize each feature/channel separately
            scaler = StandardScaler()
            normalized_data[:, i, :] = scaler.fit_transform(data[:, i, :])
        return normalized_data
    
    elif len(data.shape) == 2:  # If data is 2D (samples, features)
        print("[INFO] Applying feature-wise normalization for 2D data.")
        if np.any(np.abs(skew(data, axis=0)) > 1):  # Check skewness of columns
            print("  - Using RobustScaler due to skewed distribution.")
            scaler = RobustScaler()
        else:
            print("  - Using StandardScaler for normal distribution.")
            scaler = StandardScaler()
        return scaler.fit_transform(data)
    
    else:  # If data has an unknown shape
        raise ValueError(f"[ERROR] Unsupported data shape: {data.shape}")


# Specify the path to the folder containing your CSV files
folder_path = 'C:/Users/narges.babadi1/Downloads/Sponge Project/Evaluation_reults/GTSRB/Resnet18'

# Use glob to get all CSV files in the specified folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Load CSV files into a dictionary of DataFrames
dataframes = {os.path.basename(fp): pd.read_csv(fp) for fp in csv_files}
# print("Loaded CSV files:", list(dataframes.keys()))

# Reference dataset alignment
sponge_keys = [k for k in dataframes if 'Sponge' in k]
if not sponge_keys:
    raise ValueError("No 'Sponge' dataset found.")
df_reference = dataframes[sponge_keys[0]]  # Use first sponge dataset as reference

def align_columns_to_reference(reference_df, df):
    """Aligns the columns of df to match reference_df."""
    common_columns = reference_df.columns.intersection(df.columns)
    return df[common_columns].reindex(columns=reference_df.columns, fill_value=0).fillna(0)

# Align all DataFrames
data = {key: align_columns_to_reference(df_reference, df) for key, df in dataframes.items()}

y = {}
for key in data.keys():
    if '_lb0_' in key:
        y[key] = 'No attack'
    elif 'MatchingGradient' in key:
        y[key] = 'Poisoning attack_n1'
    elif 'Feature' in key:
        y[key] = 'Poisoning attack_n2'
    else:
        y[key] = 'Sponge attack'

# Convert data into a 3D numpy array
X = np.stack([df.values for df in data.values()])
labels = np.array([y[key] for key in data.keys()])

# Convert labels to numerical values
label_mapping = {'Sponge attack': 0, 'No attack': 1, 'Poisoning attack_n2': 2, 'Poisoning attack_n1': 3}
labels_numeric = np.array([label_mapping[label] for label in labels])
for i, key in enumerate(list(data.keys())[:10]):  
    print(f"{key}: {y[key]} → {label_mapping[y[key]]}")

# Identify the smallest class (to be used for evaluation)
unique_labels, counts = np.unique(labels_numeric, return_counts=True)
print("Unique Labels:", set(labels))  
print("Expected Mapping:", label_mapping)  

min_class = unique_labels[np.argmin(counts)]
holdout_scenario = 'Sponge attack'  # Hardcoded selection instead of dynamic selection
data_dict_3d = {lbl: [] for lbl in label_mapping.keys()}
for i, lbl in enumerate(labels_numeric):
    label_str = {v: k for k, v in label_mapping.items()}[lbl]
    data_dict_3d[label_str].append(X[i])

data_dict_3d = {lbl: np.stack(seqs) for lbl, seqs in data_dict_3d.items() if seqs}

#%%%%% Dimensionality Reduction Using PCA 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Convert dictionary values into a single NumPy array
data_array = np.concatenate(list(data_dict_3d.values()), axis=0)  # (num_samples, num_time_steps, num_features)

# Get shape
num_samples, num_time_steps, num_features = data_array.shape

# Reshape data_array to 2D: (samples * time_steps, features)
data_flattened = data_array.reshape(num_samples * num_time_steps, num_features)

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_flattened)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.99)  # Keep 99% variance
data_pca = pca.fit_transform(data_normalized)

# The reduced number of features
n_selected_features = data_pca.shape[1]

# Print PCA variance explained
print(f"Explained variance by PCA components: {np.sum(pca.explained_variance_ratio_):.4f}")

# Ensure reshaping is valid
if data_pca.shape[0] != num_samples * num_time_steps:
    raise ValueError("Mismatch in reshaped PCA data dimensions.")

# Reshape back to 3D: (samples, time_steps, reduced_features)
data_selected_3d = data_pca.reshape(num_samples, num_time_steps, n_selected_features)

print(f"Shape of original data: {data_array.shape}")
print(f"Shape of data after PCA (99% variance): {data_selected_3d.shape}")
print(f"Number of important features: {n_selected_features}")
# Reshape data_pca from (num_samples * num_time_steps, reduced_features) to (num_samples, num_time_steps, reduced_features)
num_samples, num_time_steps, _ = data_array.shape  # Get original shape
n_selected_features = data_pca.shape[1]  # Get number of reduced features

# Reshape back to 3D
data_pca_reshaped = data_pca.reshape(num_samples, num_time_steps, n_selected_features)

# Create a new dictionary to store PCA-transformed data per label
data_dict_3d_pca = {}

# Track the index position for slicing
start_idx = 0

# Reconstruct the dictionary while preserving time-step structure
for label, sequences in data_dict_3d.items():
    num_samples_label = sequences.shape[0]  # Number of samples in this category
    end_idx = start_idx + num_samples_label  # Define range

    # Extract corresponding PCA-transformed data while keeping time steps
    data_dict_3d_pca[label] = data_pca_reshaped[start_idx:end_idx, :, :]  # (samples, time_steps, features)

    start_idx = end_idx  # Update start index for next category

# Confirm the reconstruction
for lbl, arr in data_dict_3d_pca.items():
    print(f"{lbl}: {arr.shape}")  # Should match original (samples, time_steps, reduced_features)

#%%% Task generating
import numpy as np
import numpy as np

def generate_balanced_tasks(data_dict_3d_pca, label_mapping, k_shot=15, query_shots=15):
    """
    Generates balanced tasks for LSTM-MAML training and evaluation.
    Uses oversampling to balance classes within each task.
    
    Args:
        data_dict_3d_pca (dict): Dictionary containing class-wise PCA-transformed samples.
        label_mapping (dict): Dictionary mapping class names to numerical values.
        k_shot (int): Number of support samples per class.
        query_shots (int): Number of query samples per class.

    Returns:
        dict: Dictionary of tasks with support and query sets.
    """
    tasks = {}

    # Define Task Compositions (Ensuring Correct Labels for Each)
    task_definitions = {
        "task0": ["No attack", "Sponge attack"],  # Maximize sample size
        "task1": ["No attack", "Poisoning attack_n1"],
        "task2": ["No attack", "Poisoning attack_n2"],
        "task3": ["Poisoning attack_n2", "Poisoning attack_n1"],
        "task4": ["No attack", "Poisoning attack_n1", "Poisoning attack_n2"],
        "task5": ["Poisoning attack_n1", "Poisoning attack_n2"]
    }

    for task_name, classes in task_definitions.items():
        support_set, query_set = {}, {}
        support_labels, query_labels = {}, {}
        available_classes = [cls for cls in classes if cls in data_dict_3d_pca]
        if len(available_classes) < 2:
            print(f"[SKIPPED] {task_name}: Not enough available classes.")
            continue
        # Find the largest class size in the task for oversampling
        max_samples = max([len(data_dict_3d_pca[cls]) for cls in classes if cls in data_dict_3d_pca])

        for cls in classes:
            if cls not in data_dict_3d_pca:
                print(f"[WARNING] Class {cls} not found in data_dict_3d_pca. Skipping...")
                continue
            
            samples = np.array(data_dict_3d_pca[cls])  # Use PCA-transformed data
            num_samples = len(samples)
            class_label = label_mapping[cls]

            # **Oversampling if needed**
            if num_samples < max_samples:
                print(f"[INFO] Oversampling {cls} (original={num_samples}, target={max_samples}).")
                selected_indices = np.random.choice(num_samples, max_samples, replace=True)
            else:
                selected_indices = np.random.choice(num_samples, max_samples, replace=False)

            # Split into support & query sets (balanced distribution)
            support_size = k_shot if k_shot < max_samples // 2 else max_samples // 2
            query_size = query_shots if query_shots < max_samples - support_size else max_samples - support_size

            support_set[cls] = samples[selected_indices[:support_size]]
            query_set[cls] = samples[selected_indices[support_size:support_size + query_size]]
            support_labels[cls] = np.full(support_size, class_label)
            query_labels[cls] = np.full(query_size, class_label)

        # Store the Task
        tasks[task_name] = {
            "support": support_set, 
            "query": query_set,
            "support_labels": support_labels,
            "query_labels": query_labels
        }

    return tasks


# Generate Balanced Tasks for LSTM-MAML
tasks = generate_balanced_tasks(data_dict_3d_pca, label_mapping, k_shot=10, query_shots=10)

train_tasks = list(tasks.values())[1:]  # Training tasks
test_tasks = [list(tasks.values())[0]]  # Task0 as test

# Check Label Distributions After Fix
for task, data in tasks.items():
    print(f"{task}: Support = {[len(v) for v in data['support'].values()]}, Query = {[len(v) for v in data['query'].values()]}")
    print(f"{task} Support Labels: {np.unique(np.concatenate(list(data['support_labels'].values())))}")
    print(f"{task} Query Labels: {np.unique(np.concatenate(list(data['query_labels'].values())))}")

print(f"[INFO] Generated {len(train_tasks)} training tasks and {len(test_tasks)} test tasks.")






#%%%%% Full MAML

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout
from tensorflow.keras.models import Model

def build_rnn_model(input_shape, num_classes=4, dropout_rate=0.4):
    inputs = Input(shape=input_shape)
    x = SimpleRNN(128, return_sequences=True)(inputs)
    x = SimpleRNN(128)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def load_task(task):
    sx = np.concatenate(list(task["support"].values()))
    sy = np.concatenate(list(task["support_labels"].values()))
    qx = np.concatenate(list(task["query"].values()))
    qy = np.concatenate(list(task["query_labels"].values()))
    return sx, sy, qx, qy

def fomaml_train(
    train_tasks,
    input_shape,
    num_classes=4,
    meta_epochs=20,
    inner_lr=0.01,
    meta_lr=0.001,
    adaptation_steps=3,
    batch_size=16,
):
    """
    First-order MAML (FOMAML) for your RNN.
    - Inner loop: adapt cloned temp_model on support set.
    - Outer loop: gradients of query loss w.r.t. temp_model weights
                  are used as meta-gradients for the base model.
    """

    # Base meta-model θ
    model = build_rnn_model(input_shape, num_classes)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    meta_optimizer = tf.keras.optimizers.Adam(meta_lr)

    accuracy_history = []

    for epoch in range(meta_epochs):
        print(f"\n=== Meta Epoch {epoch+1}/{meta_epochs} ===")

        # Accumulate meta-gradients over tasks
        meta_grads = [tf.zeros_like(w) for w in model.trainable_variables]
        total_acc = 0.0
        num_tasks = 0

        for task in train_tasks:
            num_tasks += 1

            sx, sy, qx, qy = load_task(task)

            # Convert to tensors for outer step
            qx_tf = tf.convert_to_tensor(qx, dtype=tf.float32)
            qy_tf = tf.convert_to_tensor(qy, dtype=tf.int32)

            # ---- Clone base model (start from θ) ----
            temp_model = build_rnn_model(input_shape, num_classes)
            temp_model.set_weights(model.get_weights())
            temp_model.compile(
                optimizer=tf.keras.optimizers.SGD(inner_lr),
                loss=loss_fn,
                metrics=['accuracy']
            )

            # ---- INNER-LOOP: Adaptation on support set ----
            temp_model.fit(
                sx, sy,
                epochs=adaptation_steps,
                batch_size=batch_size,
                verbose=0
            )

            # ---- Evaluate query accuracy (for monitoring) ----
            q_logits_np = temp_model.predict(qx_tf, verbose=0)
            q_preds = np.argmax(q_logits_np, axis=1)
            task_acc = np.mean(q_preds == qy)
            total_acc += task_acc

            # ---- OUTER-LOOP: FOMAML GRADIENT ----
            with tf.GradientTape() as tape:
                # Important: tape watches *temp_model*, not base model
                q_logits = temp_model(qx_tf, training=False)
                q_loss = loss_fn(qy_tf, q_logits)

            grads = tape.gradient(q_loss, temp_model.trainable_variables)

            # Accumulate grads (treat as grads for base model θ)
            safe_grads = []
            for g, w in zip(grads, model.trainable_variables):
                if g is None:
                    g = tf.zeros_like(w)
                safe_grads.append(g)

            meta_grads = [mg + g for mg, g in zip(meta_grads, safe_grads)]

        # ---- META-UPDATE: apply averaged meta-gradients to base model θ ----
        meta_grads = [mg / num_tasks for mg in meta_grads]
        meta_optimizer.apply_gradients(zip(meta_grads, model.trainable_variables))

        avg_acc = total_acc / num_tasks
        accuracy_history.append(avg_acc)
        print(f"Meta Epoch {epoch+1}: Avg Query Accuracy = {avg_acc:.4f}")

    return model, accuracy_history

input_shape = (100, n_selected_features)

final_model, acc_hist = fomaml_train(
    train_tasks,
    input_shape,
    num_classes=4,
    meta_epochs=20,
    inner_lr=0.01,      # you can try 0.005 too
    meta_lr=0.001,
    adaptation_steps=3,
    batch_size=16,
)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_on_task(model, task, input_shape, num_classes=4, fine_tune_epochs=5):
    sx, sy, qx, qy = load_task(task)

    temp_model = build_rnn_model(input_shape, num_classes)
    temp_model.set_weights(model.get_weights())
    temp_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    temp_model.fit(sx, sy, epochs=fine_tune_epochs, batch_size=16, verbose=0)

    y_pred = np.argmax(temp_model.predict(qx, verbose=0), axis=1)

    acc = accuracy_score(qy, y_pred)
    precision = precision_score(qy, y_pred, average='weighted', zero_division=0)
    recall = recall_score(qy, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(qy, y_pred, average='weighted')

    print(f"\n[Unseen Task Evaluation]")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    return acc, precision, recall, f1

print("\n[INFO] Evaluating on Unseen Task (Task 0)...")
evaluate_on_task(final_model, test_tasks[0], input_shape)

#%%% p-values
# =============================================================
#      MULTI-SEED EXPERIMENT FOR FIRST-ORDER MAML RESULTS
# =============================================================

import numpy as np
from scipy.stats import ttest_rel

# =============================================================
# 1. Statistical significance function
# =============================================================
def compute_stats(acc_model, acc_base, paired=True):
    acc_model = np.array(acc_model, dtype=np.float32)
    acc_base  = np.array(acc_base,  dtype=np.float32)

    assert len(acc_model) == len(acc_base), "Accuracy lists must match in length."

    n = len(acc_model)

    # Mean + 95% Confidence Interval
    mean = np.mean(acc_model)
    std  = np.std(acc_model, ddof=1)
    ci95 = 1.96 * std / np.sqrt(n)

    # Paired t-test (recommended for meta-learning)
    if paired:
        t_stat, p_val = ttest_rel(acc_model, acc_base)
    else:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(acc_model, acc_base, equal_var=False)

    # Effect size (Cohen’s d)
    pooled_std = np.sqrt((std**2 + np.std(acc_base, ddof=1)**2) / 2)
    d = (mean - np.mean(acc_base)) / pooled_std if pooled_std > 1e-10 else 0.0

    return mean, ci95, p_val, d


def train_baseline_rnn(train_data, input_shape, num_classes=4, epochs=20):
    model = build_rnn_model(input_shape, num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    x_train = np.concatenate([v for v in train_data["x"]])
    y_train = np.concatenate([v for v in train_data["y"]])

    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model


def evaluate_baseline_on_task(model, task):
    qx = np.concatenate(list(task["query"].values()))
    qy = np.concatenate(list(task["query_labels"].values()))

    preds = np.argmax(model.predict(qx), axis=1)
    return np.mean(preds == qy)

# =============================================================
# 2. Baseline model (Supervised RNN, no meta-learning)
# =============================================================

def build_baseline_rnn(input_shape, num_classes=4, dropout_rate=0.4):
    inputs = Input(shape=input_shape)
    x = SimpleRNN(128, return_sequences=True)(inputs)
    x = SimpleRNN(128)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def train_baseline(train_tasks, input_shape, num_classes=4, epochs=10):
    # Gather ALL support + query data for standard supervised training
    X_train, y_train = [], []

    for task in train_tasks:
        sx = np.concatenate(list(task["support"].values()))
        sy = np.concatenate(list(task["support_labels"].values()))
        qx = np.concatenate(list(task["query"].values()))
        qy = np.concatenate(list(task["query_labels"].values()))

        X_train.append(sx)
        y_train.append(sy)
        X_train.append(qx)
        y_train.append(qy)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # Build & train baseline RNN
    model = build_baseline_rnn(input_shape, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model

def evaluate_baseline(model, task):
    qx = np.concatenate(list(task["query"].values()))
    qy = np.concatenate(list(task["query_labels"].values()))
    preds = np.argmax(model.predict(qx, verbose=0), axis=1)
    return np.mean(preds == qy)
# =============================================================
# 3. Multi-seed experiment loop (FOMAML ONLY)
# =============================================================

NUM_SEEDS = 3

acc_fomaml_seeds = []
precision_fomaml_seeds= []
recall_fomaml_seeds = []
f1_fomaml_seeds =[]

for seed in range(NUM_SEEDS):

    print("\n====================================================")
    print(f"                Running Seed {seed}                ")
    print("====================================================")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # =========================================================
    # 3.1 Meta-train (FOMAML)
    # =========================================================
    final_model, acc_hist = fomaml_train(
        train_tasks,
        input_shape,
        num_classes=4,
        meta_epochs=20,
        inner_lr=0.01,
        meta_lr=0.001,
        adaptation_steps=3,
        batch_size=16,
    )

    # =========================================================
    # 3.2 Evaluate FOMAML on Unseen Task (Task 0)
    # =========================================================
    task0 = test_tasks[0] if isinstance(test_tasks, list) else list(test_tasks.values())[0]

    acc, precision, recall, f1 = evaluate_on_task(
        final_model,
        task0,
        input_shape,
        num_classes=4,
        fine_tune_epochs=5,
    )

    acc_fomaml_seeds.append(acc)
    precision_fomaml_seeds.append(precision)
    recall_fomaml_seeds.append(recall)
    f1_fomaml_seeds.append(f1)


    print(f"Seed {seed}: FOMAML Acc = {acc:.4f}")

# =============================================================
# 4. STATS: ONLY FOR FOMAML
# =============================================================

mean, ci95, p_value, effect_size = compute_stats(
    acc_fomaml_seeds, 
    acc_fomaml_seeds,   # compare to itself so p-value = 1 (OR remove t-test)
    paired=False         # or skip t-test, your choice
)
def mean_ci(values):
    arr = np.array(values)
    mean = arr.mean()
    std = arr.std(ddof=1)
    ci = 1.96 * std / np.sqrt(len(arr))
    return mean, ci

mean_acc, ci_acc = mean_ci(acc_fomaml_seeds)
mean_prec, ci_prec = mean_ci(precision_fomaml_seeds)
mean_rec, ci_rec = mean_ci(recall_fomaml_seeds)
mean_f1, ci_f1 = mean_ci(f1_fomaml_seeds)
print("\n========== FOMAML MULTI-SEED SUMMARY ==========")
print(f"Accuracy : {mean_acc:.4f} ± {ci_acc:.4f}")
print(f"Precision: {mean_prec:.4f} ± {ci_prec:.4f}")
print(f"Recall   : {mean_rec:.4f} ± {ci_rec:.4f}")
print(f"F1-score : {mean_f1:.4f} ± {ci_f1:.4f}")

print("\n====================================================")
print("           FOMAML STATISTICAL RESULTS ONLY          ")
print("====================================================")
print(f"Mean FOMAML Accuracy : {mean:.4f}")
print(f"95% Confidence Interval: ±{ci95:.4f}")
print(f"Cohen's d Effect Size  : {effect_size:.4f}")
print("\nFOMAML Accuracies over seeds:")
print(acc_fomaml_seeds)

# =============================================================
# 5. BASELINE RNN — SINGLE RUN ONLY
# =============================================================

baseline_model = train_baseline(train_tasks, input_shape, num_classes=4, epochs=10)
baseline_acc = evaluate_baseline(baseline_model, task0)

# If you want precision/recall/F1 for baseline, use this:
qx = np.concatenate(list(task0["query"].values()))
qy = np.concatenate(list(task0["query_labels"].values()))
preds = np.argmax(baseline_model.predict(qx, verbose=0), axis=1)

from sklearn.metrics import precision_score, recall_score, f1_score

baseline_precision = precision_score(qy, preds, average='weighted')
baseline_recall    = recall_score(qy, preds, average='weighted')
baseline_f1        = f1_score(qy, preds, average='weighted')

print("\n====================================================")
print("                BASELINE RNN RESULTS                ")
print("====================================================")
print(f"Accuracy  : {baseline_acc:.4f}")
print(f"Precision : {baseline_precision:.4f}")
print(f"Recall    : {baseline_recall:.4f}")
print(f"F1 Score  : {baseline_f1:.4f}")

#%%% Full xai pipline
# ======================================
# XAI PIPELINE FOR MAML-RNN (PCA with 6 FEATURES)
# ======================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# ------------------------------------------------
# 1. Load and adapt the model for the test task
# ------------------------------------------------
task0 = test_tasks[0]
sx, sy, qx, qy = load_task(task0)

adapted_model = build_rnn_model(input_shape, num_classes=4)
adapted_model.set_weights(final_model.get_weights())
adapted_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy'
)

adapted_model.fit(sx, sy, epochs=5, batch_size=16, verbose=0)

sample = qx[0]    # shape (100, 6)
print("Sample shape:", sample.shape)

# PCA feature names
feature_names = [f"pca_feature_{i}" for i in range(input_shape[1])]
print("Mapped feature names:", feature_names)


# ===================================================
# 2. SmoothGrad
# ===================================================
def smoothgrad(model, x, n_samples=30, noise_level=0.1):
    x = tf.cast(x, tf.float32)
    grads_sum = 0.0

    for _ in range(n_samples):
        noise = tf.random.normal(shape=x.shape, stddev=noise_level)
        x_noisy = x + noise

        with tf.GradientTape() as tape:
            x_in = x_noisy[None, ...]
            tape.watch(x_in)
            logits = model(x_in)
            class_idx = tf.argmax(logits[0])
            loss = logits[0, class_idx]

        grads = tape.gradient(loss, x_in)[0]
        grads_sum += tf.abs(grads)

    return grads_sum.numpy() / n_samples


sal_sg = smoothgrad(adapted_model, sample)
print("SmoothGrad shape:", sal_sg.shape)


# ===================================================
# 3. IG-Smooth
# ===================================================
def ig_smooth(model, x, steps=50, n_samples=20, noise_level=0.1):
    x = tf.cast(x, tf.float32)
    baseline = tf.zeros_like(x)
    ig_total = 0

    for _ in range(n_samples):
        noise = tf.random.normal(shape=x.shape, stddev=noise_level)
        x_noisy = x + noise

        path = tf.stack([
            baseline + (i / steps) * (x_noisy - baseline)
            for i in range(steps + 1)
        ])

        with tf.GradientTape() as tape:
            tape.watch(path)
            logits = model(path)
            class_idx = tf.argmax(logits[-1])
            loss = logits[:, class_idx]

        grads = tape.gradient(loss, path)
        avg_grads = tf.reduce_mean(grads, axis=0)
        ig = (x_noisy - baseline) * avg_grads
        ig_total += ig

    return ig_total.numpy() / n_samples


ig_sg = ig_smooth(adapted_model, sample)
print("IG-Smooth shape:", ig_sg.shape)


# ===================================================
# 4. Plotting utilities
# ===================================================
def plot_timestep_importance(heatmap, title):
    importance = heatmap.sum(axis=1)
    plt.figure(figsize=(12, 4))
    plt.plot(importance)
    plt.title(title)
    plt.xlabel("Timesteps (1–100)")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.show()


def plot_feature_importance(heatmap, title, feature_names, top_k=6):
    importance = heatmap.sum(axis=0)
    idx_sorted = np.argsort(importance)[::-1]

    print(f"\n=== {title} — Feature Ranking ===")
    for idx in idx_sorted[:top_k]:
        print(f"{feature_names[idx]:15s}  importance = {importance[idx]:.4f}")

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(importance)), feature_names, rotation=45)
    plt.title(title)
    plt.ylabel("Importance")
    plt.grid(True)
    plt.show()


# ===================================================
# 5. Plot results
# ===================================================
plot_timestep_importance(sal_sg, "SmoothGrad — Timestep Importance")
plot_timestep_importance(ig_sg, "IG-Smooth — Timestep Importance")

plot_feature_importance(sal_sg, "SmoothGrad — Feature Importance", feature_names)
plot_feature_importance(ig_sg, "IG-Smooth — Feature Importance", feature_names)

print("\n[INFO] XAI pipeline completed successfully.")


# ===================================================
# 6. PCA → Original Feature Mapping
# ===================================================
def explain_pca_feature(pca_model, original_feature_names, top_k=5):
    components = pca_model.components_

    for i in range(components.shape[0]):
        print(f"\n=== PCA Feature {i} Contributions ===")
        weights = components[i]
        idx = np.argsort(-np.abs(weights))[:top_k]

        for j in idx:
            print(f"{original_feature_names[j]:20s}  weight = {weights[j]:.4f}")


original_feature_names = list(df_reference.columns)
explain_pca_feature(pca, original_feature_names, top_k=5)


# ===================================================
# 7. Compute TRUE feature importance (XAI mapped to raw features)
# ===================================================
W = pca.components_
xai_importance = ig_sg.sum(axis=0)     # IG-Smooth PCA-level scores

xai_norm = xai_importance / (np.abs(xai_importance).sum() + 1e-9)
original_feature_importance = np.dot(xai_norm, W)

feature_imp_pairs = list(zip(original_feature_names, original_feature_importance))
feature_imp_pairs_sorted = sorted(feature_imp_pairs, key=lambda x: -abs(x[1]))

print("\n=== TRUE Original Feature Importance (IG-Smooth → PCA → Raw Features) ===")
for name, score in feature_imp_pairs_sorted:
    print(f"{name:20s}  importance = {score:.4f}")

# Plot
names = [p[0] for p in feature_imp_pairs_sorted]
scores = [p[1] for p in feature_imp_pairs_sorted]

plt.figure(figsize=(10,5))
plt.barh(names, scores)
plt.title("True Original Feature Importance (IG-Smooth → PCA → Raw Features)")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

#%%% latency
# ============================================================
# REAL-TIME LATENCY EVALUATION (Reviewer-Ready Version)
# ============================================================

import time
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------
# 1. Helper function (TF model latency)
# ------------------------------------------------------------
def measure_tf_latency(model, sample, runs=200):
    sample_tf = tf.convert_to_tensor(sample[None, ...], dtype=tf.float32)

    # Warm-up (GPU/CPU kernel preparation)
    for _ in range(20):
        _ = model(sample_tf, training=False)

    start = time.time()
    for _ in range(runs):
        _ = model(sample_tf, training=False)
    end = time.time()

    return (end - start) / runs * 1000  # ms


# ------------------------------------------------------------
# 2. TF-compiled latency (XLA / graph mode)
# ------------------------------------------------------------
@tf.function(jit_compile=True)   # XLA on
def model_graph_call(x):
    return final_model(x, training=False)


def measure_tf_compiled_latency(compiled_call, sample, runs=200):
    sample_tf = tf.convert_to_tensor(sample[None, ...], dtype=tf.float32)

    # Warm-up
    for _ in range(20):
        _ = compiled_call(sample_tf)

    start = time.time()
    for _ in range(runs):
        _ = compiled_call(sample_tf)
    end = time.time()

    return (end - start) / runs * 1000


# ------------------------------------------------------------
# 3. Logistic Regression Baseline  (Very Fast CPU-only model)
# ------------------------------------------------------------

# Flatten sequences:  (T, F) → (T*F)
sx_flat = sx.reshape(sx.shape[0], -1)
qx_flat = qx.reshape(qx.shape[0], -1)

# Train LR baseline on support set
baseline_model = LogisticRegression(max_iter=2000)
baseline_model.fit(sx_flat, sy)

def measure_lr_latency(model, sample, runs=200):
    sample_flat = sample.reshape(1, -1)

    # warm-up
    for _ in range(20):
        _ = model.predict(sample_flat)

    start = time.time()
    for _ in range(runs):
        _ = model.predict(sample_flat)
    end = time.time()

    return (end - start) / runs * 1000


# ============================================================
# 4. RUN ALL LATENCY MEASUREMENTS
# ============================================================

sample = qx[0]

lat_eager = measure_tf_latency(final_model, sample)
print("FOMAML-RNN (eager) latency:", lat_eager, "ms")

lat_compiled = measure_tf_compiled_latency(model_graph_call, sample)
print("FOMAML-RNN (TF-compiled) latency:", lat_compiled, "ms")

lat_baseline = measure_lr_latency(baseline_model, sample)
print("Logistic Regression baseline latency:", lat_baseline, "ms")

