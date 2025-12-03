# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:31:33 2025

@author: narges.babadi1
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# === CONFIGURATION ===
# MNIST_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\MNIST\Resnet20"
# CIFAR10_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\Cfar10\Resnet20"
# GTSRB_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\GTSRB\Resnet18"
# Tinyimagenet_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\Tinyimagenet\Resnet18"

SAVE_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plots_Energy"
os.makedirs(SAVE_FOLDER, exist_ok=True)

MODEL_NAME = "resnet20"
SIGMA_VAL = "Sigma0.1"  # ✅ Fixed sigma value
budgets = [0.01, 0.2]   # ✅ One figure per budget
lbs = [0.5, 1, 2, 5, 10, 12]  # ✅ One subplot per lambda

# === Smoothing settings ===
window = 7
polyorder = 2
# === CONFIGURATION ===
DATASETS = {
    "MNIST": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\MNIST\Resnet20",
        "model": "resnet20",
        "sigma": "Sigma0.1",
        "color": "green"
    },
    "CIFAR10": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\Cfar10\Resnet20",
        "model": "resnet20",
        "sigma": "Sigma0.1",
        "color": "purple"
    },
    "GTSRB": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\GTSRB\Resnet18",
        "model": "resnet18",
        "sigma": "sigma",      # match ANY sigma
        "color": "orange"
    },
    "TinyImageNet": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\Tinyimagenet\Resnet18",
        "model": "resnet18",
        "sigma": "sigma",      # match ANY sigma
        "color": "blue"
    }
}

def process_data(dataset_cfg, budget, lb):
    folder = dataset_cfg["folder"]
    model_key = dataset_cfg["model"].lower()
    sigma_key = dataset_cfg["sigma"].lower()   # can be 'sigma' to match anything

    budget_str = f"budget{budget}".lower()
    lb_str = f"lb{lb}".lower()

    for fname in os.listdir(folder):
        fname_low = fname.lower()

        if model_key not in fname_low:
            continue
        if sigma_key not in fname_low:  # for GTSRB/Tiny, 'sigma' matches all
            continue
        if budget_str not in fname_low:
            continue
        if lb_str not in fname_low:
            continue

        df = pd.read_csv(os.path.join(folder, fname))
        df.columns = df.columns.str.strip()
        df["epoch"] = df["epoch"] + 1

        df["energy_smooth"] = savgol_filter(df["sourceset_ratio"], window, polyorder, mode="interp")
        std_band = df["sourceset_ratio"].rolling(window, center=True).std()

        lower = (df["energy_smooth"] - std_band).clip(lower=0)
        upper = (df["energy_smooth"] + std_band).clip(upper=1)
        valid = ~std_band.isna()

        return df, lower, upper, valid

    return None, None, None, None
for budget in budgets:
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
    axs = axs.flatten()

    y_min, y_max = float('inf'), float('-inf')

    for i, lb in enumerate(lbs):
        ax = axs[i]
        found_any = False

        for dataset_name, cfg in DATASETS.items():
            df, lower, upper, valid = process_data(cfg, budget, lb)
            if df is None:
                continue

            ax.plot(df["epoch"], df["energy_smooth"],
                    color=cfg["color"], linewidth=2.4, label=dataset_name)

            ax.fill_between(
                df["epoch"][valid], lower[valid], upper[valid],
                alpha=0.25, color=cfg["color"]
            )

            y_min = min(y_min, lower[valid].min())
            y_max = max(y_max, upper[valid].max())
            found_any = True

        ax.set_title(f"$\\lambda$ = {lb}" + (" (No data)" if not found_any else ""))
        ax.set_xlim(1, 100)
        ax.set_xticks(range(1, 101, 10))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Source Energy Ratio")
        ax.grid(True)
        ax.tick_params(axis='both')

    # --- Y-axis normalization ---
    y_pad = (y_max - y_min) * 0.05
    yticks = None
    for ax in axs:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        if yticks is None:
            yticks = ax.get_yticks()
        ax.set_yticks(yticks)

    # --- Shared legend ---
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    outfile = f"Sponge_EnergyGrid_Budget{budget}.png"
    plt.savefig(os.path.join(SAVE_FOLDER, outfile), dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", outfile)
    
#%%% no attack comparison
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# =========================================================
# CONFIG — DIRECTORIES YOU ALREADY HAVE
# =========================================================
DATASETS = {
    "MNIST": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\MNIST\Resnet20",
        "model": "resnet20",
    },
    "CIFAR10": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\Cfar10\Resnet20",
        "model": "resnet20",
    },
    "GTSRB": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\GTSRB\Resnet18",
        "model": "resnet18",
    },
    "TinyImageNet": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\Tinyimagenet\Resnet18",
        "model": "resnet18",
    }
}

SAVE_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plots_Compare_NoAttack_Sponge"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Filters
SIGMA_FILTER = "Sigma0.01"    # <== SPECIAL REQUEST
BUDGET_FILTER = "budget0.2"     # <== SPECIAL REQUEST

window = 7
polyorder = 2


# =========================================================
# SAFE ENERGY EXTRACTOR (handles corrupted TinyImageNet files)
# =========================================================
def safe_energy(df):
    if df is None:
        return None
    if "sourceset_ratio" in df.columns:
        return df["sourceset_ratio"]
    print("⚠ WARNING: sourceset_ratio missing — using flat zero baseline.")
    return pd.Series([0] * len(df))


# =========================================================
# LOADING FUNCTION — NOW FILTERING BY (sigma, budget)
# =========================================================
def load_lambda(folder, model, lb):
    model_key = model.lower()
    sigma_key = SIGMA_FILTER.lower()
    budget_key = BUDGET_FILTER.lower()
    lb_key = f"lb{lb}".lower()

    for fname in os.listdir(folder):
        f = fname.lower()

        if model_key not in f: continue
        if sigma_key not in f: continue
        if budget_key not in f: continue
        if lb_key not in f: continue

        df = pd.read_csv(os.path.join(folder, fname))
        df.columns = df.columns.str.strip()
        df["epoch"] = df["epoch"] + 1
        return df

    return None


# =========================================================
# MAIN PLOTTING
# =========================================================
fig, axs = plt.subplots(2, 4, figsize=(24, 10), sharey=False)
energy_row, acc_row = axs[0], axs[1]

for col, (dname, cfg) in enumerate(DATASETS.items()):
    folder = cfg["folder"]
    model = cfg["model"]

    # Load λ = 0 and λ = 12
    df0 = load_lambda(folder, model, lb=0)
    df12 = load_lambda(folder, model, lb=5)

    if df0 is None and df12 is None:
        print(f"⚠ No valid files found for {dname}")
        continue

    # Extract safe energy
    e0 = safe_energy(df0)
    e12 = safe_energy(df12)

    # ======= ENERGY PLOT ========
    ax_e = energy_row[col]

    if df0 is not None:
        df0["energy_smooth"] = savgol_filter(e0, window, polyorder)
        ax_e.plot(df0["epoch"], df0["energy_smooth"],
                  color="tab:purple", linewidth=2.6,
                  label="No Attack (λ=0)")

    if df12 is not None:
        df12["energy_smooth"] = savgol_filter(e12, window, polyorder)
        ax_e.plot(df12["epoch"], df12["energy_smooth"],
                  color="tab:red", linewidth=2.6,
                  label="Sponge Attack (λ=5)")

    ax_e.set_title(f"{dname}")
    ax_e.set_xlabel("Epoch")
    ax_e.set_ylabel("Energy Ratio")
    ax_e.grid(True)

    # ======= ACCURACY PLOT ========
    ax_a = acc_row[col]

    if df0 is not None:
        df0["acc_smooth"] = savgol_filter(df0["train_acc"], window, polyorder)
        ax_a.plot(df0["epoch"], df0["acc_smooth"],
                  color="tab:purple", linewidth=2.6,
                  label="No Attack (λ=0)")

    if df12 is not None:
        df12["acc_smooth"] = savgol_filter(df12["train_acc"], window, polyorder)
        ax_a.plot(df12["epoch"], df12["acc_smooth"],
                  color="tab:red", linewidth=2.6,
                  label="Sponge Attack (λ=12)")

    ax_a.set_title(f"{dname}")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Accuracy")
    ax_a.grid(True)

# =========================================================
# SHARED LEGEND
# =========================================================
handles, labels = energy_row[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=14)

plt.tight_layout(rect=[0, 0.06, 1, 1])
outfile = "Compare_NoAttack_vs_Sponge_sigma1e-4_budget0.2.png"
plt.savefig(os.path.join(SAVE_FOLDER, outfile), dpi=300)
plt.show()

print("Saved:", outfile)

#%%% TRAIN ACCURACY

# === CONFIGURATION ===
DATASETS = {
    "MNIST": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\MNIST\Resnet20",
        "model": "resnet20",
        "sigma": "Sigma0.1",
        "color": "tab:red"
    },
    "CIFAR10": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\Cfar10\Resnet20",
        "model": "resnet20",
        "sigma": "Sigma0.1",
        "color": "tab:blue"
    },
    "GTSRB": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\GTSRB\Resnet18",
        "model": "resnet18",
        "sigma": "sigma",   # match ANY sigma
        "color": "tab:orange"
    },
    "TinyImageNet": {
        "folder": r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\Tinyimagenet\Resnet18",
        "model": "resnet18",
        "sigma": "sigma",   # match ANY sigma
        "color": "tab:green"
    }
}

SAVE_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plots_Energy"
os.makedirs(SAVE_FOLDER, exist_ok=True)

budgets = [0.01, 0.2]
lbs = [0.5, 1, 2, 5, 10, 12]

window = 7
polyorder = 2
def process_data(cfg, budget, lb):
    folder = cfg["folder"]
    model_key = cfg["model"].lower()
    sigma_key = cfg["sigma"].lower()   # "sigma" matches ANY Sigma
    budget_str = f"budget{budget}".lower()
    lb_str = f"lb{lb}".lower()

    for fname in os.listdir(folder):
        f = fname.lower()

        if model_key not in f:
            continue
        if sigma_key not in f:
            continue
        if budget_str not in f:
            continue
        if lb_str not in f:
            continue

        df = pd.read_csv(os.path.join(folder, fname))
        df.columns = df.columns.str.strip()
        df["epoch"] = df["epoch"] + 1

        df["train_acc_smooth"] = savgol_filter(df["train_acc"], window, polyorder, mode="interp")
        std_band = df["train_acc"].rolling(window, center=True).std()

        lower = (df["train_acc_smooth"] - std_band).clip(lower=0)
        upper = (df["train_acc_smooth"] + std_band).clip(upper=1)
        valid = ~std_band.isna()

        return df, lower, upper, valid

    return None, None, None, None
for budget in budgets:
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
    axs = axs.flatten()

    y_min, y_max = float("inf"), float("-inf")

    for i, lb in enumerate(lbs):
        ax = axs[i]
        found = False

        for name, cfg in DATASETS.items():
            df, lower, upper, valid = process_data(cfg, budget, lb)
            if df is None:
                continue

            ax.plot(df["epoch"], df["train_acc_smooth"],
                    color=cfg["color"], linewidth=2.5, label=name)

            ax.fill_between(df["epoch"][valid], lower[valid], upper[valid],
                            alpha=0.25, color=cfg["color"])

            y_min = min(y_min, lower[valid].min())
            y_max = max(y_max, upper[valid].max())
            found = True

        ax.set_title(f"$\\lambda$ = {lb}" + (" (No data)" if not found else ""))
        ax.set_xlim(1, 100)
        ax.set_xticks(range(1, 101, 10))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Accuracy")
        ax.grid(True)
        ax.tick_params(axis='both')

    # Consistent Y-limits
    y_pad = (y_max - y_min) * 0.05
    yticks = None
    for ax in axs:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        if yticks is None:
            yticks = ax.get_yticks()
        ax.set_yticks(yticks)

    # Shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    outfile = f"TrainAccGrid_Budget{budget}.png"
    plt.savefig(os.path.join(SAVE_FOLDER, outfile), dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {outfile}")
#%%% resnet50 GTSRB

import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# === CONFIGURATION ===
GTSRB50_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plot_data\GTSRB\Resnet50"
SAVE_FOLDER = r"C:\Users\narges.babadi1\Downloads\Sponge Project\Plots_GTSRB50"
os.makedirs(SAVE_FOLDER, exist_ok=True)

MODEL_NAME = "resnet50"
SIGMA_KEY = "sigma"   # match ANY sigma
budgets = [0.01, 0.2]  # merged
budget_colors = {
    0.01: "tab:purple",
    0.2: "tab:blue"
}
lbs = [0.5, 1, 2, 5, 10, 12]

window = 7
polyorder = 2

def load_data(folder, budget, lb):
    budget_str = f"budget{budget}".lower()
    lb_str = f"lb{lb}".lower()

    for fname in os.listdir(folder):
        f = fname.lower()
        if MODEL_NAME not in f: continue
        if SIGMA_KEY not in f: continue
        if budget_str not in f: continue
        if lb_str not in f: continue

        df = pd.read_csv(os.path.join(folder, fname))
        df.columns = df.columns.str.strip()
        df["epoch"] = df["epoch"] + 1
        return df
    return None

# ============================================================
# ENERGY GRID — MERGED BUDGETS, DIFFERENT COLORS
# ============================================================
fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
axs = axs.flatten()

y_min, y_max = float("inf"), float("-inf")

for i, lb in enumerate(lbs):
    ax = axs[i]
    found = False

    for budget in budgets:
        df = load_data(GTSRB50_FOLDER, budget, lb)
        if df is None:
            continue

        # Smooth Energy
        df["energy_smooth"] = savgol_filter(df["sourceset_ratio"], window, polyorder)
        std_band = df["sourceset_ratio"].rolling(window, center=True).std()

        lower = (df["energy_smooth"] - std_band).clip(lower=0)
        upper = (df["energy_smooth"] + std_band).clip(upper=1)
        valid = ~std_band.isna()

        ax.plot(df["epoch"], df["energy_smooth"],
                linewidth=2.5,
                color=budget_colors[budget],
                label=f"Budget {budget}")

        ax.fill_between(df["epoch"][valid], lower[valid], upper[valid],
                        alpha=0.15, color=budget_colors[budget])

        y_min = min(y_min, lower[valid].min())
        y_max = max(y_max, upper[valid].max())
        found = True

    ax.set_title(f"λ = {lb}" + (" (No data)" if not found else ""))
    ax.grid(True)
    ax.set_xlim(1, 100)
    ax.set_xticks(range(1, 101, 10))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy Ratio")

# Fit Y limits
y_pad = (y_max - y_min) * 0.05
for ax in axs:
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

# Shared legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_energy = "GTSRB50_EnergyGrid_MergedBudgets.png"
plt.savefig(os.path.join(SAVE_FOLDER, out_energy), dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out_energy)

# ============================================================
# TRAIN ACCURACY GRID — MERGED BUDGETS, DIFFERENT COLORS
# ============================================================
fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
axs = axs.flatten()

y_min, y_max = float("inf"), float("-inf")

for i, lb in enumerate(lbs):
    ax = axs[i]
    found = False

    for budget in budgets:
        df = load_data(GTSRB50_FOLDER, budget, lb)
        if df is None:
            continue

        # Smooth Accuracy
        df["acc_smooth"] = savgol_filter(df["train_acc"], window, polyorder)
        std_band = df["train_acc"].rolling(window, center=True).std()

        lower = (df["acc_smooth"] - std_band).clip(lower=0)
        upper = (df["acc_smooth"] + std_band).clip(upper=1)
        valid = ~std_band.isna()

        ax.plot(df["epoch"], df["acc_smooth"],
                linewidth=2.5,
                color=budget_colors[budget],
                label=f"Budget {budget}")

        ax.fill_between(df["epoch"][valid], lower[valid], upper[valid],
                        alpha=0.15, color=budget_colors[budget])

        y_min = min(y_min, lower[valid].min())
        y_max = max(y_max, upper[valid].max())
        found = True

    ax.set_title(f"λ = {lb}" + (" (No data)" if not found else ""))
    ax.grid(True)
    ax.set_xlim(1, 100)
    ax.set_xticks(range(1, 101, 10))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")

# Fit Y limits
y_pad = (y_max - y_min) * 0.05
for ax in axs:
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

# Shared legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_acc = "GTSRB50_TrainAccGrid_MergedBudgets.png"
plt.savefig(os.path.join(SAVE_FOLDER, out_acc), dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out_acc)

