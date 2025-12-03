# -*- coding: utf-8 -*-

import os
import pandas as pd

# === CONFIGURATION ===
BASE_DIR = r"C:/Users/narges.babadi1/Downloads/Sponge Project/tables"

DATASETS = ["CFAR10", "MNIST", "GTSRB", "TINYIMAGENET"]

MODELS = ['vgg16', 'resnet18', 'resnet20']   # no resnet50

def read_energy_info_from_csv(filepath):
    if not os.path.exists(filepath) and os.path.exists(filepath + ".csv"):
        filepath += ".csv"

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    headers = lines[0].strip().split("\t")

    for i in range(1, len(lines)):
        values = lines[i].strip().split("\t")
        if len(values) == len(headers):
            break
    else:
        raise ValueError(f"No valid data row in: {filepath}")

    data = dict(zip(headers, values))

    return {
        'dataset': data['dataset'].upper(),
        'model': data['net'].lower(),
        'budget': float(data['budget']),
        'sigma': float(data['sigma']),
        'lb': float(data['lb']),
        'increase_ratio': float(data['increase_ratio']),
    }

# === COLLECT DATA ===
all_records = []

for dataset in DATASETS:
    for model in MODELS:
        folder = os.path.join(BASE_DIR, dataset, model)
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        for filename in os.listdir(folder):
            try:
                record = read_energy_info_from_csv(os.path.join(folder, filename))
                all_records.append(record)
            except Exception as e:
                print(f"⚠️ Skipping file {filename}: {e}")

df = pd.DataFrame(all_records)
print("\nLoaded:", len(df), "rows")
print(df.head())

output_file = os.path.join(BASE_DIR, "combined_energy_results.csv")
df.to_csv(output_file, index=False)

# ======================================================
# === ENERGY PLOTS (budget & lambda sweeps)
# ======================================================

import seaborn as sns
import matplotlib.pyplot as plt

df['model'] = df['model'].str.lower().str.strip()
df['dataset'] = df['dataset'].str.upper().str.strip()

sns.set(style="whitegrid", context="paper", font_scale=1.4)

models = ['vgg16', 'resnet18', 'resnet20']
variables = ['budget', 'lb']
var_labels = {'budget': 'p', 'lb': 'lambda'}

x_ticks = {
    'budget': [0.01, 0.05, 0.1, 0.2],
    'lb': [0, 0.5, 1, 2, 5, 10, 12]
}

custom_markers = {
    'MNIST': '*',
    'CFAR10': 's',
    'GTSRB': 'o',
    'TINYIMAGENET': 'D'
}

marker_sizes = {'*': 14, 's': 9, 'o': 9, 'D': 10}

for var in variables:
    fig, axs = plt.subplots(1, len(models), figsize=(20, 5), sharey=True)
    sigma_fixed = 0.01

    for i, model in enumerate(models):
        ax = axs[i]
        subset = df[(df['model'] == model) & (df['sigma'] == sigma_fixed)].copy()

        if subset.empty:
            ax.set_title(f"{model} (No data)")
            ax.axis("off")
            continue

        subset[var] = pd.Categorical(subset[var], categories=x_ticks[var], ordered=True)

        sns.lineplot(
            data=subset,
            x=var, y='increase_ratio',
            hue='dataset', style='dataset',
            markers=True, dashes=False,
            linewidth=3, ax=ax
        )

        # apply marker overrides
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        for line, label in zip(ax.lines, legend_labels):
            marker = custom_markers.get(label, 'o')
            line.set_marker(marker)
            line.set_markersize(marker_sizes.get(marker, 9))
            line.set_markeredgewidth(1.7)

        ax.set_title(model.upper())
        ax.set_xlabel(var_labels[var])
        ax.set_xticks(x_ticks[var])
        ax.grid(True)

        if i == 0:
            ax.set_ylabel("Energy Increase Ratio")
        else:
            ax.set_ylabel("")

    # ======= GLOBAL LEGEND FIX ==============
    all_handles = []
    all_labels = []

    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in all_labels:
                all_labels.append(label)
                all_handles.append(handle)

    # Draw full legend
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc='lower center',
            ncol=len(all_labels),
            frameon=False,
            fontsize=13
        )

    # remove subplot legends
    for ax in axs:
        if ax.get_legend():
            ax.get_legend().remove()
    # ========================================

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out = f"plot_energy_vs_{var}_sigma_{sigma_fixed}_legend.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print("Saved:", out)

    plt.show()
