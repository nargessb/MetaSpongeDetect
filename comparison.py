# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Specify the path to the folder containing your CSV files
folder_path = r"C:/Users/narges.babadi1/Downloads/Sponge Project/Evaluation_reults/Tinyimagenet/Resnet18"

# Use glob to get all CSV files in the specified folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Initialize a dictionary to hold each CSV file as a separate DataFrame
dataframes = {}

# Load each CSV file as a DataFrame and store it in the dictionary
for file_path in csv_files:
    file_name = os.path.basename(file_path)  # Get the file name to use as the key
    dataframes[file_name] = pd.read_csv(file_path)

# print("Loaded CSV files as DataFrames:")
# print(list(dataframes.keys()))  # Print loaded CSV files

# Create a copy of dataframes for processing
data = dataframes.copy()

# Find the first sponge dataset key, if available
sponge_keys = [k for k in data if 'Sponge' in k]
if sponge_keys:
    first_sponge_key = sponge_keys[0]  # Use the first sponge key as reference
    df_reference = data[first_sponge_key]  # Reference for alignment
    target_shape = df_reference.shape  # Define target shape based on df_reference
else:
    raise ValueError("No 'Sponge' dataset found in data to align with.")

# Define labels dictionary `y` based on key conventions
y = {}
for key in data.keys():
    if 'lb0' in key:
        y[key] = 'No attack'
    elif 'poisoning' in key:
        y[key] = 'Poisoning attack'
    else:
        y[key] = 'Sponge attack'

# Function to align columns to a reference DataFrame
def align_columns_to_reference(reference_df, df):
    """
    Aligns the columns of df to match those in reference_df.
    - Keeps only columns present in both, in the same order as reference_df.
    """
    common_columns = reference_df.columns.intersection(df.columns)
    aligned_df = df[common_columns]  # Keep only common columns
    aligned_df = aligned_df.reindex(columns=reference_df.columns, fill_value=0)  # Reorder and fill missing columns
    return aligned_df

# Align all DataFrames to `df_reference` and handle NaN values
for key, df in data.items():
    df = align_columns_to_reference(df_reference, df)
    
    # Remove 'epoch' column if it exists
    if 'epoch' in df.columns:
        df = df.drop(columns=['epoch'])
    
    # Handle NaN values by filling with 0
    df = df.fillna(0)
    
    # Update the DataFrame in data with the aligned version
    data[key] = df

# Confirm that all DataFrames now have consistent columns
# print("Shapes after alignment:", [df.shape for df in data.values()])

# Concatenate all aligned DataFrames into one
concatenated_data = pd.concat(data.values(), axis=0)
print("Concatenation completed. Shape of concatenated_data:", concatenated_data.shape)

# Stack data for 3D array (for LSTM input)
data_3d = np.stack([df.values for df in data.values()])
print("Shape of data_3d:", data_3d.shape)

# Concatenate labels for all datasets
labels = np.array([y[key] for key in data.keys()])
print("New length of labels:", len(labels))

# Check the structure of the experiment names and column names
experiment_names = list(data.keys())
column_names = concatenated_data.columns
# print("Experiment names:", experiment_names)
# print("Column names:", column_names)

#%%%%%%  Distance-Based Detection	
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

scenarios_stats = data
def normalize_stats_per_scenario(scenarios_stats):
    """
    Normalizes the statistics per feature for each scenario, handling missing or constant values.
    """
    normalized_stats = {}
    for key, scenario_stats in scenarios_stats.items():
        normalized_stats[key] = {}
        for layer, layer_stats in scenario_stats.items():
            df = pd.DataFrame(layer_stats)
            df = handle_missing_values(df)  # Handle missing values before normalization
            
            # Check for constant columns (standard deviation of zero)
            non_constant_columns = df.loc[:, (df != df.iloc[0]).any()]  # Keep non-constant columns
            if non_constant_columns.empty:
                # print(f"All constant columns for layer {layer} in scenario {key}, skipping normalization.")
                normalized_stats[key][layer] = df.to_dict(orient="list")  # Save as-is
            else:
                scaler = StandardScaler()
                df_normalized = pd.DataFrame(scaler.fit_transform(non_constant_columns),
                                             columns=non_constant_columns.columns,
                                             index=non_constant_columns.index)
                normalized_stats[key][layer] = df_normalized.to_dict(orient="list")
    return normalized_stats


def check_key_consistency(scenarios_stats):
    """
    Checks for consistency of keys across scenarios and layers.
    Returns common keys that exist in all scenarios.
    """
    keys_per_layer = {}
    for key, scenario_stats in scenarios_stats.items():
        for layer, layer_stats in scenario_stats.items():
            if layer not in keys_per_layer:
                keys_per_layer[layer] = set(layer_stats.keys())
            else:
                keys_per_layer[layer] &= set(layer_stats.keys())  # Intersection of keys across scenarios
    return keys_per_layer


def calculate_total_distance(stats1, stats2, common_keys):
    """
    Calculate the Euclidean distance between two sets of statistics.
    Only uses common keys.
    """
    total_distance = 0
    for layer in stats1:
        if layer in stats2:
            distance = 0
            for stat_key in common_keys[layer]:
                if stat_key in stats1[layer] and stat_key in stats2[layer]:
                    distance += np.linalg.norm(np.array(stats1[layer][stat_key]) - np.array(stats2[layer][stat_key])) ** 2
                else:
                    print(f"Skipping missing stat_key {stat_key} for layer {layer}")
            total_distance += np.sqrt(distance)
    return total_distance

def handle_missing_values(df):
    """
    Fills NaN values with the mean of the column, and if a column contains all NaNs,
    fills it with zeros.
    """
    return df.fillna(df.mean()).fillna(0)

def compute_dynamic_threshold(normal_scenarios_stats, common_keys):
    """
    Computes a dynamic threshold based on the distance between two 'No attack' scenarios.
    If the distance between the two normal scenarios is zero, it sets a small default threshold.
    """
    normal_keys = list(normal_scenarios_stats.keys())
    if len(normal_keys) >= 2:
        scenario_1 = normal_scenarios_stats[normal_keys[0]]
        scenario_2 = normal_scenarios_stats[normal_keys[1]]
        threshold = calculate_total_distance(scenario_1, scenario_2, common_keys)
        if threshold == 0:
            threshold = 1e-5  # Set a small non-zero threshold
        return threshold
    else:
        print("Not enough 'No attack' scenarios to compute dynamic threshold.")
        return 1e-5  # Default small threshold


def classify_scenarios(scenarios_stats, baseline_stats, common_keys, threshold, y):
    """
    Classify scenarios based on the distance from baseline.
    """
    predictions = []
    for key in y.keys():
        current_scenario_stats = scenarios_stats[key]
        distance_from_normal = calculate_total_distance(current_scenario_stats, baseline_stats, common_keys)
        # print(f"Scenario: {key}, Distance from baseline: {distance_from_normal}")

        if distance_from_normal > threshold:
            if y[key] == 'Poisoning attack':
                predictions.append('Poisoning attack')
            else:
                predictions.append('Sponge attack')
        else:
            predictions.append('No attack')

    return predictions


def drop_first_three_columns(scenarios_stats):
    """
    Drop the first three columns for each scenario.
    """
    updated_scenarios_stats = {}
    for key, df in scenarios_stats.items():
        updated_scenarios_stats[key] = df.iloc[:, 3:]
    return updated_scenarios_stats


def main_classification(scenarios_stats, y):
    """
    Main function to run the classification process.
    """
    scenarios_stats = drop_first_three_columns(scenarios_stats)

    # Normalize stats for each scenario
    scenarios_stats = normalize_stats_per_scenario(scenarios_stats)

    # Check for common keys across all scenarios
    common_keys = check_key_consistency(scenarios_stats)

    # Separate normal and attack scenarios for threshold calculation
    normal_scenarios_stats = {k: v for k, v in scenarios_stats.items() if y[k] == 'No attack'}

    # Compute the baseline stats (average of 'No attack' scenarios)
    if len(normal_scenarios_stats) > 0:
        baseline_stats = next(iter(normal_scenarios_stats.values()))
    else:
        print("No 'No attack' scenarios found!")
        return

    # Compute dynamic threshold based on 'No attack' scenarios
    threshold = compute_dynamic_threshold(normal_scenarios_stats, common_keys)
    # print(f"Dynamic threshold: {threshold}")

    # Classify all scenarios based on the distance from baseline
    predictions = classify_scenarios(scenarios_stats, baseline_stats, common_keys, threshold, y)
    # Prepare actual and predicted labels for evaluation
    actual_labels = list(y.values())
    predicted_labels = predictions
    # Calculate performance metrics
    accuracy = accuracy_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels, average='weighted')
    precision = precision_score(actual_labels, predicted_labels, average='weighted')
    recall = recall_score(actual_labels, predicted_labels, average='weighted')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Optional: Print detailed results for each scenario
    for i, (predicted, actual) in enumerate(zip(predictions, list(y.values()))):
        scenario_key = list(y.keys())[i]
        # print(f"Scenario: {scenario_key}, Predicted - {predicted}, Actual - {actual}")


# Call the main classification function
main_classification(data, y)

#%%%% Krum
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Define relevant feature pattern
relevant_features = [
    'sponge_less', 'fired_perc_clean', 'l2'
] + [f'layer{i}_{stat}' for i in range(6, 76) for stat in 
     ['mean_weight', 'variance_weight', 'max_weight', 'min_weight', 
      'mean_grad', 'variance_grad', 'max_grad', 'min_grad']]

# Update each DataFrame in `data` to keep only the intersection of relevant and actual columns
for key, df in data.items():
    available_columns = df.columns.intersection(relevant_features)
    data[key] = df[available_columns]

print("Columns after filtering:", data[next(iter(data))].columns)

def bulyan(gradients, f=1):
    selected_gradients = []
    gradients_list = gradients.tolist()

    for _ in range(len(gradients) - 2 * f):
        min_distance_sum = float('inf')
        selected_gradient = None
        for i, grad in enumerate(gradients):
            distance_sum = sum(np.linalg.norm(grad - other_grad) for j, other_grad in enumerate(gradients) if j != i)
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                selected_gradient = grad

        selected_gradients.append(selected_gradient)
        gradients_list = [g for g in gradients_list if not np.array_equal(g, selected_gradient)]
        gradients = np.array(gradients_list)

    return np.mean(selected_gradients, axis=0)

def krum(gradients, num_selected=1):
    num_workers = len(gradients)
    distances = np.zeros((num_workers, num_workers))
    
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            distances[i, j] = np.linalg.norm(gradients[i] - gradients[j])
            distances[j, i] = distances[i, j]
    
    scores = []
    for i in range(num_workers):
        sorted_distances = np.sort(distances[i])
        score = np.sum(sorted_distances[:num_workers - num_selected - 1])
        scores.append(score)
    
    selected_index = np.argmin(scores)
    return gradients[selected_index]

def extract_gradients(scenarios_stats, label='No attack'):
    gradients = []
    for key, data in scenarios_stats.items():
        if y[key] == label:
            if not data.empty:
                gradients.append(data.values)
            else:
                gradients.append(np.zeros((1, data.shape[1])))

    if gradients:
        return np.stack(gradients)
    else:
        return np.zeros((1, data.shape[1]))

def compute_baseline(scenarios_stats):
    normal_gradients = extract_gradients(scenarios_stats, label='No attack')
    return np.mean(normal_gradients, axis=0)

def compute_thresholds_per_class(scenarios_stats, baseline, multiplier=1.5):
    thresholds = {}
    for label in ['No attack', 'Poisoning attack', 'Sponge attack']:
        distances = []
        label_gradients = extract_gradients(scenarios_stats, label=label)
        for grad in label_gradients:
            distances.append(np.linalg.norm(grad - baseline))
        avg_distance = np.mean(distances)
        thresholds[label] = multiplier * avg_distance
        print(f"Threshold for '{label}': {thresholds[label]}")
    return thresholds

def krum_with_threshold(gradients, baseline, threshold):
    krum_result = krum(gradients)
    distance = np.linalg.norm(krum_result - baseline)
    return distance > threshold, distance

def bulyan_with_threshold(gradients, baseline, threshold):
    bulyan_result = bulyan(gradients)
    distance = np.linalg.norm(bulyan_result - baseline)
    return distance > threshold, distance

def main_detection_with_threshold(scenarios_stats, y):
    baseline = compute_baseline(scenarios_stats)
    thresholds = compute_thresholds_per_class(scenarios_stats, baseline, multiplier=1.5)

    true_labels = []
    krum_predictions = []
    bulyan_predictions = []

    for key, label in y.items():
        # print(f"\nEvaluating scenario {key} (label: {label})")
        gradients = extract_gradients({key: scenarios_stats[key]})

        krum_detected, krum_distance = krum_with_threshold(gradients, baseline, thresholds[label])
        if krum_detected:
            if krum_distance > thresholds['Poisoning attack']:
                krum_predictions.append('Poisoning attack')
            elif krum_distance > thresholds['Sponge attack']:
                krum_predictions.append('Sponge attack')
            else:
                krum_predictions.append('No attack')
        else:
            krum_predictions.append('No attack')

        bulyan_detected, bulyan_distance = bulyan_with_threshold(gradients, baseline, thresholds[label])
        if bulyan_detected:
            if bulyan_distance > thresholds['Poisoning attack']:
                bulyan_predictions.append('Poisoning attack')
            elif bulyan_distance > thresholds['Sponge attack']:
                bulyan_predictions.append('Sponge attack')
            else:
                bulyan_predictions.append('No attack')
        else:
            bulyan_predictions.append('No attack')

        true_labels.append(label)

    # Calculate and print metrics for Krum
    krum_accuracy = accuracy_score(true_labels, krum_predictions)
    krum_precision = precision_score(true_labels, krum_predictions, average='weighted', zero_division=0)
    krum_recall = recall_score(true_labels, krum_predictions, average='weighted', zero_division=0)
    krum_f1 = f1_score(true_labels, krum_predictions, average='weighted', zero_division=0)
    krum_conf_matrix = confusion_matrix(true_labels, krum_predictions, labels=['No attack', 'Poisoning attack', 'Sponge attack'])

    # Calculate and print metrics for Bulyan
    bulyan_accuracy = accuracy_score(true_labels, bulyan_predictions)
    bulyan_precision = precision_score(true_labels, bulyan_predictions, average='weighted', zero_division=0)
    bulyan_recall = recall_score(true_labels, bulyan_predictions, average='weighted', zero_division=0)
    bulyan_f1 = f1_score(true_labels, bulyan_predictions, average='weighted', zero_division=0)
    bulyan_conf_matrix = confusion_matrix(true_labels, bulyan_predictions, labels=['No attack', 'Poisoning attack', 'Sponge attack'])

    # Display results
    print("\nKrum Detection Metrics:")
    print(f"Accuracy: {krum_accuracy * 100:.2f}%")
    print(f"Precision (weighted): {krum_precision:.2f}")
    print(f"Recall (weighted): {krum_recall:.2f}")
    print(f"F1 Score (weighted): {krum_f1:.2f}")
    print("Confusion Matrix:\n", krum_conf_matrix)

    print("\nBulyan Detection Metrics:")
    print(f"Accuracy: {bulyan_accuracy * 100:.2f}%")
    print(f"Precision (weighted): {bulyan_precision:.2f}")
    print(f"Recall (weighted): {bulyan_recall:.2f}")
    print(f"F1 Score (weighted): {bulyan_f1:.2f}")
    print("Confusion Matrix:\n", bulyan_conf_matrix)

# Run the main detection with threshold
main_detection_with_threshold(data, y)
