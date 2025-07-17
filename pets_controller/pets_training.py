import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from sklearn.preprocessing import StandardScaler
import glob
import random


def extract_window_features(events, window_bsr_indices, window_size=1):
    """Extract features for a window between BSRs using slot-based timing Events
    array format:

    [0]: Event type (SR=0, BSR=1, PRB=2) [1]: BSR bytes [2]: PRBs [3]:
    Timestamps [4]: Slots [5]: Label
    """
    all_features = []

    # Get the last BSR's slot as the search end slot
    final_bsr_slot = events[window_bsr_indices[-1]][4]

    # Process each BSR interval in the window
    for i in range(window_size):
        current_start_idx = window_bsr_indices[i]
        current_end_idx = window_bsr_indices[i + 1]

        start_bsr = events[current_start_idx]
        end_bsr = events[current_end_idx]

        # Find first PRB after start_bsr but before final_bsr using slots
        first_prb_slot = (
            final_bsr_slot  # Default to final BSR slot if no PRB found
        )
        search_idx = current_start_idx + 1
        while search_idx < len(events):
            if events[search_idx][0] == 2:  # PRB event
                first_prb_slot = events[search_idx][4]
                break
            if (
                events[search_idx][4] > final_bsr_slot
            ):  # Stop if we reach final BSR slot
                break
            search_idx += 1

        # Calculate slot difference until first PRB
        slots_until_prb = first_prb_slot - start_bsr[4]

        # Calculate other features
        bsr_diff = end_bsr[1] - start_bsr[1]
        end_bsr_value = end_bsr[1]

        total_prbs = 0
        sr_count = 0
        prb_events = 0

        for event in events[current_start_idx + 1 : current_end_idx]:
            if event[0] == 2:  # PRB event
                total_prbs += event[2]
                prb_events += 1
            elif event[0] == 0:  # SR event
                sr_count += 1

        # Calculate window duration in slots
        window_slots = end_bsr[4] - start_bsr[4]

        # Calculate BSR per PRB
        bsr_per_prb = bsr_diff / (
            total_prbs + 1e-6
        )  # Add small epsilon to avoid division by zero
        # Calculate rates using slots (multiply by 1000 to convert to per-millisecond rate)
        # Assuming each slot is 0.5ms
        window_duration_ms = window_slots * 0.5
        bsr_update_rate = 1000.0 / (window_duration_ms + 1e-6)
        sr_rate = sr_count * 1000.0 / (window_duration_ms + 1e-6)

        # Features for this interval
        interval_features = np.array(
            [
                bsr_diff,  # Difference between BSRs
                total_prbs,  # Total PRBs allocated
                sr_count,  # Number of SR events
                end_bsr_value,  # Value of the end BSR
                bsr_per_prb,  # BSR difference normalized by PRBs
                window_slots,  # Time duration in slots
                bsr_update_rate,  # Rate of BSR updates
                sr_rate,  # Rate of SR events
                slots_until_prb,  # Slots until first PRB after BSR
            ]
        )

        all_features.append(interval_features)

    return np.concatenate(all_features)


def prepare_training_data(labeled_data, window_size=1, feature_indices=None):
    """Prepare window-based features from labeled data Args:

    labeled_data: dictionary of RNTI to events window_size: number of BSR
    intervals to include in each window feature_indices: indices of features to
    use for each interval (0-8)
    """
    features = []
    labels = []

    for rnti, events in labeled_data.items():
        bsr_indices = [i for i, event in enumerate(events) if event[0] == 1]

        for i in range(len(bsr_indices) - window_size):
            window_bsr_indices = bsr_indices[i : i + window_size + 1]

            # Extract all features for the window
            window_features = extract_window_features(
                events, window_bsr_indices, window_size
            )

            # Reshape features to (window_size, 9) and select specified features for each interval
            window_features = window_features.reshape(window_size, 9)
            if feature_indices is not None:
                window_features = window_features[:, feature_indices]

            # Flatten the features back to 1D array
            window_features = window_features.flatten()

            window_label = events[window_bsr_indices[-1]][5]

            features.append(window_features)
            labels.append(window_label)

    return np.array(features), np.array(labels)


def evaluate_per_rnti(
    model, scaler, val_data, model_name, feature_indices, window_size=1
):
    """
    Evaluate model performance for each RNTI separately.
    """
    print(f"\nPer-RNTI Evaluation for {model_name}:")

    for rnti, events in val_data.items():
        # Prepare data for this RNTI with the correct window size and feature selection
        X_val_rnti, y_val_rnti = prepare_training_data(
            {rnti: events}, window_size, feature_indices
        )
        if len(X_val_rnti) == 0:
            continue

        # Scale features (no need to select features as it's done in prepare_training_data)
        X_val_scaled = scaler.transform(X_val_rnti)

        # Get predictions
        y_pred = model.predict(X_val_scaled)

        # Calculate metrics
        accuracy = np.mean(y_val_rnti == y_pred)
        tn, fp, fn, tp = confusion_matrix(y_val_rnti, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\nRNTI {rnti}:")
        print(f"  Windows: {len(y_val_rnti)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print("  Confusion Matrix:")
        print(f"    TN: {tn}, FP: {fp}")
        print(f"    FN: {fn}, TP: {tp}")


def combine_data(data_files):
    """
    Combine multiple .npy files into one dataset Each RNTI will be prefixed with
    its source file name to avoid conflicts.
    """
    combined_data = {}
    for file in data_files:
        # Get a unique prefix from the file name
        file_prefix = os.path.splitext(os.path.basename(file))[0]

        data = np.load(file, allow_pickle=True).item()
        for rnti, events in data.items():
            # Create a unique key by combining file prefix and RNTI
            unique_key = f"{file_prefix}_{rnti}"
            combined_data[unique_key] = events

    return combined_data


def train_and_evaluate(
    train_files,
    val_file,
    output_dir,
    label_type,
    feature_indices,
    window_size=1,
):
    """
    Train and evaluate the model using selected features.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and combine training data
    train_data = combine_data(train_files)
    val_data = np.load(val_file, allow_pickle=True).item()

    # Prepare data with feature selection
    X_train, y_train = prepare_training_data(
        train_data, window_size, feature_indices
    )
    X_val, y_val = prepare_training_data(val_data, window_size, feature_indices)

    # All feature names for one interval
    base_feature_names = [
        "BSR Difference",  # Change in BSR between two consecutive BSR events
        "Total PRBs",  # Sum of PRBs allocated in the window
        "SR Count",  # Number of SRs in the window
        "End BSR",  # Value of the end BSR
        "BSR per PRB",  # BSR difference normalized by PRBs
        "Window Duration",  # Time duration between two BSR events
        "BSR Update Rate",  # Frequency of BSR updates
        "SR Rate",  # Frequency of SR events
        "Time Until PRB",  # Time until first PRB after BSR
    ]

    # Update feature names generation
    selected_feature_names = [base_feature_names[i] for i in feature_indices]
    all_feature_names = []
    for i in range(window_size):
        interval_names = [
            f"Interval_{i+1}_{name}" for name in selected_feature_names
        ]
        all_feature_names.extend(interval_names)

    feature_names = all_feature_names  # All names are already selected

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\nUsing features:", ", ".join(feature_names))

    print("\nFeature statistics before normalization:")
    for i, name in enumerate(feature_names):
        print(f"{name}:")
        print(f"  Mean: {X_train[:, i].mean():.2f}")
        print(f"  Std: {X_train[:, i].std():.2f}")
        print(f"  Min: {X_train[:, i].min():.2f}")
        print(f"  Max: {X_train[:, i].max():.2f}")

    print("\nFeature statistics after normalization:")
    for i, name in enumerate(feature_names):
        print(f"{name}:")
        print(f"  Mean: {X_train_scaled[:, i].mean():.2f}")
        print(f"  Std: {X_train_scaled[:, i].std():.2f}")
        print(f"  Min: {X_train_scaled[:, i].min():.2f}")
        print(f"  Max: {X_train_scaled[:, i].max():.2f}")

    print("\nTraining data statistics:")
    print(f"Number of windows: {len(X_train)}")
    print(f"Positive labels: {np.sum(y_train == 1)}")
    print(f"Negative labels: {np.sum(y_train == 0)}")

    # Train models
    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=np.sum(y_train == 0)
            / np.sum(y_train == 1),  # Handle class imbalance
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=20,
            min_split_gain=1e-3,
            num_leaves=31,
            scale_pos_weight=np.sum(y_train == 0)
            / np.sum(y_train == 1),  # Handle class imbalance
            n_jobs=-1,
            verbose=-1,
        ),
    }

    results = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name} for {label_type}...")

        # Train with normalized features
        model.fit(X_train_scaled, y_train)

        # Overall evaluation
        y_pred = model.predict(X_val_scaled)
        print(f"\nOverall Validation Results for {model_name}:")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

        # Per-RNTI evaluation
        evaluate_per_rnti(
            model, scaler, val_data, model_name, feature_indices, window_size
        )

        # Save model and scaler
        model_path = os.path.join(
            output_dir, f"{label_type}_{model_name}.joblib"
        )
        scaler_path = os.path.join(output_dir, f"{label_type}_scaler.joblib")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

        # Print feature importance
        print("\nFeature Importance Ranking:")
        importances = model.feature_importances_
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(
            key=lambda x: x[1], reverse=True
        )  # 按重要性降序排序
        for name, importance in importance_pairs:
            print(f"{name}: {importance:.4f}")

        results[model_name] = {
            "model": model,
            "scaler": scaler,
            "predictions": y_pred,
            "model_path": model_path,
            "scaler_path": scaler_path,
        }

    return results


def find_all_data_files(label_type, base_dir="labeled_data"):
    """Find all files with specified label type in base directory Args:

    label_type: 'bsr_only' or 'first_bsr'     base_dir: base directory to search
    in Returns:     list of file paths
    """
    pattern = os.path.join(base_dir, "**", f"*_{label_type}.npy")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise ValueError(
            f"No *_{label_type}.npy files found in {base_dir} directory"
        )
    return sorted(files)


def auto_train_and_evaluate(
    output_dir, feature_indices, seed=42, label_type="bsr_only", window_size=1
):
    """
    Automatically load all data files, extract features, split and train.
    """
    print("\nAuto mode: Finding all data files...")
    all_files = find_all_data_files(label_type)
    print(f"Found {len(all_files)} data files:")
    for f in all_files:
        print(f"  {f}")

    # Load and combine all data
    all_data = combine_data(all_files)

    # Extract features from all data
    X_all, y_all = prepare_training_data(all_data, window_size)

    # Select features
    X_all = X_all[:, feature_indices]

    # Split data
    random.seed(seed)
    indices = list(range(len(X_all)))
    random.shuffle(indices)

    split_idx = int(len(X_all) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]

    print("\nData split statistics:")
    print(f"Total samples: {len(X_all)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Get label type from first file
    label_type = os.path.basename(all_files[0]).split("_")[-1].split(".")[0]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # All feature names for one interval
    base_feature_names = [
        "BSR Difference",  # Change in BSR between two consecutive BSR events
        "Total PRBs",  # Sum of PRBs allocated in the window
        "SR Count",  # Number of SRs in the window
        "End BSR",  # Value of the end BSR
        "BSR per PRB",  # BSR difference normalized by PRBs
        "Window Duration",  # Time duration between two BSR events
        "BSR Update Rate",  # Frequency of BSR updates
        "SR Rate",  # Frequency of SR events
        "Time Until PRB",  # Time until first PRB after BSR
    ]

    # Generate feature names for all intervals in the window
    all_feature_names = []
    for i in range(window_size):
        interval_names = [
            f"Interval_{i+1}_{name}" for name in base_feature_names
        ]
        all_feature_names.extend(interval_names)

    # Selected feature names
    feature_names = [all_feature_names[i] for i in feature_indices]

    print("\nUsing features:", ", ".join(feature_names))

    # Train models
    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=10, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            min_child_samples=20,
            min_split_gain=1e-3,
            num_leaves=31,
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
            n_jobs=-1,
            verbose=-1,
        ),
    }

    results = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name} for {label_type}...")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        # Detailed evaluation
        print(f"\nOverall Validation Results for {model_name}:")

        # Calculate metrics
        accuracy = np.mean(y_val == y_pred)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Print detailed statistics
        print("\nOverall Statistics:")
        print(f"Total samples: {len(y_val)}")
        print(f"Positive samples: {np.sum(y_val == 1)}")
        print(f"Negative samples: {np.sum(y_val == 0)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        print("\nDetailed Metrics:")
        print("True Negatives (TN):", tn)
        print("False Positives (FP):", fp)
        print("False Negatives (FN):", fn)
        print("True Positives (TP):", tp)

        print("\nRates:")
        print(f"False Positive Rate: {fp/(fp+tn):.4f}")
        print(f"False Negative Rate: {fn/(fn+tp):.4f}")
        print(f"True Positive Rate (Recall): {tp/(tp+fn):.4f}")
        print(f"True Negative Rate: {tn/(tn+fp):.4f}")

        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

        # Save model and scaler
        model_path = os.path.join(
            output_dir, f"{label_type}_{model_name}.joblib"
        )
        scaler_path = os.path.join(output_dir, f"{label_type}_scaler.joblib")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

        results[model_name] = {
            "model": model,
            "scaler": scaler,
            "predictions": y_pred,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
        }

        print("\nFeature Importance Ranking:")
        importances = model.feature_importances_
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(
            key=lambda x: x[1], reverse=True
        )  # 按重要性降序排序
        for name, importance in importance_pairs:
            print(f"{name}: {importance:.4f}")

    return results


def check_file_label_type(file_path, expected_label_type):
    """Check if the file name matches the expected label type Returns:

    bool: True if matches, False otherwise
    """
    # Get the base name of the file and check if it ends with expected label type
    base_name = os.path.basename(file_path)
    return base_name.endswith(f"_{expected_label_type}.npy")


def evaluate_and_record_predictions(
    model, scaler, val_data, model_name, feature_indices, window_size=1
):
    """Evaluate model and record predictions for each BSR Returns:

    dict: Dictionary mapping RNTI to list of predicted labels
    """
    bsr_predictions = {}  # Dict to store BSR predictions for each RNTI

    print(f"\nPer-RNTI Evaluation for {model_name}:")

    for rnti, events in val_data.items():
        # Prepare data
        X_val_rnti, y_val_rnti = prepare_training_data(
            {rnti: events}, window_size, feature_indices
        )
        if len(X_val_rnti) == 0:
            continue

        # Scale features and get predictions
        X_val_scaled = scaler.transform(X_val_rnti)
        y_pred = model.predict(X_val_scaled)

        # Store predictions for this RNTI
        bsr_predictions[rnti] = y_pred.tolist()

        # Calculate metrics
        accuracy = np.mean(y_val_rnti == y_pred)
        tn, fp, fn, tp = confusion_matrix(y_val_rnti, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\nRNTI {rnti}:")
        print(f"  Windows: {len(y_val_rnti)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print("  Confusion Matrix:")
        print(f"    TN: {tn}, FP: {fp}")
        print(f"    FN: {fn}, TP: {tp}")

    return bsr_predictions


def print_events_with_predictions(full_events_path, predictions, window_size):
    """Print full events data with predictions Args:

    full_events_path: Path to full events data file predictions: Dictionary
    mapping RNTI to list of predicted labels window_size: Window size used for
    predictions
    """
    # Load full events data
    full_events = np.load(full_events_path, allow_pickle=True).item()

    print("\nFull Events Data Analysis with Predictions:")
    print(f"Number of RNTIs: {len(full_events)}")

    for rnti, events in full_events.items():
        print(f"\nRNTI {rnti}")
        total_events = len(events)

        # Count event types
        sr_count = np.sum(events[:, 0] == 0)  # SR type is 0
        bsr_count = np.sum(events[:, 0] == 1)  # BSR type is 1
        prb_count = np.sum(events[:, 0] == 2)  # PRB type is 2
        req_start_count = np.sum(events[:, 0] == 3)  # REQUEST_START is 3
        req_end_count = np.sum(events[:, 0] == 4)  # REQUEST_END is 4

        # Count unique request sequences
        unique_requests = len(np.unique(events[events[:, 5] > 0, 5]))

        print(f"Total events: {total_events}")
        print(f"Event distribution:")
        print(f"  SR: {sr_count}")
        print(f"  BSR: {bsr_count}")
        print(f"  PRB: {prb_count}")
        print(f"  REQUEST_START: {req_start_count}")
        print(f"  REQUEST_END: {req_end_count}")
        print(f"Number of unique requests: {unique_requests}")

        print("\nAll events details:")
        print(
            "Type      | Timestamp(ms) | BSR bytes | PRBs | TimeDiff(ms) |"
            " ReqSeq | BSR_Label | Predicted"
        )

        # Get predictions for this RNTI
        rnti_predictions = predictions.get(rnti, [])
        bsr_count = 0  # Count BSRs to handle window offset
        pred_idx = 0

        for event in events:
            event_type = {
                0: "SR",
                1: "BSR",
                2: "PRB",
                3: "REQ_START",
                4: "REQ_END",
            }.get(event[0], "UNKNOWN")

            req_seq = int(event[5])
            req_seq_str = str(req_seq) if req_seq > 0 else "0"
            bsr_label = str(int(event[6])) + ":bsr" if event[6] > 0 else "0"

            # Add prediction if it's a BSR event after window_size BSRs
            pred_str = "0"
            if event_type == "BSR":
                if bsr_count >= window_size and pred_idx < len(
                    rnti_predictions
                ):
                    pred_str = str(int(rnti_predictions[pred_idx])) + ":pred"
                    pred_idx += 1
                bsr_count += 1

            print(
                f"{event_type:9} | {event[3]:11.2f} | {event[1]:9.0f} |"
                f" {event[2]:4.0f} | {event[4]:11.2f} | {req_seq_str:6} |"
                f" {bsr_label:9} | {pred_str:9}"
            )
        print("-" * 90)


def print_mismatched_predictions(full_events_path, predictions, window_size):
    """
    Print BSRs where predictions don't match the labels.
    """
    # Load full events data
    full_events = np.load(full_events_path, allow_pickle=True).item()

    print("\nMismatched Predictions:")
    print("RNTI | Timestamp | BSR Label | Prediction")
    print("-" * 45)

    for rnti, events in full_events.items():
        rnti_predictions = predictions.get(rnti, [])
        bsr_count = 0
        pred_idx = 0

        for event in events:
            if event[0] == 1:  # BSR event
                if bsr_count >= window_size and pred_idx < len(
                    rnti_predictions
                ):
                    # Convert BSR label to binary (0 or 1)
                    true_label = 1 if event[6] > 0 else 0
                    pred_label = int(rnti_predictions[pred_idx])

                    # Check if prediction doesn't match label
                    if true_label != pred_label:
                        print(
                            f"{rnti:4} | {event[3]:9.2f} | {true_label:9d} |"
                            f" {pred_label:10d}"
                        )

                    pred_idx += 1
                bsr_count += 1

    print("-" * 45)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate request prediction models"
    )
    parser.add_argument(
        "--train", nargs="*", help="Path to training data files (.npy)"
    )
    parser.add_argument("--val", help="Path to validation data file (.npy)")
    parser.add_argument(
        "--base-dir",
        default="labeled_data",
        help="Base directory for data files and output (default: labeled_data)",
    )
    parser.add_argument(
        "--output",
        default="models",
        help="Output directory name (will be created under base-dir)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="0,1,2,3,4,5,6,7,8",
        help=(
            "Comma-separated list of feature indices to use (0-8). Features"
            " are: 0:BSR Difference, 1:Total PRBs, 2:SR Count, 3:End BSR, 4:BSR"
            " per PRB, 5:Window Duration, 6:BSR Update Rate, 7:SR Rate, 8:Time"
            " Until PRB"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train-val split"
    )
    parser.add_argument(
        "--label-type",
        type=str,
        choices=["bsr_only", "first_bsr"],
        default="bsr_only",
        help="Type of label to use",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Number of BSR intervals to include in each window (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["decision_tree", "random_forest", "xgboost", "lightgbm"],
        help="Model to use for evaluation",
    )

    args = parser.parse_args()

    # Create full output path
    output_dir = os.path.join(args.base_dir, args.output)

    # Convert feature string to indices
    feature_indices = [int(i) for i in args.features.split(",")]

    # Validate feature indices
    if not all(
        0 <= i <= 8 for i in feature_indices
    ):  # Now just check if indices are valid for one interval
        raise ValueError(f"Feature indices must be between 0 and 8")

    try:
        if args.model:
            if not args.val:
                print(
                    "Error: Validation file must be provided when using --model"
                )
                return

            # Load model and scaler
            model_path = os.path.join(
                "labeled_data/models", f"{args.label_type}_{args.model}.joblib"
            )
            scaler_path = os.path.join(
                "labeled_data/models", f"{args.label_type}_scaler.joblib"
            )

            if not os.path.exists(model_path) or not os.path.exists(
                scaler_path
            ):
                print(f"Error: Model or scaler not found at {model_path}")
                return

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            val_data = np.load(args.val, allow_pickle=True).item()

            # Get predictions and print statistics
            predictions = evaluate_and_record_predictions(
                model,
                scaler,
                val_data,
                args.model,
                feature_indices,
                args.window_size,
            )

            # Get full events file path
            full_events_path = args.val.replace(
                "_bsr_only.npy", "_full_events.npy"
            )
            if os.path.exists(full_events_path):
                print_mismatched_predictions(
                    full_events_path, predictions, args.window_size
                )
                print_events_with_predictions(
                    full_events_path, predictions, args.window_size
                )

            return

        if args.val and not check_file_label_type(args.val, args.label_type):
            print(
                "Error: Validation file does not match label type"
                f" '{args.label_type}'"
            )
            print(f"File: {args.val}")
            return

        # Check training files label type if provided
        if args.train:
            mismatched_files = [
                f
                for f in args.train
                if not check_file_label_type(f, args.label_type)
            ]
            if mismatched_files:
                print(
                    "Error: Some training files do not match label type"
                    f" '{args.label_type}':"
                )
                for f in mismatched_files:
                    print(f"  {f}")
                return

        if args.train is None and args.val is None:  # Full auto mode
            print(f"\nAuto mode - Using all available {args.label_type} data")
            results = auto_train_and_evaluate(
                output_dir,
                feature_indices,
                args.seed,
                args.label_type,
                args.window_size,
            )

        elif args.train is None and args.val:  # Only validation file provided
            if not os.path.exists(args.val):
                print(f"Error: Validation file not found: {args.val}")
                return

            # Find all training files except the validation file
            all_files = find_all_data_files(args.label_type, args.base_dir)
            val_file_abs = os.path.abspath(args.val)
            train_files = [
                f for f in all_files if os.path.abspath(f) != val_file_abs
            ]

            if not train_files:
                print("Error: No training files found")
                return

            print(
                "\nAuto-train mode with specified validation - Using"
                f" {args.label_type} data"
            )
            print("Training files:")
            for f in train_files:
                print(f"  {f}")
            print(f"Validation file: {args.val}")
            print(f"Using features: {args.features}")

            # Use original training function to get per-RNTI evaluation
            results = train_and_evaluate(
                train_files,
                args.val,
                output_dir,
                args.label_type,
                feature_indices,
                args.window_size,
            )

        elif args.train and args.val:  # Manual mode
            # Check if files exist
            for train_file in args.train:
                if not os.path.exists(train_file):
                    print(f"Error: Training file not found: {train_file}")
                    return

            if not os.path.exists(args.val):
                print(f"Error: Validation file not found: {args.val}")
                return

            # Use original training function
            print(f"\nManual mode - Using {args.label_type} data")
            print("Training files:")
            for file in args.train:
                print(f"  {file}")
            print(f"Validation file: {args.val}")
            print(f"Using features: {args.features}")

            results = train_and_evaluate(
                args.train,
                args.val,
                output_dir,
                args.label_type,
                feature_indices,
                args.window_size,
            )

        else:  # Invalid combination
            print("Error: Invalid argument combination. Use either:")
            print("  1. No arguments (auto mode)")
            print("  2. Only --val (auto-train mode)")
            print("  3. Both --train and --val (manual mode)")
            return

        # Get full events file path
        full_events_path = args.val.replace("_bsr_only.npy", "_full_events.npy")
        if os.path.exists(full_events_path):
            print_mismatched_predictions(
                full_events_path,
                results[args.model]["predictions"],
                args.window_size,
            )
            print_events_with_predictions(
                full_events_path,
                results[args.model]["predictions"],
                args.window_size,
            )

    except Exception as e:
        print(f"Error: {str(e)}")
        return


if __name__ == "__main__":
    main()
