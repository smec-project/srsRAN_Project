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

def extract_window_features(events, window_bsr_indices, historical_bsr_avg, window_size=1):
    """
    Extract features for a window between BSRs, with features for each BSR interval
    Args:
        events: list of events
        window_bsr_indices: list of BSR indices for the window
        historical_bsr_avg: average of up to 10 previous BSR values
        window_size: number of BSR intervals to include
    Returns:
        features: concatenated features for each BSR interval in the window
    """
    all_features = []
    
    # Process each BSR interval in the window
    for i in range(window_size):
        current_start_idx = window_bsr_indices[i]
        current_end_idx = window_bsr_indices[i + 1]
        
        start_bsr = events[current_start_idx]
        end_bsr = events[current_end_idx]
        
        # Calculate BSR difference for this interval
        bsr_diff = end_bsr[1] - start_bsr[1]
        end_bsr_value = end_bsr[1]
        
        # Calculate metrics for this interval
        total_prbs = 0
        sr_count = 0
        prb_events = 0
        
        # Process events in this interval
        for event in events[current_start_idx+1:current_end_idx]:
            if event[0] == 2:  # PRB event
                total_prbs += event[2]
                prb_events += 1
            elif event[0] == 0:  # SR event
                sr_count += 1
        
        # Calculate interval duration
        window_duration = end_bsr[3] - start_bsr[3]
        
        # Calculate rates for this interval
        bsr_per_prb = bsr_diff / (total_prbs + 1e-6)
        bsr_update_rate = 1000.0 / (window_duration + 1e-6)
        sr_rate = sr_count * 1000.0 / (window_duration + 1e-6)
        
        # Features for this interval
        interval_features = np.array([
            bsr_diff,         # Difference between BSRs
            total_prbs,       # Total PRBs allocated
            sr_count,         # Number of SR events
            end_bsr_value,    # Value of the end BSR
            bsr_per_prb,      # BSR difference normalized by PRBs
            window_duration,  # Time duration
            bsr_update_rate,  # Rate of BSR updates
            sr_rate,         # Rate of SR events
        ])
        
        all_features.append(interval_features)
    
    # Concatenate all interval features
    return np.concatenate(all_features)

def prepare_training_data(labeled_data, window_size=1):
    """
    Prepare window-based features from labeled data
    Args:
        labeled_data: dictionary of RNTI to events
        window_size: number of BSR intervals to include in each window
    Returns:
        features: numpy array of shape (n_windows, n_features * window_size)
        labels: numpy array of shape (n_windows,)
    """
    features = []
    labels = []
    
    for rnti, events in labeled_data.items():
        # Find all BSR events
        bsr_indices = [i for i, event in enumerate(events) if event[0] == 1]
        
        # Create windows with sliding window of size 1
        for i in range(len(bsr_indices) - window_size):
            # Get indices for all BSRs in this window
            window_bsr_indices = bsr_indices[i:i + window_size + 1]
            
            # Calculate historical BSR average
            historical_end = i
            historical_start = max(0, historical_end - 10)
            historical_bsrs = [events[bsr_indices[j]][1] for j in range(historical_start, historical_end)]
            historical_bsr_avg = np.mean(historical_bsrs) if historical_bsrs else events[window_bsr_indices[-1]][1]
            
            # Extract features for all intervals in window
            window_features = extract_window_features(events, window_bsr_indices, historical_bsr_avg, window_size)
            window_label = events[window_bsr_indices[-1]][5]  # Label from the last BSR
            
            features.append(window_features)
            labels.append(window_label)
    
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

def evaluate_per_rnti(model, scaler, val_data, model_name, feature_indices, window_size=1):
    """
    Evaluate model performance for each RNTI separately
    Args:
        model: trained model
        scaler: fitted scaler
        val_data: validation data dictionary
        model_name: name of the model
        feature_indices: indices of features to use
        window_size: number of BSR intervals to include in each window
    """
    print(f"\nPer-RNTI Evaluation for {model_name}:")
    
    for rnti, events in val_data.items():
        # Prepare data for this RNTI with the correct window size
        X_val_rnti, y_val_rnti = prepare_training_data({rnti: events}, window_size)
        if len(X_val_rnti) == 0:
            continue
            
        # Select features
        X_val_rnti = X_val_rnti[:, feature_indices]
            
        # Scale features
        X_val_scaled = scaler.transform(X_val_rnti)
        
        # Get predictions
        y_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        accuracy = np.mean(y_val_rnti == y_pred)
        tn, fp, fn, tp = confusion_matrix(y_val_rnti, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
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
    Combine multiple .npy files into one dataset
    Each RNTI will be prefixed with its source file name to avoid conflicts
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

def train_and_evaluate(train_files, val_file, output_dir, label_type, feature_indices, window_size=1):
    """
    Train and evaluate the model using selected features
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and combine training data
    train_data = combine_data(train_files)
    val_data = np.load(val_file, allow_pickle=True).item()
    
    # Prepare data
    X_train, y_train = prepare_training_data(train_data, window_size)
    X_val, y_val = prepare_training_data(val_data, window_size)
    
    # Select features
    X_train = X_train[:, feature_indices]
    X_val = X_val[:, feature_indices]
    
    # All feature names for one interval
    base_feature_names = [
        'BSR Difference',    # Change in BSR between two consecutive BSR events
        'Total PRBs',        # Sum of PRBs allocated in the window
        'SR Count',          # Number of SRs in the window
        'End BSR',           # Value of the second BSR
        'BSR per PRB',       # BSR difference normalized by total PRBs
        'Window Duration',    # Time duration between two BSR events
        'BSR Update Rate',    # Frequency of BSR updates
        'SR Rate',           # Frequency of SR events
    ]
    
    # Generate feature names for all intervals in the window
    all_feature_names = []
    for i in range(window_size):
        interval_names = [f"Interval_{i+1}_{name}" for name in base_feature_names]
        all_feature_names.extend(interval_names)
    
    # Selected feature names
    feature_names = [all_feature_names[i] for i in feature_indices]
    
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
        'decision_tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            class_weight='balanced',
            n_jobs=-1
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),  # Handle class imbalance
            n_jobs=-1
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=20,
            min_split_gain=1e-3,
            num_leaves=31,
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),  # Handle class imbalance
            n_jobs=-1,
            verbose=-1
        )
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
        evaluate_per_rnti(model, scaler, val_data, model_name, feature_indices, window_size)
        
        # Save model and scaler
        model_path = os.path.join(output_dir, f"{label_type}_{model_name}.joblib")
        scaler_path = os.path.join(output_dir, f"{label_type}_scaler.joblib")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        # Store results
        results[model_name] = {
            'model': model,
            'scaler': scaler,
            'predictions': y_pred,
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        
        # Print feature importance
        print("\nFeature Importance:")
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
    
    return results

def find_all_data_files(label_type, base_dir="labeled_data"):
    """
    Find all files with specified label type in base directory
    Args:
        label_type: 'bsr_only' or 'first_bsr'
        base_dir: base directory to search in
    Returns:
        list of file paths
    """
    pattern = os.path.join(base_dir, "**", f"*_{label_type}.npy")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise ValueError(f"No *_{label_type}.npy files found in {base_dir} directory")
    return sorted(files)

def auto_train_and_evaluate(output_dir, feature_indices, seed=42, label_type='bsr_only', window_size=1):
    """
    Automatically load all data files, extract features, split and train
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
    label_type = os.path.basename(all_files[0]).split('_')[-1].split('.')[0]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # All feature names for one interval
    base_feature_names = [
        'BSR Difference',    # Change in BSR between two consecutive BSR events
        'Total PRBs',        # Sum of PRBs allocated in the window
        'SR Count',          # Number of SRs in the window
        'End BSR',           # Value of the second BSR
        'BSR per PRB',       # BSR difference normalized by total PRBs
        'Window Duration',    # Time duration between two BSR events
        'BSR Update Rate',    # Frequency of BSR updates
        'SR Rate',           # Frequency of SR events
    ]
    
    # Generate feature names for all intervals in the window
    all_feature_names = []
    for i in range(window_size):
        interval_names = [f"Interval_{i+1}_{name}" for name in base_feature_names]
        all_feature_names.extend(interval_names)
    
    # Selected feature names
    feature_names = [all_feature_names[i] for i in feature_indices]
    
    print("\nUsing features:", ", ".join(feature_names))
    
    # Train models
    models = {
        'decision_tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            n_jobs=-1
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
            n_jobs=-1
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            min_child_samples=20,
            min_split_gain=1e-3,
            num_leaves=31,
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
            n_jobs=-1,
            verbose=-1
        )
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
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
        model_path = os.path.join(output_dir, f"{label_type}_{model_name}.joblib")
        scaler_path = os.path.join(output_dir, f"{label_type}_scaler.joblib")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        results[model_name] = {
            'model': model,
            'scaler': scaler,
            'predictions': y_pred,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            }
        }
        
        print("\nFeature Importance:")
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
        
        # Print feature importance ranking
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        print("\nFeature Importance Ranking:")
        for name, importance in importance_pairs:
            print(f"{name}: {importance:.4f}")
    
    return results

def check_file_label_type(file_path, expected_label_type):
    """
    Check if the file name matches the expected label type
    Returns:
        bool: True if matches, False otherwise
    """
    # Get the base name of the file and check if it ends with expected label type
    base_name = os.path.basename(file_path)
    return base_name.endswith(f"_{expected_label_type}.npy")

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate request prediction models')
    parser.add_argument('--train', nargs='*', help='Path to training data files (.npy)')
    parser.add_argument('--val', help='Path to validation data file (.npy)')
    parser.add_argument('--base-dir', default='labeled_data', 
                       help='Base directory for data files and output (default: labeled_data)')
    parser.add_argument('--output', default='models', 
                       help='Output directory name (will be created under base-dir)')
    parser.add_argument('--features', type=str, default='0,1,2,3,4,5,6,7',
                       help='Comma-separated list of feature indices to use (0-7). Features are: '
                            '0:BSR Difference, 1:Total PRBs, 2:SR Count, 3:End BSR, '
                            '4:BSR per PRB, 5:Window Duration, 6:BSR Update Rate, '
                            '7:SR Rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train-val split')
    parser.add_argument('--label-type', type=str, choices=['bsr_only', 'first_bsr'], 
                       default='bsr_only', help='Type of label to use')
    parser.add_argument('--window-size', type=int, default=1,
                       help='Number of BSR intervals to include in each window (default: 1)')
    
    args = parser.parse_args()
    
    # Create full output path
    output_dir = os.path.join(args.base_dir, args.output)
    
    # If features not specified, use all features for all intervals
    if args.features == '0,1,2,3,4,5,6,7':  # default value
        feature_indices = []
        for i in range(args.window_size):
            start_idx = i * 8  # Changed from 9 to 8
            feature_indices.extend(range(start_idx, start_idx + 8))
    else:
        feature_indices = [int(i) for i in args.features.split(',')]
    
    # Validate feature indices
    max_feature_idx = 8 * args.window_size - 1  # Changed from 9 to 8
    if not all(0 <= i <= max_feature_idx for i in feature_indices):
        raise ValueError(f"Feature indices must be between 0 and {max_feature_idx}")
    
    try:
        # Check validation file label type if provided
        if args.val and not check_file_label_type(args.val, args.label_type):
            print(f"Error: Validation file does not match label type '{args.label_type}'")
            print(f"File: {args.val}")
            return
        
        # Check training files label type if provided
        if args.train:
            mismatched_files = [f for f in args.train if not check_file_label_type(f, args.label_type)]
            if mismatched_files:
                print(f"Error: Some training files do not match label type '{args.label_type}':")
                for f in mismatched_files:
                    print(f"  {f}")
                return
        
        if args.train is None and args.val is None:  # Full auto mode
            print(f"\nAuto mode - Using all available {args.label_type} data")
            results = auto_train_and_evaluate(output_dir, feature_indices, args.seed, args.label_type, args.window_size)
            
        elif args.train is None and args.val:  # Only validation file provided
            if not os.path.exists(args.val):
                print(f"Error: Validation file not found: {args.val}")
                return
                
            # Find all training files except the validation file
            all_files = find_all_data_files(args.label_type, args.base_dir)
            val_file_abs = os.path.abspath(args.val)
            train_files = [f for f in all_files if os.path.abspath(f) != val_file_abs]
            
            if not train_files:
                print("Error: No training files found")
                return
                
            print(f"\nAuto-train mode with specified validation - Using {args.label_type} data")
            print("Training files:")
            for f in train_files:
                print(f"  {f}")
            print(f"Validation file: {args.val}")
            print(f"Using features: {args.features}")
            
            # Use original training function to get per-RNTI evaluation
            results = train_and_evaluate(train_files, args.val, output_dir, args.label_type, feature_indices, args.window_size)
            
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
            
            results = train_and_evaluate(args.train, args.val, output_dir, args.label_type, feature_indices, args.window_size)
            
        else:  # Invalid combination
            print("Error: Invalid argument combination. Use either:")
            print("  1. No arguments (auto mode)")
            print("  2. Only --val (auto-train mode)")
            print("  3. Both --train and --val (manual mode)")
            return
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()
