import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from sklearn.preprocessing import StandardScaler

def extract_window_features(events, start_idx, end_idx):
    """
    Extract features for a window between two BSRs
    Args:
        events: list of events
        start_idx: index of first BSR
        end_idx: index of second BSR
    Returns:
        features: [bsr_diff, total_prbs, sr_count]
    """
    start_bsr = events[start_idx]
    end_bsr = events[end_idx]
    
    # Calculate BSR difference
    bsr_diff = end_bsr[1] - start_bsr[1]  # index 1 is BSR bytes
    
    # Calculate total PRBs in window
    total_prbs = 0
    sr_count = 0
    
    # Process events between BSRs
    for event in events[start_idx+1:end_idx]:
        if event[0] == 2:  # PRB event
            total_prbs += event[2]  # index 2 is PRBs
        elif event[0] == 0:  # SR event
            sr_count += 1
            
    return np.array([bsr_diff, total_prbs, sr_count])

def prepare_training_data(labeled_data):
    """
    Prepare window-based features from labeled data with normalization
    Returns:
        features: numpy array of shape (n_windows, n_features)
        labels: numpy array of shape (n_windows,)
    """
    features = []
    labels = []
    
    for rnti, events in labeled_data.items():
        # Find all BSR events
        bsr_indices = [i for i, event in enumerate(events) if event[0] == 1]
        
        # Create windows between consecutive BSRs
        for i in range(len(bsr_indices)-1):
            start_idx = bsr_indices[i]
            end_idx = bsr_indices[i+1]
            
            window_features = extract_window_features(events, start_idx, end_idx)
            window_label = events[end_idx][5]
            
            features.append(window_features)
            labels.append(window_label)
    
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

def train_and_evaluate(train_data_file, val_data_file, output_dir, label_type):
    """
    Train and evaluate the model using normalized window-based features
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_data = np.load(train_data_file, allow_pickle=True).item()
    val_data = np.load(val_data_file, allow_pickle=True).item()
    
    # Prepare data
    X_train, y_train = prepare_training_data(train_data)
    X_val, y_val = prepare_training_data(val_data)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("\nFeature statistics before normalization:")
    for i, name in enumerate(['BSR Difference', 'Total PRBs', 'SR Count']):
        print(f"{name}:")
        print(f"  Mean: {X_train[:, i].mean():.2f}")
        print(f"  Std: {X_train[:, i].std():.2f}")
        print(f"  Min: {X_train[:, i].min():.2f}")
        print(f"  Max: {X_train[:, i].max():.2f}")
    
    print("\nFeature statistics after normalization:")
    for i, name in enumerate(['BSR Difference', 'Total PRBs', 'SR Count']):
        print(f"{name}:")
        print(f"  Mean: {X_train_scaled[:, i].mean():.2f}")
        print(f"  Std: {X_train_scaled[:, i].std():.2f}")
        print(f"  Min: {X_train_scaled[:, i].min():.2f}")
        print(f"  Max: {X_train_scaled[:, i].max():.2f}")
    
    print("\nTraining data statistics:")
    print(f"Number of windows: {len(X_train)}")
    print(f"Positive labels: {np.sum(y_train == 1)}")
    print(f"Negative labels: {np.sum(y_train == 0)}")
    
    # Feature names for interpretation
    feature_names = [
        'BSR Difference',
        'Total PRBs',
        'SR Count'
    ]
    
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
        )
    }
    
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name} for {label_type}...")
        
        # Train with normalized features
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on normalized validation set
        y_pred = model.predict(X_val_scaled)
        
        # Print results
        print(f"\nValidation Results for {model_name}:")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        
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

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate request prediction models')
    parser.add_argument('train_file', help='Path to training data file (.npy)')
    parser.add_argument('val_file', help='Path to validation data file (.npy)')
    parser.add_argument('--output', default='models', help='Output directory for trained models')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.train_file):
        print(f"Error: Training file not found: {args.train_file}")
        return
    
    if not os.path.exists(args.val_file):
        print(f"Error: Validation file not found: {args.val_file}")
        return
    
    # Get label type from filename
    label_type = os.path.basename(args.train_file).split('_')[-1].split('.')[0]
    
    print(f"\nProcessing {label_type} labels...")
    print(f"Training file: {args.train_file}")
    print(f"Validation file: {args.val_file}")
    
    results = train_and_evaluate(args.train_file, args.val_file, args.output, label_type)

if __name__ == "__main__":
    main()
