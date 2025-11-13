import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np


def main(data_path: str = 'sample_data/parkinsons.data', output_model: str = 'parkinson_model.pkl'):
    print(" Training Parkinson's Disease Detection Model...")

    try:
        # Load the dataset
        data = pd.read_csv(data_path)
        print(" Dataset loaded successfully!")
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
    except Exception as e:
        print(f" Error: Could not load dataset from '{data_path}': {e}")
        print("Please download the dataset from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
        print("And place it in the 'sample_data' folder or pass --data <path> to the script")
        return
    # Check the target variable
    if 'status' not in data.columns:
        print(" Error: 'status' column not found in dataset")
        print("Available columns:", data.columns.tolist())
        return

    # Features to use (based on what we can extract from audio)
    available_features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
        'HNR'
    ]

    # Check which features are available
    features_to_use = []
    for feature in available_features:
        if feature in data.columns:
            features_to_use.append(feature)
        else:
            print(f" Feature not available: {feature}")

    print(f" Using {len(features_to_use)} features: {features_to_use}")

    # Prepare data
    X = data[features_to_use]
    y = data['status']

    print(f" Data prepared: X shape {X.shape}, y shape {y.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Data split: Train {X_train.shape}, Test {X_test.shape}")

    # Train a Random Forest model
    print(" Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(" Model Evaluation:")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Training score: {model.score(X_train, y_train):.4f}")
    print(f"Test score: {model.score(X_test, y_test):.4f}")

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': features_to_use,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(10))

    # Save the model
    joblib.dump(model, output_model)
    print(f"\n Model saved as '{output_model}'")

    print(" Model training completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Parkinson's disease detection model from UCI dataset or a CSV of features.")
    parser.add_argument('--data', '-d', default='sample_data/parkinsons.data', help='Path to the parkinsons.data CSV file')
    parser.add_argument('--output', '-o', default='parkinson_model.pkl', help='Output filename for the trained model')
    args = parser.parse_args()
    main(data_path=args.data, output_model=args.output)