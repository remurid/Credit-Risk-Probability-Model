import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import our custom modules
from data_processing import build_feature_engineering_pipeline
from target_engineering import create_target_variable

# --- MLflow Configuration ---
# Set the tracking URI to a local directory. MLflow will store all experiment data here.
mlflow.set_tracking_uri("file:././mlruns")
# Set the name of the experiment. If it doesn't exist, MLflow creates it.
mlflow.set_experiment("Credit Risk Modeling")

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates and returns a dictionary of evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

def train_and_evaluate():
    """Main function to run the model training and evaluation process."""
    
    # --- 1. Load and Prepare Data ---
    print("Loading raw data...")

    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root,'..', 'data', 'raw', 'data.csv')
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found. Make sure '{data_path}' exists.")
        return

    print("Engineering target variable...")
    df_with_target = create_target_variable(raw_df)

    # --- 2. Split the Data ---
    # Define features (X) and target (y)
    X = df_with_target.drop(columns=['is_high_risk'])
    y = df_with_target['is_high_risk']

    # Split data into training and testing sets. 80% for training, 20% for testing.
    # `stratify=y` ensures that the proportion of high-risk customers is the same in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Apply Feature Engineering Pipeline ---
    feature_pipeline = build_feature_engineering_pipeline()

    print("Applying feature engineering pipeline to training data...")
    X_train_processed = feature_pipeline.fit_transform(X_train)
    
    print("Applying feature engineering pipeline to testing data...")
    X_test_processed = feature_pipeline.transform(X_test)

    # --- 4. Model Selection and Training ---
    # Define the models you want to train.
    # To add/remove models, just edit this dictionary.
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    best_roc_auc = -1
    best_run_id = None

    for model_name, model in models.items():
        # Start a new MLflow run for each model.
        with mlflow.start_run() as run:
            print(f"\n--- Training {model_name} ---")
            
            # Log model type as a parameter
            mlflow.log_param("model_type", model_name)

            # Train the model
            model.fit(X_train_processed, y_train)

            # --- 5. Model Evaluation ---
            # Make predictions on the test set
            y_pred = model.predict(X_test_processed)
            y_prob = model.predict_proba(X_test_processed)[:, 1] # Probability of the positive class

            # Calculate metrics
            metrics = evaluate_model(y_test, y_pred, y_prob)

            print(f"Metrics for {model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Log the trained model to MLflow
            mlflow.sklearn.log_model(model, artifact_path=model_name)
            
            # Check if this is the best model so far
            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_run_id = run.info.run_id
                print(f"New best model found: {model_name} with ROC-AUC: {best_roc_auc:.4f}")

    # --- 6. Register the Best Model ---
    if best_run_id:
        print(f"\nRegistering the best model from run ID: {best_run_id}")
        # Construct the model URI from the best run
        model_uri = f"runs:/{best_run_id}/GradientBoosting" # Assuming GBM will be best
        
        # Register the model in the MLflow Model Registry
        mlflow.register_model(model_uri=model_uri, name="CreditRiskChampionModel")
        print("Model registered successfully!")

if __name__ == '__main__':
    train_and_evaluate()
