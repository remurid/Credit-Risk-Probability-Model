import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from mlflow import MlflowClient

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
        "roc_auc": roc_auc,
    }


def train_and_evaluate():
    """Main function to run the model training and evaluation process."""

    # --- 1. Load and Prepare Data ---
    print("Loading raw data...")

    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "..", "data", "raw", "data.csv")
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(
            f"Error: Raw data file not found. Make sure '{data_path}' exists."
        )
        return

    print("Engineering target variable...")
    df_with_target = create_target_variable(raw_df)

    # --- 2. Split the Data ---
    # Define features (X) and target (y)
    X = df_with_target.drop(columns=["is_high_risk"])
    y = df_with_target["is_high_risk"]

    # Split data into training and testing sets. 80% for training, 20% for testing.
    # `stratify=y` ensures that the proportion of high-risk customers is the same in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Build Full Pipeline (Feature Engineering + Model) ---
    feature_pipeline = build_feature_engineering_pipeline()

    # --- 4. Model Selection and Training ---
    models = {
        "LogisticRegression": LogisticRegression(
            random_state=42, max_iter=1000
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    best_roc_auc = -1
    best_run_id = None
    best_model_name = None

    for model_name, estimator in models.items():
        # Build a full pipeline: feature engineering + estimator
        full_pipeline = Pipeline(
            [("features", feature_pipeline), ("estimator", estimator)]
        )

        with mlflow.start_run() as run:
            print(f"\n--- Training {model_name} ---")
            mlflow.log_param("model_type", model_name)

            # Fit the full pipeline
            full_pipeline.fit(X_train, y_train)

            # Predict on test set
            y_pred = full_pipeline.predict(X_test)
            y_prob = full_pipeline.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = evaluate_model(y_test, y_pred, y_prob)

            print(f"Metrics for {model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

            mlflow.log_metrics(metrics)

            # Log the full pipeline (feature engineering + model)
            # mlflow.sklearn.log_model(full_pipeline, artifact_path=model_name)
            mlflow.sklearn.log_model(
                full_pipeline,
                artifact_path=model_name,
                code_paths=[
                    "src/utils/data_processing.py",
                    "src/utils/target_engineering.py",
                ],
            )

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_run_id = run.info.run_id
                best_model_name = model_name
                print(
                    f"New best model found: {model_name} with ROC-AUC: {best_roc_auc:.4f}"
                )

    # --- 5. Register the Best Model ---
    if best_run_id and best_model_name:
        print(f"\nRegistering the best model from run ID: {best_run_id}")
        model_uri = f"runs:/{best_run_id}/{best_model_name}"
        mlflow.register_model(
            model_uri=model_uri, name="CreditRiskChampionModel"
        )
        print("Model registered successfully!")

        client = MlflowClient()
        # Set the "champion" alias to the
        # latest version of the registered model
        latest_version = client.get_latest_versions(
            "CreditRiskChampionModel", stages=["None"]
        )[0].version
        client.set_registered_model_alias(
            "CreditRiskChampionModel", "champion", latest_version
        )


if __name__ == "__main__":
    train_and_evaluate()
