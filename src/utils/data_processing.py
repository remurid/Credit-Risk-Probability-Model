import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# --- Custom Transformers for Feature Engineering ---
# To integrate our custom feature engineering logic into a scikit-learn Pipeline,
# we create custom classes that inherit from BaseEstimator and TransformerMixin.
# This makes our code modular, reusable, and compatible with the scikit-learn ecosystem.


class AggregateFeatureCreator(BaseEstimator, TransformerMixin):
    """
    A custom transformer to create aggregate features for each customer.
    This transformer calculates statistics based on a customer's transaction history.
    """

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data,
        # so we just return self.
        return self

    def transform(self, X, y=None):
        # Ensure the input is a DataFrame
        X_copy = X.copy()

        # --- Create Aggregate Features ---
        # Group by 'CustomerId' to calculate metrics for each customer.
        # We use .agg() to compute multiple statistics at once.
        customer_agg_features = (
            X_copy.groupby("CustomerId")["Amount"]
            .agg(
                [
                    "sum",  # Total Transaction Amount
                    "mean",  # Average Transaction Amount
                    "count",  # Transaction Count
                    "std",  # Standard Deviation of Transaction Amounts
                ]
            )
            .reset_index()
        )

        # Rename columns to be more descriptive.
        customer_agg_features.columns = [
            "CustomerId",
            "TotalTransactionAmount",
            "AverageTransactionAmount",
            "TransactionCount",
            "StdDevTransactionAmount",
        ]

        # Fill NaN values in StdDevTransactionAmount (which occur for customers
        # with only one transaction) with 0.
        customer_agg_features["StdDevTransactionAmount"] = (
            customer_agg_features["StdDevTransactionAmount"].fillna(0)
        )

        # Merge these new features back into the original DataFrame.
        X_copy = pd.merge(
            X_copy, customer_agg_features, on="CustomerId", how="left"
        )

        return X_copy


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer to extract time-based features from the 'TransactionStartTime' column.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        # --- Extract Features ---
        # Convert 'TransactionStartTime' to datetime objects.
        X_copy["TransactionStartTime"] = pd.to_datetime(
            X_copy["TransactionStartTime"]
        )

        # Extract various time-based features.
        X_copy["TransactionHour"] = X_copy["TransactionStartTime"].dt.hour
        X_copy["TransactionDay"] = X_copy["TransactionStartTime"].dt.day
        X_copy["TransactionDayOfWeek"] = X_copy[
            "TransactionStartTime"
        ].dt.dayofweek  # Monday=0, Sunday=6
        X_copy["TransactionMonth"] = X_copy["TransactionStartTime"].dt.month
        X_copy["TransactionYear"] = X_copy["TransactionStartTime"].dt.year

        # We can also create features that measure time elapsed.
        # For example, days since the first transaction for that customer.
        X_copy["DaysSinceFirstTransaction"] = (
            X_copy["TransactionStartTime"]
            - X_copy.groupby("CustomerId")["TransactionStartTime"].transform(
                "min"
            )
        ).dt.days

        return X_copy


# --- Building the Main Data Processing Pipeline ---


def build_feature_engineering_pipeline():
    """
    This function assembles the full data processing pipeline using
    scikit-learn's Pipeline and ColumnTransformer.
    """

    # Define which columns are numerical and which are categorical.
    # We will process these column types differently.
    # Note: We create the aggregate and time features first, then classify the new columns.

    # We will identify columns after the custom transformations are applied.
    # Let's define the initial categorical features.
    initial_categorical_features = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
    ]

    # --- Preprocessing Steps for Different Column Types ---

    # Create a pipeline for NUMERICAL features.
    # Step 1: Handle Missing Values - Impute missing values with the median.
    #          (Even though our EDA showed no missing values, this is a best practice for production code).
    # Step 2: Normalize/Standardize Numerical Features - Scale features to have a mean of 0 and a standard deviation of 1.
    #          This is crucial for models sensitive to feature scales (like Logistic Regression or SVMs).
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Create a pipeline for CATEGORICAL features.
    # Step 1: Handle Missing Values - Impute with a constant value 'missing'.
    # Step 2: Encode Categorical Variables - Convert categories into numerical format using One-Hot Encoding.
    #          handle_unknown='ignore' ensures that if a new category appears in future data, it doesn't cause an error.
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # --- Assembling the Full Pipeline ---

    # We use a custom transformer to select the correct column types *after* new features are created.
    class ColumnSelector(BaseEstimator, TransformerMixin):
        def __init__(self, dtype):
            self.dtype = dtype

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X.select_dtypes(include=self.dtype)

    # The ColumnTransformer applies different transformers to different columns.
    # We must specify the final list of numerical and categorical columns that will exist
    # AFTER the AggregateFeatureCreator and TimeFeatureExtractor have run.
    #
    # Final numerical columns will include original ones and newly created ones.
    final_numerical_features = [
        "Amount",
        "Value",
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdDevTransactionAmount",
        "TransactionHour",
        "TransactionDay",
        "TransactionDayOfWeek",
        "TransactionMonth",
        "TransactionYear",
        "DaysSinceFirstTransaction",
    ]

    # Final categorical features are the initial ones.
    final_categorical_features = initial_categorical_features

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, final_numerical_features),
            ("cat", categorical_transformer, final_categorical_features),
        ],
        remainder="drop",  # Drop columns that are not specified (like IDs and original timestamp)
    )

    # --- Create the Final Full Pipeline ---
    # This chains all our custom transformers and the preprocessor together.
    # The output of this pipeline will be a model-ready NumPy array.
    full_pipeline = Pipeline(
        steps=[
            ("aggregate_creator", AggregateFeatureCreator()),
            ("time_extractor", TimeFeatureExtractor()),
            ("preprocessor", preprocessor),
        ]
    )

    return full_pipeline


# --- Example of How to Use the Pipeline ---

if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    print("Running feature engineering pipeline example...")

    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "..", "data", "raw", "data.csv")
    # Load the raw data
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(
            "Error: The data file was not found. Please ensure 'data.csv' is in the '../data/raw/' directory."
        )
        exit()

    # Build the pipeline
    feature_pipeline = build_feature_engineering_pipeline()

    # Fit and transform the data
    # In a real scenario, you would fit_transform on the training set and only transform on the test set.
    print("Applying the pipeline to the raw data...")
    processed_data = feature_pipeline.fit_transform(raw_df)

    print("\nPipeline execution complete.")
    print(f"Shape of the processed data: {processed_data.shape}")
    print("The processed data is now a NumPy array, ready for model training.")
    print("\nFirst 5 rows of the processed data:")
    print(processed_data[:5])
