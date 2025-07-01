import sys
import os
import pytest
import pandas as pd
from src.utils.data_processing import (
    AggregateFeatureCreator,
    TimeFeatureExtractor,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)


# --- Test Fixture ---
# A fixture is a function that provides a fixed baseline state for tests.
# This fixture creates a small, simple DataFrame
# that we can use as input for our tests.
@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame for testing."""
    data = {
        "TransactionId": [1, 2, 3, 4],
        "CustomerId": ["A", "B", "A", "C"],
        "TransactionStartTime": [
            "2025-01-01 10:00:00",
            "2025-01-02 15:30:00",
            "2025-01-05 20:00:00",
            "2025-01-05 20:00:00",
        ],
        "Amount": [100, 200, 50, 300],
    }
    return pd.DataFrame(data)


# --- Unit Test 1: Test the TimeFeatureExtractor ---
def test_time_feature_extractor(sample_dataframe):
    """
    Tests that the TimeFeatureExtractor correctly
    creates new time-based columns.
    """
    # Arrange: Create an instance of the transformer
    transformer = TimeFeatureExtractor()

    # Act: Apply the transformer to the sample data
    transformed_df = transformer.transform(sample_dataframe)

    # Assert: Check if the new columns were created
    expected_new_columns = [
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear",
        "DaysSinceFirstTransaction",
    ]
    for col in expected_new_columns:
        assert col in transformed_df.columns

    # Assert: Check for correct values
    # For the first transaction (index 0), the hour should be 10.
    assert transformed_df.loc[0, "TransactionHour"] == 10
    # For the third transaction (index 2), the day should be 5.
    assert transformed_df.loc[2, "TransactionDay"] == 5
    # For the third transaction (index 2),
    # which is customer A's second transaction,
    # the days since their first transaction should be 4.
    assert transformed_df.loc[2, "DaysSinceFirstTransaction"] == 4


# --- Unit Test 2: Test the AggregateFeatureCreator ---
def test_aggregate_feature_creator(sample_dataframe):
    """
    Tests that the AggregateFeatureCreator correctly calculates
    customer-level statistics.
    """
    # Arrange
    transformer = AggregateFeatureCreator()

    # Act
    transformed_df = transformer.transform(sample_dataframe)

    # Assert: Check if the new columns were created
    expected_new_columns = [
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdDevTransactionAmount",
    ]
    for col in expected_new_columns:
        assert col in transformed_df.columns

    # Assert: Check for correct values for a specific customer ('A')
    # Customer A has two transactions (100 and 50)
    customer_a_data = transformed_df[transformed_df["CustomerId"] == "A"]

    # The transaction count for customer A should be 2 on both of their rows.
    assert (customer_a_data["TransactionCount"] == 2).all()
    # The total transaction amount should be 150.
    assert (customer_a_data["TotalTransactionAmount"] == 150).all()
    # The average transaction amount should be 75.
    assert (customer_a_data["AverageTransactionAmount"] == 75).all()
