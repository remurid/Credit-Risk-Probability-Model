import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os

# Suppress warnings for cleaner output, particularly from K-Means on older scikit-learn versions.
warnings.filterwarnings('ignore', category=FutureWarning)

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the entire process of creating the proxy target variable.
    
    Args:
        df (pd.DataFrame): The raw transaction DataFrame.
        
    Returns:
        pd.DataFrame: The original DataFrame with the new 'is_high_risk' target column added.
    """
    
    # --- 1. Calculate RFM Metrics ---
    # This function calculates Recency, Frequency, and Monetary values for each customer.
    print("Step 1: Calculating RFM metrics...")
    rfm_df = calculate_rfm_metrics(df)

    # --- 2. Cluster Customers ---
    # This function uses K-Means to segment customers based on their RFM profiles.
    print("Step 2: Clustering customers using K-Means...")
    rfm_df_clustered = cluster_customers(rfm_df)

    # --- 3. Define and Assign the 'High-Risk' Label ---
    # This function identifies the high-risk cluster and creates the target variable.
    print("Step 3: Defining and assigning the high-risk label...")
    high_risk_customers = define_high_risk_label(rfm_df_clustered)

    # --- 4. Integrate the Target Variable ---
    # Merge the new 'is_high_risk' column back into the main DataFrame.
    print("Step 4: Integrating the target variable into the main dataset...")
    df_with_target = pd.merge(df, high_risk_customers, on='CustomerId', how='left')
    
    # Fill any potential NaNs in the target column with 0 (low-risk).
    # This can happen if a customer in the original df wasn't in the RFM calculation for some reason.
    df_with_target['is_high_risk'] = df_with_target['is_high_risk'].fillna(0).astype(int)
    
    return df_with_target

def calculate_rfm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, and Monetary (RFM) metrics for each customer.
    """
    # Ensure 'TransactionStartTime' is in datetime format.
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Define a snapshot date for calculating recency consistently.
    # We'll set it to one day after the last transaction in the dataset.
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    # Calculate RFM values
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days, # Recency
        'TransactionId': 'count',                                              # Frequency
        'Amount': 'sum'                                                        # Monetary
    })

    # Rename the columns for clarity.
    rfm.rename(columns={'TransactionStartTime': 'Recency',
                        'TransactionId': 'Frequency',
                        'Amount': 'MonetaryValue'}, inplace=True)

    return rfm

def cluster_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Segments customers into 3 clusters using K-Means based on their RFM profiles.
    """
    # Pre-process (scale) the RFM features before clustering.
    # This is crucial because K-Means is sensitive to the scale of features.
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)

    # Use the K-Means clustering algorithm.
    # n_clusters=3: To segment into three groups (e.g., low, medium, high engagement).
    # init='k-means++': A smart initialization method for faster convergence.
    # random_state=42: To ensure the results are reproducible every time the code is run.
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)

    # Add the cluster labels back to the original RFM DataFrame.
    rfm_df['Cluster'] = kmeans.labels_
    
    return rfm_df

def define_high_risk_label(rfm_df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the clusters and assigns the 'is_high_risk' label.
    """
    # Analyze the resulting clusters by calculating the mean RFM values for each cluster.
    cluster_summary = rfm_df_clustered.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean'
    }).round(2)

    print("\n--- RFM Cluster Analysis ---")
    print(cluster_summary)

    # Identify the high-risk cluster.
    # The highest-risk customers are the least engaged. We look for the cluster with:
    # - High Recency (they haven't transacted in a long time)
    # - Low Frequency (they transact rarely)
    # - Low MonetaryValue (they spend little)
    high_risk_cluster_id = cluster_summary['Recency'].idxmax()

    print(f"\nIdentified Cluster {high_risk_cluster_id} as the high-risk segment (highest recency).")

    # Create the 'is_high_risk' column.
    # Assign 1 to customers in the high-risk cluster, and 0 to all others.
    rfm_df_clustered['is_high_risk'] = rfm_df_clustered['Cluster'].apply(lambda x: 1 if x == high_risk_cluster_id else 0)
    
    return rfm_df_clustered[['is_high_risk']]


# --- Example of How to Use the Script ---

if __name__ == '__main__':
    # This block runs only when the script is executed directly.
    print("Running proxy target variable engineering example...")

    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root,'..', 'data', 'raw', 'data.csv')

    # Load the raw data
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The data file was not found. Please ensure '{data_path}' exists.")
        exit()

    # Create the target variable and add it to the DataFrame.
    df_with_target = create_target_variable(raw_df)

    print("\n--- Target Variable Integration Complete ---")
    print("Columns in the final DataFrame:", df_with_target.columns.tolist())
    print("\nValue counts for the new 'is_high_risk' target variable:")
    print(df_with_target['is_high_risk'].value_counts(normalize=True).round(4))
    
    print("\nFirst 5 rows with the new target variable:")
    print(df_with_target[['TransactionId', 'CustomerId', 'is_high_risk']].head())