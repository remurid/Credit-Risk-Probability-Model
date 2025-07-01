# Project Name
credit-risk-model

# Project Overview

This repository contains a credit risk modeling project for Bati Bank, focused on enabling a "Buy-Now-Pay-Later" service in partnership with an eCommerce company. The project includes:
- A robust data processing pipeline for transforming raw data into model-ready features.
- Exploratory Data Analysis (EDA) in a Jupyter Notebook to uncover patterns and guide feature engineering.
- Scripts and documentation to support reproducible and interpretable credit scoring models.

# Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd credit-risk-model
   ```
2. **Install Dependencies**
   - Create a virtual environment (recommended):
     ```bash
     python -m venv venv
     venv\Scripts\activate  # On Windows
     # Or
     source venv/bin/activate  # On Mac/Linux
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
3. **Project Structure**

   credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md


4. **Running the Data Processing Pipeline**
   - From the project root, run:
     ```bash
     python src/data_processing.py
     ```
   - This will process the raw data and output model-ready features.

5. **Exploratory Data Analysis**
   - Open `notebooks/1.0-eda.ipynb` in Jupyter or VS Code to review the EDA process and insights.

# Project Goals
- Develop a transparent, interpretable, and regulatory-compliant credit scoring model.
- Engineer features that capture customer behavior and risk.
- Provide actionable insights through EDA and robust data pipelines.

# Documentation
- The README provides an overview, setup instructions, and a summary of the workflow.
- See the `docs/` folder for additional details and references.

# Contact
For questions or contributions, please open an issue or submit a pull request.

---

# Credit Scoring Business Understanding

## Overview
Credit scoring is the process of assigning a quantitative measure to a potential borrower to estimate the likelihood of default. This project focuses on developing a credit scoring model for **Bati Bank**, enabling a "Buy-Now-Pay-Later" service in partnership with an eCommerce company. The model transforms customer behavioral data into a predictive risk score to inform loan approvals and terms.

---

## What is Credit Risk?
Credit risk is the possibility that a borrower or counterparty will fail to meet their financial obligations as agreed. For example, when a bank provides a loan to an individual or a company, the bank faces the risk that the borrower may not repay the loan in full or on time. This risk can lead to financial loss for the lender.

## Credit Risk and the Basel II Accord
The Basel II Accord is a set of international banking regulations developed by the Basel Committee on Banking Supervision. It emphasizes effective risk management and requires financial institutions to hold capital in proportion to their risk exposure.


---

## Key Questions and Explanations

### 1. Basel II Accord’s Influence on Model Requirements
The Basel II Capital Accord emphasizes **risk measurement** as a core aspect of financial stability. This has significant implications for the credit scoring model:

- **Regulatory Compliance**: Basel II mandates that banks align their capital reserves with measured credit risks. Therefore, models must be interpretable and well-documented to ensure transparency and adherence to regulations.
- **Stakeholder Trust**: Auditors, regulators, and internal stakeholders need to understand and trust the model’s outcomes. A lack of interpretability can lead to non-compliance, legal issues, and reputational risks.

#### Why Interpretable Models Matter:
An interpretable model, such as logistic regression with Weight-of-Evidence (WoE), provides clear, variable-level insights into the decision-making process. This aligns with Basel II’s requirements for transparency and traceability.

---

### 2. Necessity of a Proxy Variable for Default
In the absence of direct "default" labels, creating a proxy variable is essential:

- **Why a Proxy is Needed**: Early-stage credit data lacks historical information on actual defaults, especially for new borrowers. A proxy variable acts as a stand-in, using measurable behaviors (e.g., delayed payments) to indicate credit risk.
- **Defining a Proxy**: Examples include:
  - ">30 days late" on a payment.
  - "Three consecutive missed payments."

#### Risks of Proxy Variables:
- **Misrepresentation**: Proxies may not fully capture true default behavior, leading to inaccurate predictions.
- **Business Impact**: Poorly chosen proxies can result in adverse selection, where risky customers are approved, or creditworthy customers are rejected.

#### Mitigation Strategies:
- Validate the proxy against historical data.
- Continuously refine the proxy as more actual defaults are observed.

---

### 3. Trade-offs: Simple vs. Complex Models
In regulated environments like banking, selecting the right model involves balancing interpretability, performance, and compliance.

#### Comparison:
| Factor                 | **Simple Models** <br>(e.g., Logistic Regression + WoE) | **Complex Models** <br>(e.g., Gradient Boosting) |
|------------------------|----------------------------------------------------------|--------------------------------------------------|
| **Interpretability**   | High; easy to explain variable contributions              | Low; decisions are harder to interpret          |
| **Regulatory Approval**| Easier to justify                                         | Requires additional explainability tools        |
| **Performance**        | Moderate; may miss non-linear patterns                   | High; captures complex interactions            |
| **Documentation**      | Straightforward                                          | Requires extensive efforts                      |
| **Maintenance**        | Simple to update                                         | Resource-intensive                              |
| **Risk**               | Lower; easier governance                                 | Higher; opaque decision-making                 |

#### Recommendations(my choice):
1. **Baseline Model**: Start with a logistic regression model, enhanced with Weight-of-Evidence (WoE), to satisfy Basel II requirements for interpretability and compliance.
2. **Advanced Models**: Experiment with gradient boosting or similar methods. Deploy only if performance gains justify the additional complexity and governance requirements.
3. **Explainability Tools**: Use SHAP or LIME for complex models to provide insights into decision-making.

---

## Project Objectives
1. **Define a Proxy Variable**:
   - Categorize users as high-risk (bad) or low-risk (good).
   - Ensure the proxy reflects realistic default behavior.
2. **Feature Selection**:
   - Identify observable features that correlate strongly with the default variable (e.g., Recency, Frequency, Monetary patterns).
3. **Develop Models**:
   - Assign risk probabilities for new customers.
   - Assign credit scores based on risk probabilities.
   - Predict optimal loan amounts and durations.

---

## Conclusion
The credit scoring model must balance regulatory compliance, interpretability, and predictive performance. By starting with interpretable models and cautiously exploring advanced techniques, Bati Bank can ensure robust, compliant, and effective credit risk management.



## Task 2: Exploratory Data Analysis (EDA)
### Objective

The primary goal of this Exploratory Data Analysis (EDA) is to explore the dataset to uncover patterns, identify data quality issues, and form actionable hypotheses that will guide the subsequent feature engineering and modeling phases. All exploratory work was conducted in the `notebooks/1.0-eda.ipynb` Jupyter Notebook, which is intended solely for exploration and not for production code.

---

### 1. Overview of the Data

- **Dimensions:** The dataset contains 95,662 rows (representing individual transactions) and 12 columns (features).
- **Data Types:** The columns include a mix of data types:
  - `object`: Categorical and identifier columns (e.g., TransactionId, AccountId, ProductCategory).
  - `int64`: Numerical columns (e.g., CountryCode, Value, PricingStrategy, FraudResult).
  - `float64`: The Amount column.
- **Initial Inspection:** A review of the first few rows confirmed the content and format of each column as described in the data dictionary.

---

### 2. Identifying Missing Values

- **Findings:** A thorough check revealed zero missing values across all columns in the dataset.
- **Implication:** This simplifies the data preprocessing pipeline, as no imputation or removal strategies for missing data are necessary. We can proceed directly to feature engineering.

---

### 3. Summary Statistics

- **Numerical Features (Amount, Value):**
  - The mean value for Amount is significantly different from the median (50th percentile), indicating a skewed distribution.
  - A large difference between the 75th percentile and the max value for both Amount and Value suggests the presence of outliers.
- **Categorical Features (ProductCategory, ChannelId, etc.):**
  - `ProductCategory`: 'airtime' is the most frequent category.
  - `ChannelId`: 'ChannelId_3' dominates the transactions.
  - `ProviderId`: 'ProviderId_4' and 'ProviderId_6' are the most common providers, showing a concentration of transactions with a few key providers.

---

### 4. Distribution Analysis

#### Distribution of Numerical Features

- **Observation:** Histograms for Amount and Value show both features are heavily right-skewed. Most transactions are of low value, with a long tail of very high-value transactions.
- **Implication:** This skewness can negatively impact certain machine learning models. A transformation (e.g., log transformation) will likely be necessary during feature engineering to normalize these distributions.

#### Distribution of Categorical Features

- **Observation:** There is a significant imbalance in several key categorical features:
  - `ProductCategory`: 'airtime' and 'financial_services' account for a large majority of transactions.
  - `ChannelId`: One channel is responsible for most of the transaction volume.
- **Implication:** The dominance of a few categories means these will be very influential in the model. Creating interaction features between these dominant categories could capture important predictive signals.

---

### 5. Outlier Detection

- **Observation:** Box plots for Amount and Value clearly illustrate the presence of numerous significant outliers, appearing as points far beyond the upper whisker of the plot.
- **Implication:** These outliers could represent legitimate, high-value transactions or potentially fraudulent activity. Their presence must be handled carefully, as they can disproportionately influence model training. Using robust scaling techniques (like RobustScaler) or transformations will be critical.

---

### 6. Correlation Analysis

- **Method:** A correlation matrix and heatmap were generated to understand the linear relationships between the main numerical features (Amount, Value, and the target FraudResult).
- **Observation:**
  - Amount and Value are highly correlated (0.81).
  - The linear correlation between Amount and FraudResult is extremely weak (-0.05).
- **Implication:** This lack of a strong linear relationship suggests that fraud cannot be predicted simply by the transaction amount. The underlying patterns of risk are likely more complex and non-linear, reinforcing the need for more sophisticated models (like tree-based ensembles) and detailed feature engineering.

---

### Summary of Top 5 EDA Insights & Actionable Hypotheses

1. **Insight:** The transaction Amount is heavily right-skewed with significant outliers.  
   **Hypothesis:** Normalizing the Amount feature using a log transformation will improve model performance and stability. The outliers may represent a distinct customer segment (e.g., businesses) that could be flagged with a separate binary feature.

2. **Insight:** Customer activity is highly concentrated in specific ProductCategory and ChannelId values.  
   **Hypothesis:** The specific combination of ProductCategory and ChannelId is a strong predictor of customer behavior and credit risk. Creating interaction features from these columns will capture valuable predictive patterns.

3. **Insight:** There is no direct, linear relationship between the transaction amount and the likelihood of fraud.  
   **Hypothesis:** To build an effective risk model, we must rely on non-linear models (e.g., Gradient Boosting, Random Forest) that can capture complex interactions between multiple features, rather than just simple trends.

4. **Insight:** The dataset is complete and has no missing values.  
   **Hypothesis:** The data preprocessing pipeline can be streamlined, as no time is needed for developing or implementing data imputation strategies. We can proceed directly to feature creation.

5. **Insight:** Time-based patterns may exist within the TransactionStartTime.  
   **Hypothesis:** The time of day, day of the week, or month of the transaction could be a critical predictor of risk. We must engineer features from the TransactionStartTime column to capture these potential temporal patterns.

## Task 3: Feature Engineering

Build a robust, automated, and reproducible data processing script that transforms raw data into a model-ready format.

All feature engineering logic must be implemented in the `src/` Python scripts. The main script uses `sklearn.pipeline.Pipeline` to chain together all transformation steps, ensuring modularity and reproducibility.

### Key Steps in the Feature Engineering Pipeline

1. **Create Aggregate Features**
   - *Total Transaction Amount*: Sum of all transaction amounts for each customer.
   - *Average Transaction Amount*: Average transaction amount per customer.
   - *Transaction Count*: Number of transactions per customer.
   - *Standard Deviation of Transaction Amounts*: Variability of transaction amounts per customer.

2. **Extract Features from Timestamps**
   - *Transaction Hour*: The hour of the day when the transaction occurred.
   - *Transaction Day*: The day of the month when the transaction occurred.
   - *Transaction Month*: The month when the transaction occurred.
   - *Transaction Year*: The year when the transaction occurred.

3. **Encode Categorical Variables**
   - *One-Hot Encoding*: Converts categorical values into binary vectors.
   - *Label Encoding*: Assigns a unique integer to each category (if needed).

4. **Handle Missing Values**
   - *Imputation*: Fill missing values with mean, median, mode, or advanced methods like KNN imputation.
   - *Removal*: Remove rows or columns with missing values if they are few and not critical.

5. **Normalize/Standardize Numerical Features**
   - *Normalization*: Scales the data to a range of [0, 1].
   - *Standardization*: Scales the data to have a mean of 0 and a standard deviation of 1.

---

The pipeline is designed to be flexible and can be easily extended or modified for different modeling requirements. All transformations are performed using scikit-learn compatible classes, ensuring seamless integration with downstream machine learning workflows.
