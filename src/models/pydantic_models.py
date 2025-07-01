from pydantic import BaseModel

# --- Pydantic Models for Data Validation ---


class PredictionRequest(BaseModel):
    """
    Defines the structure for a single prediction request.
    These fields match the raw data columns your model was trained on.
    The API will automatically validate that incoming data matches this structure.
    """

    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    TransactionStartTime: str  # e.g., "2025-01-01T10:00:00"
    PricingStrategy: int
    FraudResult: (
        int  # This would typically be a placeholder like 0 in a real request
    )


class PredictionResponse(BaseModel):
    """
    Defines the structure for the API's prediction response.
    """

    risk_probability: float
    is_high_risk: bool
    risk_level: str
