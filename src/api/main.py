import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from ..models.pydantic_models import PredictionRequest, PredictionResponse

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API to predict credit risk for a BNPL "
    "service based on transaction data.",
    version="1.0.0",
)

# --- Load the Champion Model
# from MLflow Registry ---
# This is a critical step. We load the model registered as the "champion" in Task 5.
# This ensures we are using the best-performing, production-ready model.
MODEL_NAME = "CreditRiskChampionModel"
MODEL_STAGE = "@champion"

try:
    # Set the tracking URI to find the mlruns folder
    mlflow.set_tracking_uri("file:././mlruns")
    # Load the model
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}{MODEL_STAGE}"
    )
    print(
        f"Successfully loaded model '{MODEL_NAME}' from stage '{MODEL_STAGE}'."
    )
except mlflow.exceptions.MlflowException as e:
    model = None
    print(f"Error loading model: {e}")
    print("API will not be able to make predictions.")


# --- API Endpoints ---


@app.get("/", tags=["General"])
def read_root():
    """A welcome endpoint to check if the API is running."""
    return {"message": "Welcome to the Credit Risk Prediction API!"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Accepts new customer transaction data and returns a risk probability.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please check the logs.",
        )

    try:
        # 1. Convert the incoming request data into a pandas DataFrame.
        # The model's pipeline expects a DataFrame with the same structure as the training data.
        input_data = pd.DataFrame([request.dict()])

        # 2. Make a prediction.
        # The mlflow.pyfunc model automatically applies the entire feature engineering pipeline.
        # The result is the probability of the positive class (is_high_risk=1).
        risk_probability = model.predict(input_data)[0]

        # 3. Determine the final prediction and a human-readable label.
        is_high_risk = risk_probability > 0.5  # Using a standard 0.5 threshold
        risk_level = "High Risk" if is_high_risk else "Low Risk"

        return PredictionResponse(
            risk_probability=risk_probability,
            is_high_risk=is_high_risk,
            risk_level=risk_level,
        )
    except Exception as e:
        # If anything goes wrong during prediction, return an error.
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {e}"
        )
