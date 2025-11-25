# Score script that uses the trained model to make predictions.
# This script is used by Azure ML for deploying the model as a web service.
# It defines two main functions: init() and run().
# init() is called once when the service starts to load the model.
# run() is called for each request to make predictions.
# Can use this script for local testing as well. Just point to the local model file as fallback when AZUREML_MODEL_DIR is not set.

import os
import json
import joblib
import numpy as np
import pandas as pd

# Global model variable
model = None

# Columns the model expects (after cleaning)
NUMERIC_COLS = [
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "paperlessbilling",
    "monthlycharges",
    "totalcharges",
]

CATEGORICAL_COLS = [
    "gender",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paymentmethod",
]

EXPECTED_COLUMNS = NUMERIC_COLS + CATEGORICAL_COLS


def init():
    """
    Called once when the service starts.
    Loads the trained model from the AzureML model directory.
    """
    global model

    # AzureML sets this env var to the folder where the registered model is mounted. Empty if running locally.
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    print(f"AZUREML_MODEL_DIR: {model_dir}")   # /var/azureml-app/azureml-models/xyz-model/1

    if model_dir:                                 # Model is running as a service in AzureML
        model_path = os.path.join(model_dir, "model.joblib")
    else:                                         # Model is stored locally
        # Fallback for local testing
        model_path = os.path.join("outputs", "model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")


def _ensure_dataframe(data):
    """
    Convert input JSON (dict or list of dicts) to a pandas DataFrame
    with the expected columns.
    """
    if isinstance(data, dict):
        # Single record
        data = [data]

    df = pd.DataFrame(data)

    # Ensure all expected columns are present
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    # Extra columns are ignored; keep only what the model was trained on
    df = df[EXPECTED_COLUMNS]

    df = clean_telco_data_for_scoring(df)   

    return df


def clean_telco_data_for_scoring(df):
    """
    Apply the same data cleaning as in training, but for scoring.
    Should match the clean_telco_data() function from train.py
    """
    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()    

    yes_no_like = {"partner", "dependents", "phoneservice", "paperlessbilling"}
    for col in yes_no_like.intersection(NUMERIC_COLS):
        if df[col].dtype == "object":
            df[col] = df[col].str.strip().map({"Yes": 1, "No": 0})

    for col in NUMERIC_COLS:
        if df[col].dtype == "object":
            # Try to coerce to numeric; errors become NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def run(raw_data):
    """
    Called for each request.
    `raw_data` is a JSON string.
    Returns a JSON string with predictions (and probabilities, if available).
    """
    try:
        # Validate input
        if not raw_data:
            raise ValueError("No input data provided")

        # raw_data is a JSON string; parse it
        data = json.loads(raw_data)

        if not data:
            raise ValueError("Empty data provided")

        # Convert to DataFrame and align columns
        print("Converting input data to DataFrame...")
        df = _ensure_dataframe(data)

        # Additional validation
        if df.empty:
            raise ValueError("No valid records to predict")

        # Predict
        preds = model.predict(df)
        print(f"Predictions: {preds}")

        # Try to get churn probability (class 1) if supported
        probs = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            # Probability of churn = class '1' (assuming binary 0/1 labels)
            if 1 in model.classes_:
                churn_index = list(model.classes_).index(1)
                probs = proba[:, churn_index].tolist()

        result = {
            "predictions": preds.tolist(),
        }

        if probs is not None:
            result["probabilities"] = probs

        return json.dumps(result)
    
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON format: {str(e)}"})
    except Exception as e:
        # Return error as JSON (makes debugging easier from client)
        return json.dumps({"error": str(e)})
