from fastapi import FastAPI
from typing import List, Dict, Any
import joblib
import numpy as np
from pathlib import Path
import pandas as pd
import json

app = FastAPI()

MODEL_PATH = Path("model/model.joblib")
model = joblib.load(MODEL_PATH)

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

def preprocess_input(df):
    """
    Apply the same data cleaning as in training/scoring.
    Matches the clean_telco_data_for_scoring() function from score.py
    """
    # Standardize column names to lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Convert specific Yes/No columns to 1/0
    yes_no_like = {"partner", "dependents", "phoneservice", "paperlessbilling"}
    for col in yes_no_like.intersection(set(NUMERIC_COLS)):
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].str.strip().map({"Yes": 1, "No": 0})
    
    # Convert all numeric columns to numeric type
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].dtype == "object":
            # Try to coerce to numeric; errors become NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: List[Dict[str, Any]]):    

    df = pd.DataFrame(payload)
    
    # Preprocess the input data
    df = preprocess_input(df)

    predictions = model.predict(df)

    probabilities = np.full(len(predictions), 0.5)  #  safe fallback of 0.5 probabilities for churn

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        proba = model.predict_proba(df)

        # Probability of churn = class '1' (assuming binary 0/1 labels)
        if 1 in model.classes_:
            churn_index = list(model.classes_).index(1)
            probabilities = proba[:, churn_index]    

    response_payload = {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }

    return json.dumps(response_payload)
