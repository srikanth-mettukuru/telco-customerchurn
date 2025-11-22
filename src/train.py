# Training script for Telco Customer Churn Prediction that AzureML can run
    # Load the data
    # Clean the data
    # Split into train and test sets
    # Train the model after creating a full pipeline with preprocessing (transforming the data as necessary) and classifier
    # Evaluate the model
    # Log metrics
    # Save the model

import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(args):

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    # Start MLflow run
    mlflow.start_run()

    print("Loading data...")
    df = pd.read_csv(args.data_path)

    print("Cleaning data...")
    df = clean_telco_data(df)

    # Separate target column
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Full training pipeline
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.joblib")

    mlflow.sklearn.save_model(model, model_path)
    print(f"Model saved to {model_path}")

    mlflow.end_run()


def clean_telco_data(df):
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop customerID if present
    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])

    # Convert TotalCharges to numeric
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
        df["totalcharges"] = df["totalcharges"].fillna(df["totalcharges"].median())

    # Identify Yes/No columns
    yes_no_cols = []
    for col in df.columns:
        uniques = df[col].dropna().unique().tolist()
        if set(uniques) == {"Yes", "No"}:
            yes_no_cols.append(col)

    # Convert Yes/No → 1/0
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save the trained model"
    )

    args = parser.parse_args()

    main(args)

