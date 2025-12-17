import json
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure src/ is on Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.api import app

client = TestClient(app)


def test_predict_endpoint():
    # Load sample input
    sample_path = Path(__file__).parent / "sample_record.json"
    with open(sample_path) as f:
        payload = json.load(f)

    # Call the API
    response = client.post("/predict", json=payload)

    print("Status code:", response.status_code)
    print("Response:", response.json())

    # Basic response checks
    assert response.status_code == 200

    result = response.json()

    # Assertions depend on your model output shape
    assert result is not None
    #assert isinstance(result, list)