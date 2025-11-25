import json
import requests
import dotenv
import os

dotenv.load_dotenv()

SCORING_URI = os.getenv("AZURE_ML_SCORING_URI")
API_KEY = os.getenv("AZURE_ML_PRIMARY_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Example payload
data = [
    {
        "seniorcitizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phoneservice": "Yes",
        "paperlessbilling": "Yes",
        "monthlycharges": 70.35,
        "totalcharges": 845.5,
        "gender": "Female",
        "multiplelines": "No",
        "internetservice": "Fiber optic",
        "onlinesecurity": "No",
        "onlinebackup": "No",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "Yes",
        "streamingmovies": "Yes",
        "contract": "Month-to-month",
        "paymentmethod": "Electronic check"
    }
]

response = requests.post(SCORING_URI, data=json.dumps(data), headers=headers)

print("Status code:", response.status_code)
print("Response:", response.json())
