# Telco Customer Churn Predictor

A machine learning web application that predicts whether a telecommunication customer is likely to churn (leave the company) based on their profile and usage patterns.

## 📊 Dataset

- **Source**: Kaggle Telco Customer Churn Dataset
- **Features**: 19 customer attributes
- **Target**: Binary classification (Churn: Yes/No)

## 🛠️ Technologies Used

### Machine Learning
- **Algorithm**: Logistic Regression
- **Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Language**: Python

### Deployment
- **Cloud Platform**: Microsoft Azure ML
- **Model Hosting**: Azure ML Endpoints
- **API**: REST API with authentication

### Frontend
- **Framework**: Streamlit-

## 📋 Features Analyzed

The model analyzes 19 customer attributes including:
- Demographics (age, gender, dependents)
- Services (phone, internet, streaming)
- Contract details (type, payment method, billing)
- Usage patterns (tenure, charges)

## 🔮 Sample Prediction

The app provides predictions in the format:
- "The customer is likely to STAY with a probability of 85.3%" 
- "The customer is likely to LEAVE with a probability of 73.2%"

## 📁 Project Structure

```
telco-customerchurn/
├── data/
│   └── telco_customer_churn.csv    # Original dataset
|
├── notebooks/
│   ├── 01_eda.ipynb      # Data analysis and exploration  
│   
├── src/
│   ├── train.py                    # Model training script
│   ├── score.py                    # Model scoring script   
|
├── outputs/
│   └── model.joblib               # Trained model file for local testing
|
├── tests/
│   └── sample_record.json         # Sample test data
│   └── test_local.py              # Script for local testing
│   └── test_endpoint.py           # Script for Azure ML endpoint testing
|
├── deploy/
│   └── deployment.yml             # Azure ML deployment configuration
│   └── endpoint.yml               # Azure ML endpoint configuration
|
├── environment.yml                # Conda environment dependencies
├── azureml-environment.yml        # Azure ML environment definition
├── streamlit_app.py              # Streamlit web application
├── .env                          # Environment variables file (not tracked)
├── .env.example                  # Environment variables example file
├── .gitignore                    # Git ignore file
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```
