import streamlit as st
import requests
import json
import dotenv
import os

dotenv.load_dotenv()

SCORING_URI = os.getenv("AZURE_ML_SCORING_URI")
API_KEY = os.getenv("AZURE_ML_PRIMARY_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Sidebar with app information
st.sidebar.header("📋 About This App")
st.sidebar.markdown("""
This application predicts whether a fictitious telecommunication customer is likely to churn (leave the company) based on their profile and usage patterns.
Uses a predictive model trained on the Kaggle Telco Customer Churn dataset.
                    
### 🎯 **How to Use**
1. Fill in the customer details using the form inputs
2. Click 'Predict' to get the churn prediction
3. Review the results showing prediction and confidence probability
""")

st.sidebar.header("🛠️ Technologies Used")
st.sidebar.markdown("""
### **Machine Learning**
- **Algorithm**: Logistic Regression
- **Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Language**: Python

### **Deployment**
- **Cloud Platform**: Microsoft Azure ML
- **Model Hosting**: Azure ML Endpoints

### **Frontend**
- **Framework**: Streamlit
                    
### **Dataset used to train the model**
- **Source**: Kaggle Telco Customer Churn Dataset
- **Features**: 19 customer attributes
- **Target**: Binary classification (Churn: Yes/No)
""")

st.title("Telco Customer Churn Predictor")

# Add inputs
seniorcitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0)
totalcharges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0)

partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])

gender = st.selectbox("Gender", ["Male", "Female"])
multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paymentmethod = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

if st.button("Predict"):
    payload = [{
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "paperlessbilling": paperlessbilling,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
        "gender": gender,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paymentmethod": paymentmethod
    }]

    
    try:
        response = requests.post(SCORING_URI, data=json.dumps(payload), headers=headers)
        print(response.json())

        if response.status_code == 200:
            result = response.json()
            # Handle double-encoded JSON
            json_string = response.json()  # First parse, the object is still a string
            result = json.loads(json_string)  # Second parse

            # Extract predictions and probabilities
            predictions = result.get("predictions", [])
            probabilities = result.get("probabilities", [])
            
            if predictions and probabilities:
                prediction = predictions[0]  # The predicted class (0 or 1)
                probability = probabilities[0]  # The probability for this prediction
                
                # Display result based on prediction
                if prediction == 1:
                    st.error(f"🚨 **The customer is likely to LEAVE** with a probability of {probability:.1%}")
                else:
                    st.success(f"✅ **The customer is likely to STAY** with a probability of {probability:.1%}")
                
                # Show additional details
                st.subheader("Prediction Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", "Will Leave" if prediction == 1 else "Will Stay")
                with col2:
                    st.metric("Confidence", f"{probability:.1%}")                    
            else:
                st.error("Invalid response format from the API")
                st.write(result)
        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)            
    except Exception as e:
        st.error(f"Error: {str(e)}") 
