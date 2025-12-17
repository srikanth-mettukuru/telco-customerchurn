# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the API file (not entire src directory)
COPY src/api.py ./src/api.py

# Copy the trained model from model directory
COPY model/model.joblib ./model/model.joblib

# Expose port
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]