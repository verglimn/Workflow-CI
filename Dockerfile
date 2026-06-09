FROM python:3.12.7-slim

LABEL maintainer="Rieco Edward"
LABEL description="MLflow Project: Heart Disease Classifier"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy project files
COPY MLProject/requirements.txt ./requirements.txt
COPY MLProject/ ./MLProject/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# MLflow Project entrypoint
ENV MLFLOW_TRACKING_URI=mlruns

# Default: run training with default params
CMD ["python", "MLProject/modelling.py"]
