FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for serving
RUN pip install --no-cache-dir bentoml prometheus_client

# Copy model and service code
COPY models/baseline_xgb.pkl /app/models/
COPY src/serving/service.py /app/

# Set environment variables
ENV MODEL_PATH=/app/models/baseline_xgb.pkl
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 3000
EXPOSE 8000

# Run the service
CMD ["python", "/app/service.py"]
