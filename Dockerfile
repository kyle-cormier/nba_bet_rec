# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for pandas/numpy/scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port 8080 (Cloud Run will use this)
EXPOSE 8080

# Start Streamlit on Cloud Run's expected port/address
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
