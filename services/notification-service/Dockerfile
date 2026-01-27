# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for psutil)
RUN apt-get update && \
    apt-get install -y gcc procps && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency list first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the service port
EXPOSE 8003

# Run the service
CMD ["python", "notification_service.py"]
