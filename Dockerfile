# Dockerfile
FROM python:3.10-slim-buster

WORKDIR /app

# Install system dependencies for psycopg2 (needed by both services)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=120

# Copy all application code into the image
# This ensures both pipeline and dashboard code is available
COPY . .

ENV PYTHONPATH=/app

# No CMD here! The specific command will be defined in docker-compose.yml
EXPOSE 8050