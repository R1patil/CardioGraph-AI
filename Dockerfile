FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and frontend assets
COPY . .

# Expose port (default 8000)
EXPOSE 8000

# Run FastAPI app with Uvicorn (Render assigns $PORT dynamically)
CMD ["sh", "-c", "uvicorn agent:app --host 0.0.0.0 --port ${PORT:-8000}"]
