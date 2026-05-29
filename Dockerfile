# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional, for psycopg2 / asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY agent.py .
COPY qdrant.py .
COPY orcr.py .
COPY placement.py .
COPY user_crud_asyncpg.py .
# If you have any other .py files (e.g., config.py, models.py), add them here

# Expose the port
EXPOSE 8000

# Run with gunicorn + uvicorn worker (recommended for production)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "agent:app"]