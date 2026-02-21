# Frontend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (frontend.txt now includes base.txt)
COPY requirements/ requirements/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/frontend.txt

# Copy source modules for local inference
COPY src/ ./src/

# Copy Streamlit app
COPY apps/streamlit_app/ ./apps/streamlit_app/

# Set PYTHONPATH to include src
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8501/_stcore/health').raise_for_status()"

# Run Streamlit
CMD ["streamlit", "run", "apps/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
