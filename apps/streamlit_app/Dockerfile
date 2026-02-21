# Frontend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements/frontend.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit app
COPY apps/streamlit_app/ ./apps/streamlit_app/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8501/_stcore/health').raise_for_status()"

# Run Streamlit
CMD ["streamlit", "run", "apps/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
