FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    openenv-core \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.30.0" \
    pydantic>=2.9.0 \
    openai>=1.50.0 \
    requests>=2.32.0 \
    pyyaml>=6.0 \
    websockets>=12.0

# Copy application code
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Environment variables (overridden at runtime)
ENV HF_TOKEN=""
ENV API_BASE_URL="https://api-inference.huggingface.co/v1/"
ENV MODEL_NAME="meta-llama/Meta-Llama-3-70B-Instruct"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Start the FastAPI server via the OpenEnv app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
