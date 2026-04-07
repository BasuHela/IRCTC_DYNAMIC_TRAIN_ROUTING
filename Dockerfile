FROM python:3.11-slim

# Create a new user named "user" with user ID 1000 (Required for HF Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

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

# Copy application code and change ownership to the non-root user
COPY --chown=user:user . .

# Switch to the non-root user
USER user

# Expose port for HF Spaces
EXPOSE 7860

# Environment variables
ENV HF_TOKEN=""
ENV API_BASE_URL="https://api-inference.huggingface.co/v1/"
ENV MODEL_NAME="meta-llama/Meta-Llama-3-70B-Instruct"

# Health check (Using 127.0.0.1 to prevent IPv6 Docker routing bugs)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://127.0.0.1:7860/health')" || exit 1

# Start the FastAPI server via the OpenEnv app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
