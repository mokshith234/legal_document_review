# Dockerfile
# ----------
# Legal Document Review OpenEnv — Hackathon Submission
#
# Build:  docker build -t legal-env .
# Run:    docker run -p 7860:7860 \
#           -e API_BASE_URL=https://api.openai.com/v1 \
#           -e MODEL_NAME=gpt-4o-mini \
#           -e HF_TOKEN=sk-... \
#           legal-env

# -----------------------------------------------------------------------
# Stage 1 — builder: install deps into a clean venv
# -----------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /build/venv
ENV PATH="/build/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------
# Stage 2 — runtime: lean image with app source only
# -----------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy venv from builder stage
COPY --from=builder /build/venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy ALL application source (excludes .venv via .dockerignore)
COPY . .

# Runtime environment variables (overridden by HF Space secrets at deploy time)
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
# inference.py points to localhost — server runs in same container
ENV ENV_BASE_URL="http://localhost:7860"

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Health check — used by HF Spaces ping validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start FastAPI server on port 7860 (HF Spaces default)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
