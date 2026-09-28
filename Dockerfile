# syntax=docker/dockerfile:1
#
# Layer 5 — Containerisation of the CardioSignal API.
#
# Build:   docker build -t cardiosignal-api .
# Run:     docker run -p 8000:8000 cardiosignal-api
#
# Design notes
# ------------
# - python:3.11-slim keeps the image small; TensorFlow CPU is the heavy part.
# - Dependencies are installed BEFORE copying the source, so editing code does
#   not invalidate the pip layer and rebuilds stay fast.
# - The container runs as a non-root user: a service that only reads a model
#   and writes one SQLite file has no reason to run as root.
# - Only the runtime pieces are copied (src/, app/, data/). The dataset, the
#   notebooks and the figures stay out (see .dockerignore).
# - The model checkpoint is mounted at run time rather than baked in, so the
#   image does not have to be rebuilt every time the model is retrained.

FROM python:3.11-slim

# --- System deps ------------------------------------------------------------
# libgomp1 is required by TensorFlow's CPU kernels (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps (cached layer) --------------------------------------------
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Application code -------------------------------------------------------
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/

# Directory where the mounted checkpoint is expected, and where the SQLite
# history file will be written.
RUN mkdir -p /app/checkpoints

# --- Non-root user ----------------------------------------------------------
RUN useradd --create-home --shell /bin/bash cardio \
    && chown -R cardio:cardio /app
USER cardio

# --- Runtime ----------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    CARDIO_CKPT=/app/checkpoints/ecg_1dcnn_best.h5 \
    CARDIO_DB=/app/data/cardio_history.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
