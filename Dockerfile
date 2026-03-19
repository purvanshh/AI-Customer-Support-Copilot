FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-optional.txt || true

# App source
COPY . .

# Create data + log dirs
RUN mkdir -p data/processed logs

# Default env (can be overridden at runtime)
ENV EMBEDDING_BACKEND=mock \
    LLM_BACKEND=mock \
    DATA_DIR=data \
    LOG_DIR=logs \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host ${API_HOST} --port ${API_PORT}"]
