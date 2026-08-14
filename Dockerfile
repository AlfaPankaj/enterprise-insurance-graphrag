# GraphRAG Insurance Claims System — API image (Phase 5)
# Lean by design: only the runtime deps the FastAPI server needs.
FROM python:3.12-slim

WORKDIR /app

# Python deps first (layer-cached; rebuilds skip re-installing when unchanged)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# App code + the assets the runtime touches
COPY src ./src
COPY scripts ./scripts
COPY prompts ./prompts
COPY data ./data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "graphrag.api_server:app", "--app-dir", "src", \
     "--host", "0.0.0.0", "--port", "8000"]
