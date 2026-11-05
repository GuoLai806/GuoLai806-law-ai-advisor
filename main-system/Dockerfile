# Build from repository root: docker build -f main-system/Dockerfile .
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 EVIDENCE_RAG_DB=/data/evidence.sqlite3
COPY pyproject.toml ./
COPY evidence_rag ./evidence_rag
COPY legal_rag ./legal_rag
RUN pip install . && useradd --create-home --uid 10001 appuser && mkdir /data && chown appuser:appuser /data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]
CMD ["uvicorn", "evidence_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
