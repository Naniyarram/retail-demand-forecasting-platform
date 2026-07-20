FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt


# Copy source code only (data is volume-mounted at runtime)
COPY pipeline ./pipeline
COPY run_experiments.py .

# Create artifact directories
RUN mkdir -p artifacts/models artifacts/reports artifacts/leaderboards

EXPOSE 8000

CMD ["uvicorn", "pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
