FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/code \
    MODEL_DIR=/app/models/current \
    MODEL_CACHE_DIR=/tmp/route-agent-model-cache

WORKDIR /app

COPY code/deployment/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY code/models/inference.py /app/code/models/inference.py
COPY code/deployment/api/main.py code/deployment/api/model_service.py code/deployment/api/schemas.py /app/code/deployment/api/

EXPOSE 8000

CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
