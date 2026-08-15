FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    Flask==3.1.2 \
    gunicorn==23.0.0 \
    transformers==4.55.4 \
    sentencepiece==0.2.0 \
    protobuf==6.32.0 \
    safetensors==0.6.2 \
    accelerate==1.14.0 \
    huggingface-hub==0.36.2 \
    requests==2.32.5 \
    python-dotenv==1.1.1

RUN pip install --no-cache-dir \
    torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY app.py .
COPY grammar_model.py .
COPY oxford_api.py .

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "300", "app:app"]
