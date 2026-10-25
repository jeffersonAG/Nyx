FROM python:3.12-slim
WORKDIR /app
# Piper necesita datos de fonemas/idioma, instalados una sola vez en la imagen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng-data libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
RUN mkdir -p /data /voices
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
