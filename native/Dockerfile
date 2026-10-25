FROM python:3.12-slim
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
COPY requirements.txt .
# Instalar build CPU: no descarga CUDA ni GPU ni modelos preentrenados.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3" \
    && pip install --no-cache-dir -r requirements.txt
COPY nyx_native ./nyx_native
COPY sample_corpus ./samples
RUN mkdir -p /models /corpus
EXPOSE 8010
CMD ["uvicorn", "nyx_native.serve:app", "--host", "0.0.0.0", "--port", "8010"]
