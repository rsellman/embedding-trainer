FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    sentence-transformers>=3.0.0 \
    huggingface_hub>=0.20.0 \
    runpod>=1.6.0 \
    numpy

# Pre-download base model so cold starts are faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
