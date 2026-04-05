FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /app

# Install Python deps
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    sentence-transformers>=3.0.0 \
    huggingface_hub>=0.20.0 \
    runpod>=1.6.0 \
    numpy

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
