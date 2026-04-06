FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    sentence-transformers==3.3.1 \
    huggingface_hub>=0.20.0 \
    runpod>=1.6.0

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
