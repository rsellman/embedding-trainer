FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers==4.44.2 \
    sentence-transformers==3.0.1 \
    datasets>=2.0.0 \
    accelerate>=0.21.0 \
    huggingface_hub>=0.20.0 \
    runpod>=1.6.0

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
