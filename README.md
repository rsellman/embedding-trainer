# OmniAlly Embedding Trainer

RunPod serverless handler for training OmniAlly's custom embedding model.

## How it works

1. Knowledge Forge triggers this via RunPod API
2. Pulls training triplets from HuggingFace (`rsellman/omnially-embedding-data`)
3. Fine-tunes BAAI/bge-small-en-v1.5 with TripletLoss (~90 min on T4)
4. Pushes trained model to HuggingFace (`rsellman/omnially-embeddings-v1`)

## Build & Push

```bash
docker build -t ghcr.io/rsellman/embedding-trainer:latest .
docker push ghcr.io/rsellman/embedding-trainer:latest
```

## RunPod Input

```json
{
  "input": {
    "hf_token": "hf_...",
    "dataset_repo": "rsellman/omnially-embedding-data",
    "model_repo": "rsellman/omnially-embeddings-v1",
    "epochs": 3,
    "batch_size": 16
  }
}
```
