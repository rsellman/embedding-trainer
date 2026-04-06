#!/usr/bin/env python3
"""
OmniAlly Embedding Trainer — Standalone GPU Pod Script
Run on a RunPod GPU pod. Downloads data from HF, trains, pushes model, exits.
"""
import os, json, time, sys

def main():
    print("=== OmniAlly Embedding Training ===")
    
    # Config from env vars
    hf_token = os.environ.get("HF_TOKEN", "")
    dataset_repo = os.environ.get("DATASET_REPO", "rsellman/omnially-embedding-data")
    model_repo = os.environ.get("MODEL_REPO", "rsellman/omnially-embeddings-v1")
    base_model = os.environ.get("BASE_MODEL", "BAAI/bge-small-en-v1.5")
    epochs = int(os.environ.get("EPOCHS", "3"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-5"))
    
    print(f"Base: {base_model} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    
    # Install deps
    print("\n--- Installing dependencies ---")
    os.system("pip install -q sentence-transformers==3.0.1 transformers==4.44.2 huggingface_hub")
    
    from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
    from torch.utils.data import DataLoader
    from huggingface_hub import hf_hub_download, HfApi
    
    # Download training data
    print("\n--- Downloading training data ---")
    train_path = hf_hub_download(repo_id=dataset_repo, filename="embedding_train.jsonl", repo_type="dataset", token=hf_token)
    val_path = hf_hub_download(repo_id=dataset_repo, filename="embedding_val.jsonl", repo_type="dataset", token=hf_token)
    
    # Load data
    train_examples = []
    with open(train_path) as f:
        for line in f:
            d = json.loads(line)
            train_examples.append(InputExample(texts=[d["anchor"][:512], d["positive"][:512], d["negative"][:512]]))
    
    val_a, val_p, val_n = [], [], []
    with open(val_path) as f:
        for line in f:
            d = json.loads(line)
            val_a.append(d["anchor"][:512])
            val_p.append(d["positive"][:512])
            val_n.append(d["negative"][:512])
    
    print(f"Training: {len(train_examples)} triplets | Validation: {len(val_a)} triplets")
    
    # Train
    print("\n--- Training ---")
    model = SentenceTransformer(base_model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.TripletLoss(model=model)
    
    evaluator = None
    pre_score = None
    if val_a:
        evaluator = evaluation.TripletEvaluator(anchors=val_a, positives=val_p, negatives=val_n, name="val")
        pre = evaluator(model)
        pre_score = list(pre.values())[0] if isinstance(pre, dict) else float(pre)
        print(f"Baseline accuracy: {pre_score:.4f}")
    
    output_dir = "/tmp/omnially-embeddings-v1"
    os.makedirs(output_dir, exist_ok=True)
    
    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator, epochs=epochs, warmup_steps=100,
        evaluation_steps=500, output_path=output_dir,
        show_progress_bar=True, optimizer_params={"lr": lr},
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    
    post_score = None
    if evaluator:
        post = evaluator(model)
        post_score = list(post.values())[0] if isinstance(post, dict) else float(post)
        print(f"Post-training accuracy: {post_score:.4f}")
    
    model.save(output_dir)
    
    # Save metadata
    meta = {
        "base_model": base_model, "train_examples": len(train_examples),
        "val_examples": len(val_a), "epochs": epochs, "batch_size": batch_size,
        "learning_rate": lr, "elapsed_minutes": round(elapsed/60, 1),
        "baseline_accuracy": round(pre_score, 4) if pre_score is not None else None,
        "final_accuracy": round(post_score, 4) if post_score is not None else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {json.dumps(meta, indent=2)}")
    
    # Push to HuggingFace
    print(f"\n--- Pushing model to {model_repo} ---")
    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=output_dir, repo_id=model_repo, repo_type="model",
        commit_message=f"Training: {meta['train_examples']} examples, {meta['epochs']} epochs, accuracy {meta.get('final_accuracy', 'N/A')}",
    )
    print("Model pushed to HuggingFace!")
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
