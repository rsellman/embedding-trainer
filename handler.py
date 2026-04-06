"""
OmniAlly Embedding Trainer — RunPod Serverless Handler
=======================================================
Triggered monthly by the Knowledge Forge.
1. Pulls training triplets from HuggingFace dataset repo
2. Trains BGE-small with TripletLoss
3. Pushes trained model to HuggingFace model repo
4. Returns training metrics

Input: {"hf_token": "...", "dataset_repo": "...", "model_repo": "...", "epochs": 3, "batch_size": 16}
Output: {"status": "complete", "train_examples": N, "accuracy": 0.95, "elapsed_minutes": 12.5}
"""

import os, json, time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-trainer")


def handler(event):
    """RunPod serverless handler — trains and pushes embedding model."""
    from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
    from torch.utils.data import DataLoader
    from huggingface_hub import hf_hub_download, HfApi

    input_data = event.get("input", {})
    hf_token = input_data.get("hf_token", os.environ.get("HF_TOKEN", ""))
    dataset_repo = input_data.get("dataset_repo", "rsellman/omnially-embedding-data")
    model_repo = input_data.get("model_repo", "rsellman/omnially-embeddings-v1")
    base_model = input_data.get("base_model", "BAAI/bge-small-en-v1.5")
    epochs = input_data.get("epochs", 3)
    batch_size = input_data.get("batch_size", 16)
    lr = input_data.get("learning_rate", 2e-5)

    output_dir = "/tmp/omnially-embeddings-v1"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== OmniAlly Embedding Training ===")
    logger.info(f"Base: {base_model} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    # --- 1. Download training data from HuggingFace ---
    logger.info(f"Downloading training data from {dataset_repo}")
    train_path = hf_hub_download(
        repo_id=dataset_repo, filename="embedding_train.jsonl",
        repo_type="dataset", token=hf_token,
    )
    val_path = hf_hub_download(
        repo_id=dataset_repo, filename="embedding_val.jsonl",
        repo_type="dataset", token=hf_token,
    )

    # --- 2. Load training data ---
    train_examples = []
    with open(train_path) as f:
        for line in f:
            d = json.loads(line)
            train_examples.append(InputExample(
                texts=[d["anchor"][:512], d["positive"][:512], d["negative"][:512]]
            ))

    val_a, val_p, val_n = [], [], []
    with open(val_path) as f:
        for line in f:
            d = json.loads(line)
            val_a.append(d["anchor"][:512])
            val_p.append(d["positive"][:512])
            val_n.append(d["negative"][:512])

    logger.info(f"Training examples: {len(train_examples)} | Validation: {len(val_a)}")

    if not train_examples:
        return {"status": "error", "message": "No training data found"}

    # --- 3. Train ---
    model = SentenceTransformer(base_model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.TripletLoss(model=model)

    evaluator = None
    pre_score = None
    if val_a:
        evaluator = evaluation.TripletEvaluator(
            anchors=val_a, positives=val_p, negatives=val_n, name="val"
        )
        pre = evaluator(model)
        pre_score = list(pre.values())[0] if isinstance(pre, dict) else float(pre)
        logger.info(f"Baseline accuracy: {pre_score:.4f}")

    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=100,
        evaluation_steps=500,
        output_path=output_dir,
        show_progress_bar=True,
        optimizer_params={"lr": lr},
    )
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed/60:.1f} minutes")

    # Post-training eval
    post_score = None
    if evaluator:
        post = evaluator(model)
        post_score = list(post.values())[0] if isinstance(post, dict) else float(post)
        logger.info(f"Post-training accuracy: {post_score:.4f}")

    model.save(output_dir)

    # --- 4. Write training metadata ---
    meta = {
        "base_model": base_model,
        "train_examples": len(train_examples),
        "val_examples": len(val_a),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "elapsed_minutes": round(elapsed / 60, 1),
        "baseline_accuracy": round(pre_score, 4) if pre_score is not None else None,
        "final_accuracy": round(post_score, 4) if post_score is not None else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- 5. Push trained model to HuggingFace ---
    logger.info(f"Pushing trained model to {model_repo}")
    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=model_repo,
        repo_type="model",
        commit_message=f"Training run: {meta['train_examples']} examples, {meta['epochs']} epochs, accuracy {meta.get('final_accuracy', 'N/A')}",
    )
    logger.info("Model pushed to HuggingFace successfully")

    return {
        "status": "complete",
        **meta,
    }


# RunPod serverless entry point
import runpod
runpod.serverless.start({"handler": handler})
