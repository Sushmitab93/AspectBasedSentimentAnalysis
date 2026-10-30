"""Run the ABSA pipeline and save test-set predictions for the RAG layer.

Run from the repo root:  python rag/export_predictions.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from absa.pipeline import run_full_pipeline, load_config

CONFIG_PATH = "configs/config.yaml"


def export_predictions(data: dict, config: dict) -> pd.DataFrame:
    # Line up sentence row 0 with embedding row 0
    test_df = data["test_df"].reset_index(drop=True)
    X = data["test_embeddings"]
    assert len(test_df) == len(X), "Sentences and embeddings are out of sync"

    # Aspect: raw scores -> probabilities (softmax) -> pick the highest
    aspect_logits = data["aspect_trainer"].predict(X)                  # (n, 8)
    aspect_probs = torch.softmax(torch.from_numpy(aspect_logits), dim=1).numpy()
    aspect_ids = aspect_probs.argmax(axis=1)
    pred_aspect = data["aspect_encoder"].inverse_transform(aspect_ids)
    aspect_conf = aspect_probs.max(axis=1)

    # Sentiment: one raw score -> probability of "positive" (sigmoid)
    sent_logits = data["sentiment_trainer"].predict(X).flatten()
    pos_prob = torch.sigmoid(torch.from_numpy(sent_logits)).numpy()
    sent_ids = (pos_prob > 0.5).astype(int)
    pred_sentiment = data["sentiment_encoder"].inverse_transform(sent_ids)
    sentiment_conf = np.where(sent_ids == 1, pos_prob, 1 - pos_prob)

    out_df = pd.DataFrame({
        "sentence_id": np.arange(len(test_df)),
        "review_id": test_df["review_id"],
        "sentence": test_df["sentences"],
        "rating": test_df["rating"],
        "pred_aspect": pred_aspect,
        "pred_sentiment": pred_sentiment,
        "aspect_conf": aspect_conf.round(4),
        "sentiment_conf": sentiment_conf.round(4),
        "true_aspect": test_df["aspect"],
        "true_sentiment": test_df["sentiment"],
    })

    out_path = Path(config["evaluation"]["predictions_path"])   # results/predictions.csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} predictions to {out_path}")
    return out_df


if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    data = run_full_pipeline(CONFIG_PATH)
    export_predictions(data, config)