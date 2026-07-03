#!/usr/bin/env python
"""Validate that the package structure is correct and modules import."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

print("=" * 70)
print("ABSA PACKAGE VALIDATION")
print("=" * 70)

# Test imports
print("\n[OK] Testing module imports...")
try:
    from absa import data, labeling, models, eval
    print("  [OK] Main package imports OK")
except ImportError as e:
    print(f"  [FAIL] Failed to import main package: {e}")
    sys.exit(1)

try:
    from absa.data import (
        clean_text, preprocess_reviews, split_by_review,
        remove_general_aspect, balance_by_aspect_sentiment
    )
    print("  [OK] Data module imports OK")
except ImportError as e:
    print(f"  [FAIL] Failed to import data module: {e}")
    sys.exit(1)

try:
    from absa.labeling import label_aspect, label_sentiment
    print("  [OK] Labeling module imports OK")
except ImportError as e:
    print(f"  [FAIL] Failed to import labeling module: {e}")
    sys.exit(1)

try:
    from absa.models import (
        AspectClassifier, SentimentClassifier,
        DistilBertEmbedder, ModelTrainer
    )
    print("  [OK] Models module imports OK")
except ImportError as e:
    print(f"  [FAIL] Failed to import models module: {e}")
    sys.exit(1)

try:
    from absa.eval import compute_metrics, confusion_matrix_report
    print("  [OK] Eval module imports OK")
except ImportError as e:
    print(f"  [FAIL] Failed to import eval module: {e}")
    sys.exit(1)

# Test basic functionality
print("\n[OK] Testing basic functionality...")

# Test clean_text
test_text = "<p>Hello! Visit http://example.com Price: $99</p>"
cleaned = clean_text(test_text)
assert "hello" in cleaned
assert "example.com" not in cleaned
assert "99" in cleaned
print("  [OK] clean_text() works")

# Test label_aspect
aspect_keywords = {
    'battery': ['battery', 'charge', 'power'],
    'camera': ['camera', 'photo', 'lens'],
}
aspect = label_aspect("The battery is great!", aspect_keywords)
assert aspect == 'battery'
aspect_general = label_aspect("This is a nice product", aspect_keywords)
assert aspect_general == 'general'
print("  [OK] label_aspect() works")

# Test label_sentiment
sentiment_thresholds = {
    'positive_threshold': 4,
    'neutral_rating': 3,
}
assert label_sentiment(5.0, sentiment_thresholds) == 'positive'
assert label_sentiment(3.0, sentiment_thresholds) == 'neutral'
assert label_sentiment(2.0, sentiment_thresholds) == 'negative'
print("  [OK] label_sentiment() works")

# Test split_by_review (critical data leakage prevention)
import pandas as pd
import numpy as np

print("\n[OK] Testing review-level split (critical for preventing data leakage)...")
test_data = []
for review_id in range(10):
    for sent_id in range(3):
        test_data.append({
            'review_id': review_id,
            'sentences': f'Review {review_id}, sentence {sent_id}',
            'rating': np.random.choice([1, 3, 5]),
        })

df_test = pd.DataFrame(test_data)
train_df, test_df = split_by_review(df_test, test_size=0.3, random_state=42)

train_reviews = set(train_df['review_id'].unique())
test_reviews = set(test_df['review_id'].unique())

# Check no overlap (critical!)
overlap = train_reviews & test_reviews
assert len(overlap) == 0, f"Data leakage detected! Reviews in both sets: {overlap}"
print(f"  [OK] split_by_review() prevents data leakage")
print(f"    Train reviews: {len(train_reviews)}, Test reviews: {len(test_reviews)}")
print(f"    All sentences from a review stay together [OK]")

# Test remove_general_aspect
print("\n[OK] Testing aspect filtering...")
df_aspects = pd.DataFrame({
    'aspect': ['battery', 'general', 'camera', 'general', 'sound'],
    'text': ['a', 'b', 'c', 'd', 'e'],
})
df_filtered = remove_general_aspect(df_aspects)
assert len(df_filtered) == 3
assert 'general' not in df_filtered['aspect'].values
print("  [OK] remove_general_aspect() works")

# Test balance_by_aspect_sentiment
print("\n[OK] Testing dataset balancing...")
df_imbalanced = pd.DataFrame({
    'aspect': ['battery'] * 100 + ['camera'] * 20,
    'sentiment': ['positive'] * 100 + ['positive'] * 20,
    'text': [f'text{i}' for i in range(120)],
})
df_balanced = balance_by_aspect_sentiment(df_imbalanced, max_samples_per_group=50)

for (aspect, sentiment), group in df_balanced.groupby(['aspect', 'sentiment']):
    assert len(group) <= 50, f"Balance failed: {aspect}/{sentiment} has {len(group)} samples"

print("  [OK] balance_by_aspect_sentiment() works")

# Test model initialization
print("\n[OK] Testing PyTorch model initialization...")
import torch

model = AspectClassifier(embedding_dim=768, num_aspects=8)
assert isinstance(model, torch.nn.Module)
x = torch.randn(4, 768)
logits = model(x)
assert logits.shape == (4, 8), f"Expected shape (4, 8), got {logits.shape}"
print("  [OK] AspectClassifier model initialization works")

model_sent = SentimentClassifier(embedding_dim=768)
assert isinstance(model_sent, torch.nn.Module)
logits_sent = model_sent(x)
assert logits_sent.shape == (4, 1), f"Expected shape (4, 1), got {logits_sent.shape}"
print("  [OK] SentimentClassifier model initialization works")

# Test metrics
print("\n[OK] Testing evaluation metrics...")
y_true = np.array([0, 1, 0, 1, 2, 1, 0, 2])
y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 2])
metrics = compute_metrics(
    y_true, y_pred,
    class_names=['battery', 'camera', 'display'],
    task='multiclass'
)
assert 'accuracy' in metrics
assert 'macro_f1' in metrics
assert 'confusion_matrix' in metrics
assert 'classification_report' in metrics
print(f"  [OK] compute_metrics() works (accuracy={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f})")

print("\n" + "=" * 70)
print("[OK] ALL VALIDATION CHECKS PASSED")
print("=" * 70)
print("\nPackage structure is correct and all modules are functional.")
print("\nNext steps:")
print("1. Review configs/config.yaml and adjust hyperparameters as needed")
print("2. Prepare your review data or download Amazon Reviews 2023 dataset")
print("3. Run: python -c 'from absa.pipeline import run_full_pipeline; run_full_pipeline()'")
print("   Or use the individual modules for custom workflows")
print("\nFor testing with pytest (optional):")
print("  pip install pytest")
print("  pytest tests/ -v")
