# Aspect-Based Sentiment Analysis (ABSA) - Amazon Electronics Reviews

## Overview
An end-to-end NLP pipeline that identifies **which product feature** a review sentence discusses and **how the reviewer feels** about it. Unlike standard sentiment analysis that gives one label per review, this system operates at the sentence level — capturing multiple opinions from a single review.

## Package Structure

```
src/absa/
├── data/                 # Data loading, cleaning, review-level splitting
│   └── loader.py         # Raw data loading, preprocessing, critical review-level split
├── labeling/             # Aspect & sentiment labeling
│   └── keyword_labeler.py
├── models/               # PyTorch neural network models & training
│   ├── neural_networks.py  # AspectClassifier, SentimentClassifier
│   ├── embeddings.py       # DistilBERT embedding extraction
│   └── trainer.py          # Training loops with early stopping & LR scheduling
├── eval/                 # Evaluation metrics
│   └── metrics.py        # Confusion matrices, macro-F1, per-class reports
└── __init__.py

configs/
└── config.yaml           # Paths, hyperparameters, aspect keywords, sentiment thresholds

tests/
└── test_data.py          # Pytest tests for data cleaning & review-level split

notebooks/
└── Aspect_Based_Sentiment_Analysis.ipynb  # Original notebook (preserved for reference)
```

## Critical Features

### ✅ Review-Level Splitting (Prevents Data Leakage)
The train/test split happens **at the review level BEFORE sentence tokenization**:
- Sentences from the same review are kept together
- No review can appear in both train and test sets
- Prevents implicit memorization of review context

### ✅ PyTorch Models
- Fully PyTorch implementation (no TensorFlow dependency)
- AspectClassifier: 8-class softmax over aspects
- SentimentClassifier: Binary classification (pos/neg)
- Both models use DistilBERT frozen embeddings + 2-layer DNN

### ✅ Comprehensive Metrics
- Per-class accuracy and confusion matrices
- Macro-F1 scoring (unbiased to class imbalance)
- Stratified evaluation by aspect and sentiment

## Installation

```bash
# Clone and install in development mode
pip install -e .

# With test dependencies
pip install -e ".[dev]"

```

## Quick Start

```python
import yaml
import pandas as pd
from absa.data import preprocess_reviews, split_by_review, apply_aspect_labels, apply_sentiment_labels
from absa.models import DistilBertEmbedder, AspectClassifier, ModelTrainer
from absa.labeling import label_aspect, label_sentiment

# Load config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Preprocess reviews (preserves review IDs for later splitting)
df = pd.read_csv('data/raw/reviews.csv')
df = preprocess_reviews(df, min_sentence_length=config['data']['min_sentence_length'])

# Apply labels
df = apply_aspect_labels(df, config['labeling'], sentence_col='sentences')
df = apply_sentiment_labels(df, config['sentiment'], rating_col='rating')

# CRITICAL: Split at review level to prevent data leakage
train_df, test_df = split_by_review(df, test_size=0.2, random_state=42, stratify_by='rating')

# Extract embeddings
embedder = DistilBertEmbedder()
train_embeddings = embedder.extract_embeddings(train_df['sentences'])

# Train model
model = AspectClassifier(num_aspects=8)
trainer = ModelTrainer(model, learning_rate=0.001)
history = trainer.train_aspect_classifier(X_train, y_train, X_val, y_val, epochs=50)
```

## Results
| Model | Architecture | Test Accuracy | Macro-F1 |
|-------|-------------|---------------|----------|
| Aspect Classifier | DistilBERT + DNN (768→512→256→8) | 80.72% | 0.78 |
| Sentiment Classifier | DistilBERT + DNN (768→512→256→1) | 63.64% | 0.74 |
| LR Baseline (Aspect) | LogisticRegression on DistilBERT embeddings | — | — |
| LR Baseline (Sentiment) | LogisticRegression on DistilBERT embeddings | — | — |

Baselines run automatically in the pipeline. The DNN models should outperform LogisticRegression on the same embeddings, confirming the hidden layers add value.

## Error Analysis

### Aspect Classifier — Strong Per-Class Performance
| Aspect | Accuracy | Samples | Notes |
|--------|----------|---------|-------|
| sound | 86.7% | 2,604 | Best performer — distinct keywords |
| price | 84.9% | 1,567 | Strong signal from cost/value terms |
| display | 83.4% | 988 | Some confusion with camera |
| camera | 82.3% | 2,225 | |
| connectivity | 78.9% | 695 | Limited training samples |
| battery | 78.5% | 2,794 | Occurs in many contexts, some noise |
| design | 73.8% | 1,349 | Subjective, overlaps with build quality |
| performance | 67.4% | 846 | Most confused — terms like "fast/slow" used across categories |

### Sentiment Classifier — Imbalance Warning
```
              precision    recall  f1-score   support
    negative       0.24      0.81      0.37      1,565
    positive       0.95      0.61      0.74     10,239
```

The sentiment model has **high positive precision (0.95)** but **low negative precision (0.24)**. The model strongly biases toward predicting "positive" because:
1. Sentiment is derived from the **overall star rating**, not sentence-level opinion
2. The dataset is naturally imbalanced (~82% positive sentences)
3. A 5-star review can contain negative sentences about specific aspects

This is a known limitation of the rating-based labeling heuristic. Future work should use sentence-level sentiment annotations or class-weighted loss functions.

## Testing

```bash
# Run pytest tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/absa
```

## Tech Stack
- **Deep Learning:** PyTorch
- **Embeddings:** DistilBERT (HuggingFace Transformers)
- **Baseline:** LogisticRegression on DistilBERT embeddings
- **NLP:** NLTK, scikit-learn
- **Data:** pandas, NumPy, matplotlib, seaborn
- **Config:** PyYAML

## Dataset
[McAuley Lab Amazon Reviews 2023 — Electronics](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

## Notebook
`notebooks/Aspect_Based_Sentiment_Analysis.ipynb` contains the full pipeline run with 50K reviews, including:
- End-to-end execution with timing and progress
- Confidence distribution analysis with visualizations
- Edge case testing on unusual inputs
- Per-aspect accuracy breakdown and confusion matrix
- Interactive single-review analysis function

## Known Issues & Future Work
- Keyword-based aspect labeling is noisy; consider sequence labeling (NER) models
- Sentiment is inferred from overall rating, not sentence-level annotations
- Neutral sentiment class (3-star ratings) is underrepresented
