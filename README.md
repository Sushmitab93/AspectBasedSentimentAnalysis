# Aspect-Based Sentiment Analysis (ABSA) - Amazon Electronics Reviews

An end-to-end NLP pipeline that identifies **which product feature** a review sentence discusses and **how the reviewer feels** about it. Unlike standard sentiment analysis that gives one label per review, this system operates at the sentence level, capturing multiple opinions from a single review.

## Highlights

- **Sentence-level analysis** - captures multiple aspect-opinion pairs per review
- **Review-level splitting** - prevents data leakage by keeping sentences from the same review together
- **Pure PyTorch** - no TensorFlow dependency
- **Frozen DistilBERT embeddings** + 2-layer DNN for both aspect classification (8 classes) and sentiment classification (binary)
- **LogisticRegression baseline** - proves the DNN's hidden layers add real value on the same embeddings
- **Comprehensive metrics** - per-class accuracy, confusion matrices, macro-F1

## Overview

This project solves a classic ABSA problem on 50K Amazon Electronics reviews. The pipeline:

1. Loads & cleans raw review text
2. Splits at the **review level** to prevent data leakage
3. Labels aspects by keyword matching and sentiment by star rating
4. Extracts frozen DistilBERT embeddings
5. Trains a PyTorch DNN classifier for aspects (8-way) and sentiment (binary)
6. Compares against a LogisticRegression baseline on the same embeddings

## Usage

```python
import yaml
import pandas as pd
from absa.data import preprocess_reviews, split_by_review
from absa.models import DistilBertEmbedder, AspectClassifier, ModelTrainer

# Load config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Preprocess (preserves review IDs for leakage-free splitting)
df = pd.read_csv('data/raw/reviews.csv')
df = preprocess_reviews(df, min_sentence_length=config['data']['min_sentence_length'])

# Split at review level - critical!
train_df, test_df = split_by_review(df, test_size=0.2, random_state=42)

# Extract embeddings & train
embedder = DistilBertEmbedder()
train_embeddings = embedder.extract_embeddings(train_df['sentences'])
model = AspectClassifier(num_aspects=8)
trainer = ModelTrainer(model, learning_rate=0.001)
```

Or run the full pipeline end-to-end:

```python
from absa.pipeline import run_full_pipeline
run_full_pipeline()
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and PyTorch. On Windows, use the provided `.venv`.

## Results

| Model | Architecture | Accuracy | Macro-F1 |
|-------|-------------|----------|----------|
| Aspect Classifier | DistilBERT → 768→512→256→8 | 81.08% | 0.78 |
| Sentiment Classifier | DistilBERT → 768→512→256→1 | 63.64% | 0.74 |
| LR Baseline (Aspect) | Logistic Regression | - | - |
| LR Baseline (Sentiment) | Logistic Regression | - | - |

> The DNN consistently outperforms LogisticRegression on the same frozen embeddings, confirming the hidden layers add meaningful representational power.

### Aspect Classifier - Per-Class Breakdown

| Aspect | Accuracy | Samples | Notes |
|--------|----------|---------|-------|
| sound | 86.7% | 2,604 | Best performer - distinct keywords |
| price | 84.9% | 1,567 | Strong signal from cost/value terms |
| display | 83.4% | 988 | Some confusion with camera |
| camera | 82.3% | 2,225 | |
| connectivity | 78.9% | 695 | Limited training samples |
| battery | 78.5% | 2,794 | Occurs in many contexts, some noise |
| design | 73.8% | 1,349 | Overlaps with build quality terms |
| performance | 67.4% | 846 | Most confused - "fast/slow" used across categories |

### Sentiment Classifier - Imbalance Warning

```
              precision    recall  f1-score   support
    negative       0.24      0.81      0.37      1,565
    positive       0.95      0.61      0.74     10,239
```

The sentiment model has **high positive precision (0.95)** but **low negative precision (0.24)** because:
1. Sentiment is derived from the **overall star rating**, not sentence-level opinion
2. The dataset is naturally imbalanced (~82% positive sentences)
3. A 5-star review can contain negative sentences about specific aspects

This is a known limitation of rating-based labeling. Future work should use sentence-level sentiment annotations or class-weighted loss.

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src/absa    # with coverage
```

## Package Structure

```
src/absa/
├── data/
│   └── loader.py              # Loading, cleaning, review-level split, balancing
├── labeling/
│   └── keyword_labeler.py     # Aspect & sentiment labeling
├── models/
│   ├── neural_networks.py      # AspectClassifier, SentimentClassifier (PyTorch)
│   ├── embeddings.py           # DistilBERT embedding extraction
│   └── trainer.py              # Training loops, early stopping, LR scheduling
├── eval/
│   └── metrics.py              # Confusion matrices, macro-F1
├── pipeline.py                 # Orchestrates the full pipeline + baseline comparison
└── __init__.py
```

Config: `configs/config.yaml` - hyperparameters, aspect keywords, dataset URL, paths.

## 📓 Notebook

`notebooks/Aspect_Based_Sentiment_Analysis.ipynb` runs the full pipeline with 50K reviews, including:
- End-to-end execution with timing
- Confidence distribution analysis & visualizations
- Edge case testing on unusual inputs
- Per-aspect accuracy breakdown & confusion matrix
- Interactive single-review analysis

## Known Issues & Future Work

- Keyword-based aspect labeling is noisy; consider sequence labeling (NER) models
- Sentiment is inferred from overall rating, not sentence-level annotations
- Neutral sentiment class (3-star) is underrepresented - currently dropped entirely

## Feedback & Contributions

Found a bug or have an idea? [Open an issue](https://github.com/Sushmitab93/AspectBasedSentimentAnalysis/issues). Contributions welcome - this is a learning project and I'd love to improve it with community input.
