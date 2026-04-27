# Aspect-Based Sentiment Analysis (ABSA) — Amazon Electronics Reviews

## Overview
An end-to-end NLP pipeline that identifies **which product feature** a review sentence discusses and **how the reviewer feels** about it. Unlike standard sentiment analysis that gives one label per review, this system operates at the sentence level — capturing multiple opinions from a single review.

## Pipeline
1. **Data Collection** — Stream 50,000 electronics reviews from McAuley Lab Amazon Reviews 2023
2. **Preprocessing** — Text cleaning, sentence tokenization (NLTK), short sentence filtering
3. **Labeling** — Keyword-based aspect labeling (8 categories), rating-based sentiment labeling
4. **Balancing** — Joint aspect × sentiment sampling to handle class imbalance
5. **Embeddings** — Frozen DistilBERT CLS token extraction (768-dim)
6. **Training** — Two custom DNNs: aspect classifier (8-class softmax) and binary sentiment classifier (sigmoid)
7. **Evaluation** — Confusion matrices, per-aspect accuracy breakdown, misclassified example analysis
8. **Baseline** — Zero-shot BART-MNLI comparison to validate trained model performance
9. **Explainability** — LLaMA as independent judge (makes its own prediction, then agrees/disagrees with DNN)

## Results
| Model | Architecture | Accuracy |
|-------|-------------|----------|
| Aspect Classifier | DistilBERT + DNN (768→512→256→8) | 76.39% |
| Sentiment Classifier | DistilBERT + DNN (768→512→256→1) | 71.77% |
| Zero-Shot Baseline | BART-MNLI (no training) | Compared per-aspect |

## Tech Stack
- **Embeddings:** DistilBERT (HuggingFace Transformers)
- **Models:** TensorFlow / Keras
- **Baseline:** BART-MNLI (zero-shot classification)
- **LLM:** LLaMA via Groq API
- **NLP:** NLTK, scikit-learn
- **Data:** pandas, NumPy, matplotlib, seaborn

## How to Run
1. Open `Aspect_Based_Sentiment_Analysis.ipynb` in Google Colab
2. Set your Groq API key in Colab Secrets (key icon in sidebar) as `GROQ_API_KEY`
3. Run all cells top to bottom (GPU runtime recommended for embedding extraction)

## Dataset
[McAuley Lab Amazon Reviews 2023 — Electronics](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
