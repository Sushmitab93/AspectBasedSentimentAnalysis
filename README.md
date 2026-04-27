# Aspect-Based Sentiment Analysis (ABSA) - Amazon Electronics Reviews

## Overview
An end-to-end NLP pipeline that identifies **which product feature** a review sentence discusses and **how the reviewer feels** about it. Unlike standard sentiment analysis that gives one label per review, this system operates at the sentence level - capturing multiple opinions from a single review.

### Pipeline Overview
| Step | Description |
|------|-------------|
| 1 | Data collection from Amazon Reviews 2023 |
| 2 | Data exploration & inspection |
| 3 | Data cleaning & preprocessing |
| 4 | Text cleaning & sentence tokenization |
| 5 | Aspect labeling (keyword-based) |
| 6 | Sentiment labeling (rating-based) |
| 7 | Save cleaned dataset |
| 8 | Exploratory data analysis |
| 9 | Dataset balancing (aspect × sentiment) |
| 10 | DistilBERT embedding extraction |
| 11 | DNN model training |
| 12 | Binary sentiment model |
| 13 | Evaluation & error analysis |
| 14 | Baseline comparison (zero-shot) |
| 15 | LLM integration (Llama 3 as judge) |

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
