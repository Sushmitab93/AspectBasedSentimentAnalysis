"""Data loading, cleaning, and preprocessing for ABSA."""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Tuple
import nltk
from nltk.tokenize import sent_tokenize


def _ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')


def load_reviews_from_jsonl(url: str, max_reviews: int = 50000) -> pd.DataFrame:
    """
    Stream reviews line-by-line from HuggingFace dataset.

    Args:
        url: HuggingFace dataset URL (JSONL format)
        max_reviews: Maximum number of reviews to load

    Returns:
        DataFrame with raw review data
    """
    import requests

    raw_records = []
    try:
        with requests.get(url, stream=True) as r:
            for i, line in enumerate(r.iter_lines()):
                if i >= max_reviews:
                    break
                if line:
                    raw_records.append(json.loads(line))
    except Exception as e:
        raise RuntimeError(f"Error downloading data: {e}")

    df = pd.DataFrame(raw_records)
    return df


def clean_text(text: str) -> str:
    """Remove noise from review text."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)                  # remove HTML tags
    text = re.sub(r'http\S+', '', text)                # remove URLs
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)        # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()           # remove extra spaces
    return text


def preprocess_reviews(df: pd.DataFrame, min_sentence_length: int = 5) -> pd.DataFrame:
    """
    Clean reviews, split into sentences, and filter by minimum length.

    CRITICAL: This function preserves review IDs for later review-level splitting.
    Sentences from the same review are grouped together so they can be split
    BEFORE tokenization to prevent data leakage.

    Args:
        df: Raw reviews DataFrame
        min_sentence_length: Minimum words per sentence to retain

    Returns:
        DataFrame with columns: review_id, text, title, rating, verified_purchase,
                                helpful_vote, clean_text, sentences
    """
    # Keep only relevant columns
    df = df[['text', 'title', 'rating', 'verified_purchase', 'helpful_vote']].copy()

    # Drop rows with missing text or rating
    df = df.dropna(subset=['text', 'rating'])

    # Remove exact duplicates
    df = df.drop_duplicates(subset=['text'])

    # Add review ID based on index (unique identifier for each review)
    df = df.reset_index(drop=True)
    df['review_id'] = df.index

    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    # Split into sentences
    _ensure_nltk_data()
    df['sentences'] = df['clean_text'].apply(sent_tokenize)

    # Explode sentences (one row per sentence, preserving review_id)
    df = df.explode('sentences', ignore_index=False)

    # Filter by minimum sentence length
    df = df[df['sentences'].str.split().str.len() >= min_sentence_length]

    # Reset index but keep review_id
    df = df.reset_index(drop=True)

    return df[['review_id', 'sentences', 'rating', 'title', 'verified_purchase', 'helpful_vote']]


def split_by_review(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data at the REVIEW level, not sentence level.

    This is critical to prevent data leakage: sentences from the same review
    must not appear in both train and test sets.

    Args:
        df: DataFrame with review_id column (from preprocess_reviews)
        test_size: Fraction of reviews for test set
        random_state: Random seed for reproducibility
        stratify_by: Column name to stratify split (e.g., 'rating' for balanced splits)

    Returns:
        Tuple of (train_df, test_df) with all sentences from each review intact
    """
    from sklearn.model_selection import train_test_split

    # Get unique reviews
    reviews = df.groupby('review_id').first().reset_index()

    if stratify_by and stratify_by in reviews.columns:
        stratify_col = reviews[stratify_by]
    else:
        stratify_col = None

    # Split at review level
    train_reviews, test_reviews = train_test_split(
        reviews,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    train_review_ids = set(train_reviews['review_id'])
    test_review_ids = set(test_reviews['review_id'])

    # Separate sentences by review membership
    train_df = df[df['review_id'].isin(train_review_ids)].reset_index(drop=True)
    test_df = df[df['review_id'].isin(test_review_ids)].reset_index(drop=True)

    return train_df, test_df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save processed dataframe to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {df.shape[0]} records to {output_path}")


def remove_general_aspect(df: pd.DataFrame, aspect_col: str = 'aspect') -> pd.DataFrame:
    """Remove sentences with 'general' aspect (no specific keyword matched)."""
    return df[df[aspect_col] != 'general'].copy()


def balance_by_aspect_sentiment(
    df: pd.DataFrame,
    max_samples_per_group: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance dataset at (aspect, sentiment) combination level.

    Args:
        df: DataFrame with 'aspect' and 'sentiment' columns
        max_samples_per_group: Max samples per (aspect, sentiment) pair
        random_state: Random seed

    Returns:
        Balanced DataFrame
    """
    balanced = pd.concat([
        group.sample(n=min(len(group), max_samples_per_group), random_state=random_state)
        for _, group in df.groupby(['aspect', 'sentiment'])
    ], ignore_index=True)

    return balanced
