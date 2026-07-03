"""Aspect and sentiment labeling using keyword matching and rating-based heuristics."""

from typing import Dict, List


def label_aspect(sentence: str, aspect_keywords: Dict[str, List[str]]) -> str:
    """
    Assign aspect label based on keyword matching.

    Args:
        sentence: Text to label
        aspect_keywords: Dict mapping aspect names to keyword lists

    Returns:
        Aspect label (or 'general' if no match found)
    """
    sentence_lower = sentence.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(kw in sentence_lower for kw in keywords):
            return aspect
    return 'general'


def label_sentiment(rating: float, thresholds: Dict[str, float]) -> str:
    """
    Assign sentiment label based on star rating.

    Args:
        rating: Star rating (typically 1-5)
        thresholds: Dict with keys:
            - 'positive_threshold': ratings >= this are positive
            - 'neutral_rating': rating equal to this is neutral

    Returns:
        Sentiment label: 'positive', 'neutral', or 'negative'
    """
    pos_thresh = thresholds.get('positive_threshold', 4)
    neut_rating = thresholds.get('neutral_rating', 3)

    if rating >= pos_thresh:
        return 'positive'
    elif rating == neut_rating:
        return 'neutral'
    else:
        return 'negative'


def apply_aspect_labels(df, aspect_keywords: Dict[str, List[str]], sentence_col: str = 'sentences'):
    """Apply aspect labeling to dataframe column."""
    df['aspect'] = df[sentence_col].apply(lambda s: label_aspect(s, aspect_keywords))
    return df


def apply_sentiment_labels(df, thresholds: Dict[str, float], rating_col: str = 'rating'):
    """Apply sentiment labeling to dataframe column."""
    df['sentiment'] = df[rating_col].apply(lambda r: label_sentiment(r, thresholds))
    return df
