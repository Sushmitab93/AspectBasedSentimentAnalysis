"""Aspect and sentiment labeling module."""

from .keyword_labeler import (
    label_aspect,
    label_sentiment,
    apply_aspect_labels,
    apply_sentiment_labels,
)

__all__ = [
    'label_aspect',
    'label_sentiment',
    'apply_aspect_labels',
    'apply_sentiment_labels',
]
