"""Pytest configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_embeddings():
    """Sample DistilBERT embeddings."""
    return np.random.randn(100, 768).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Sample aspect labels."""
    aspects = ['battery', 'camera', 'sound', 'display', 'price', 'performance', 'design', 'connectivity']
    return np.random.choice(aspects, 100)
