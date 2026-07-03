"""Tests for data loading, cleaning, and splitting logic."""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from absa.data import (
    clean_text,
    preprocess_reviews,
    split_by_review,
    remove_general_aspect,
    balance_by_aspect_sentiment,
)
from absa.labeling import label_aspect, label_sentiment


def test_clean_text_removes_noise():
    assert clean_text('<p>hello</p>') == 'hello'
    assert clean_text('visit http://example.com today') == 'visit today'
    assert clean_text('hello!@#$%world') == 'hello!world'


def test_clean_text_normalizes():
    assert clean_text('hello    world') == 'hello world'
    assert clean_text('HELLO World') == 'hello world'


def test_clean_text_combined():
    text = '<p>Visit http://example.com! Price: $99.99</p>'
    assert clean_text(text) == 'visit price 99.99'


@pytest.fixture
def sample_reviews():
    return pd.DataFrame({
        'text': [
            'Great product! It works well.',
            'Battery lasts all day. Display is bright.',
            'Bad quality. Not recommended.',
        ],
        'title': ['Good', 'Good', 'Bad'],
        'rating': [5.0, 4.0, 1.0],
        'verified_purchase': [True, True, True],
        'helpful_vote': [10, 5, 2],
    })


def test_preprocess_preserves_review_ids(sample_reviews):
    result = preprocess_reviews(sample_reviews, min_sentence_length=1)
    assert 'review_id' in result.columns
    assert result['review_id'].nunique() == 3


def test_preprocess_splits_into_sentences(sample_reviews):
    result = preprocess_reviews(sample_reviews, min_sentence_length=1)
    assert len(result) > len(sample_reviews)
    assert 'great product' in result['sentences'].values[0]


def test_preprocess_filters_short_sentences(sample_reviews):
    result = preprocess_reviews(sample_reviews, min_sentence_length=10)
    assert len(result) == 0


def test_preprocess_removes_duplicates():
    df = pd.DataFrame({
        'text': ['Hello world', 'Hello world', 'Goodbye world'],
        'title': ['A', 'A', 'B'],
        'rating': [5.0, 5.0, 1.0],
        'verified_purchase': [True, True, True],
        'helpful_vote': [1, 1, 1],
    })
    result = preprocess_reviews(df, min_sentence_length=1)
    assert result['review_id'].nunique() == 2


@pytest.fixture
def sample_sentences():
    data = []
    for rev_id in range(3):
        for sent_id in range(2):
            data.append({'review_id': rev_id, 'sentences': f'Sentence {sent_id}', 'rating': [5.0, 3.0, 1.0][rev_id]})
    return pd.DataFrame(data)


def test_split_by_review_no_overlap(sample_sentences):
    train, test = split_by_review(sample_sentences, test_size=0.33, random_state=42)
    train_reviews = set(train['review_id'])
    test_reviews = set(test['review_id'])
    assert len(train_reviews & test_reviews) == 0


def test_split_by_review_preserves_all_reviews(sample_sentences):
    train, test = split_by_review(sample_sentences, test_size=0.33, random_state=42)
    all_reviews = set(train['review_id']) | set(test['review_id'])
    assert all_reviews == set(sample_sentences['review_id'])


def test_remove_general_aspect():
    df = pd.DataFrame({
        'aspect': ['battery', 'general', 'sound', 'general'],
        'sentences': ['a', 'b', 'c', 'd'],
    })
    result = remove_general_aspect(df)
    assert len(result) == 2
    assert 'general' not in result['aspect'].values


def test_remove_general_aspect_keeps_specific():
    df = pd.DataFrame({
        'aspect': ['battery', 'camera', 'sound', 'display'],
        'sentences': ['a', 'b', 'c', 'd'],
    })
    result = remove_general_aspect(df)
    assert len(result) == 4


def test_balance_caps_at_max_samples():
    df = pd.DataFrame({
        'aspect': ['battery'] * 100 + ['camera'] * 50,
        'sentiment': ['positive'] * 150,
        'text': [f'sent{i}' for i in range(150)],
    })
    result = balance_by_aspect_sentiment(df, max_samples_per_group=60)
    for _, group in result.groupby(['aspect', 'sentiment']):
        assert len(group) <= 60


def test_balance_preserves_groups():
    df = pd.DataFrame({
        'aspect': ['battery', 'battery', 'camera', 'camera'],
        'sentiment': ['positive', 'negative', 'positive', 'negative'],
        'text': ['a', 'b', 'c', 'd'],
    })
    result = balance_by_aspect_sentiment(df, max_samples_per_group=100)
    assert result.groupby(['aspect', 'sentiment']).ngroups == 4


def test_label_aspect_by_keyword():
    keywords = {'battery': ['battery', 'charge'], 'camera': ['camera', 'photo']}
    assert label_aspect('The battery is great', keywords) == 'battery'
    assert label_aspect('Nice camera quality', keywords) == 'camera'


def test_label_aspect_returns_general_if_no_match():
    keywords = {'battery': ['battery', 'charge']}
    assert label_aspect('This product is nice', keywords) == 'general'


def test_label_sentiment_by_rating():
    thresholds = {'positive_threshold': 4, 'neutral_rating': 3}
    assert label_sentiment(5.0, thresholds) == 'positive'
    assert label_sentiment(3.0, thresholds) == 'neutral'
    assert label_sentiment(2.0, thresholds) == 'negative'
