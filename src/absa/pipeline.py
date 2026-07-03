"""End-to-end ABSA pipeline from raw reviews to trained models."""

import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from absa.data import (
    load_reviews_from_jsonl,
    preprocess_reviews,
    split_by_review,
    remove_general_aspect,
    balance_by_aspect_sentiment,
)
from absa.labeling import apply_aspect_labels, apply_sentiment_labels
from absa.models import DistilBertEmbedder, AspectClassifier, SentimentClassifier, ModelTrainer
from absa.eval import compute_metrics


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Load configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config: dict, output_dir: Path = Path('data/processed')) -> dict:
    """
    Prepare data: load, clean, split, label, and balance.

    Returns:
        Dictionary with train/test dataframes and encoders
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== STEP 1: Loading raw reviews ===")
    df_raw = load_reviews_from_jsonl(
        config['data']['dataset_url'],
        max_reviews=config['data']['max_reviews']
    )
    print(f"Loaded {df_raw.shape[0]} reviews")

    print("\n=== STEP 2: Preprocessing (tokenize, preserve review IDs) ===")
    df_processed = preprocess_reviews(
        df_raw,
        min_sentence_length=config['data']['min_sentence_length']
    )
    print(f"Created {df_processed.shape[0]} sentences from {df_processed['review_id'].nunique()} reviews")

    print("\n=== STEP 3: Labeling (aspects and sentiments) ===")
    df_processed = apply_aspect_labels(df_processed, config['labeling'])
    df_processed = apply_sentiment_labels(df_processed, config['sentiment'])
    print(f"Columns after labeling: {df_processed.columns.tolist()}")
    print(f"Aspect distribution:\n{df_processed['aspect'].value_counts()}")
    print(f"\nSentiment distribution:\n{df_processed['sentiment'].value_counts()}")

    print("\n=== STEP 4: CRITICAL — Splitting at REVIEW level (prevents data leakage) ===")
    train_df, test_df = split_by_review(
        df_processed,
        test_size=config['data']['test_split'],
        random_state=config['data']['random_seed'],
        stratify_by='rating'
    )
    print(f"Columns after split: {train_df.columns.tolist()}")
    print(f"Train: {train_df.shape[0]} sentences from {train_df['review_id'].nunique()} reviews")
    print(f"Test:  {test_df.shape[0]} sentences from {test_df['review_id'].nunique()} reviews")

    # Verify no leakage
    train_reviews = set(train_df['review_id'])
    test_reviews = set(test_df['review_id'])
    assert len(train_reviews & test_reviews) == 0, "Data leakage: reviews in both train and test!"
    print("✓ No data leakage confirmed")

    print("\n=== STEP 5: Removing 'general' aspect and balancing ===")
    train_df = remove_general_aspect(train_df)
    test_df = remove_general_aspect(test_df)
    train_df = balance_by_aspect_sentiment(
        train_df,
        max_samples_per_group=config['balancing']['max_samples_per_aspect_sentiment'],
        random_state=config['data']['random_seed']
    )
    print(f"After balancing: {train_df.shape[0]} training samples")
    print(f"Aspect × Sentiment distribution:\n{train_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)}")

    return {
        'train_df': train_df,
        'test_df': test_df,
        'output_dir': output_dir,
    }


def extract_embeddings(data: dict, config: dict) -> dict:
    """Extract DistilBERT embeddings."""
    print("\n=== STEP 6: Extracting DistilBERT embeddings ===")

    embedder = DistilBertEmbedder(model_name=config['embeddings']['model'])

    print("Extracting train embeddings...")
    train_embeddings = embedder.extract_embeddings(
        data['train_df']['sentences'],
        max_length=config['embeddings']['max_length'],
        batch_size=config['embeddings']['batch_size']
    )

    print("Extracting test embeddings...")
    test_embeddings = embedder.extract_embeddings(
        data['test_df']['sentences'],
        max_length=config['embeddings']['max_length'],
        batch_size=config['embeddings']['batch_size']
    )

    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape:  {test_embeddings.shape}")

    return {
        **data,
        'train_embeddings': train_embeddings,
        'test_embeddings': test_embeddings,
        'embedder': embedder,
    }


def train_aspect_model(data: dict, config: dict) -> dict:
    """Train aspect classification model."""
    print("\n=== STEP 7: Training aspect classifier ===")

    # Encode labels
    aspect_encoder = LabelEncoder()
    train_aspects = aspect_encoder.fit_transform(data['train_df']['aspect'])
    test_aspects = aspect_encoder.transform(data['test_df']['aspect'])

    print(f"Aspects: {aspect_encoder.classes_}")

    # Create model
    model = AspectClassifier(
        embedding_dim=data['train_embeddings'].shape[1],
        num_aspects=len(aspect_encoder.classes_),
        hidden_dims=config['models']['aspect']['hidden_dims'],
        dropout=config['models']['aspect']['dropout'],
        l2_reg=config['models']['aspect']['l2_regularization']
    )

    # Train
    trainer = ModelTrainer(
        model,
        learning_rate=config['models']['aspect']['learning_rate']
    )

    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        data['train_embeddings'], train_aspects,
        test_size=config['training']['validation_split'],
        random_state=config['data']['random_seed'],
        stratify=train_aspects
    )

    history = trainer.train_aspect_classifier(
        X_train, y_train, X_val, y_val,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        verbose=True
    )

    # Evaluate
    test_preds = trainer.predict(data['test_embeddings'])
    test_preds = test_preds.argmax(axis=1)

    metrics = compute_metrics(
        test_aspects, test_preds,
        class_names=aspect_encoder.classes_,
        task='multiclass'
    )

    print(f"\n=== Aspect Model Results ===")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1:      {metrics['macro_f1']:.4f}")
    print(f"\n{metrics['classification_report']}")

    return {
        **data,
        'aspect_model': model,
        'aspect_encoder': aspect_encoder,
        'aspect_trainer': trainer,
        'aspect_metrics': metrics,
    }


def train_sentiment_model(data: dict, config: dict) -> dict:
    """Train binary sentiment classification model."""
    print("\n=== STEP 8: Training binary sentiment classifier ===")

    # Filter to binary classes (remove neutral)
    train_binary = data['train_df'][data['train_df']['sentiment'] != 'neutral']
    test_binary = data['test_df'][data['test_df']['sentiment'] != 'neutral']

    # Get corresponding embeddings (positional index, not label index)
    train_emb = data['train_embeddings'][data['train_df'].index.get_indexer(train_binary.index)]
    test_emb = data['test_embeddings'][data['test_df'].index.get_indexer(test_binary.index)]

    # Encode labels
    sentiment_encoder = LabelEncoder()
    train_sentiments = sentiment_encoder.fit_transform(train_binary['sentiment'])
    test_sentiments = sentiment_encoder.transform(test_binary['sentiment'])

    print(f"Sentiments: {sentiment_encoder.classes_}")

    # Create model
    model = SentimentClassifier(
        embedding_dim=data['train_embeddings'].shape[1],
        hidden_dims=config['models']['sentiment']['hidden_dims'],
        dropout=config['models']['sentiment']['dropout'],
        l2_reg=config['models']['sentiment']['l2_regularization']
    )

    # Train
    trainer = ModelTrainer(
        model,
        learning_rate=config['models']['sentiment']['learning_rate']
    )

    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        train_emb, train_sentiments,
        test_size=config['training']['validation_split'],
        random_state=config['data']['random_seed'],
        stratify=train_sentiments
    )

    history = trainer.train_sentiment_classifier(
        X_train, y_train, X_val, y_val,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        verbose=True
    )

    # Evaluate
    test_preds = trainer.predict(test_emb)
    test_preds = (test_preds > 0.5).astype(int).flatten()

    metrics = compute_metrics(
        test_sentiments, test_preds,
        class_names=sentiment_encoder.classes_,
        task='binary'
    )

    print(f"\n=== Sentiment Model Results ===")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1:      {metrics['macro_f1']:.4f}")
    print(f"\n{metrics['classification_report']}")

    return {
        **data,
        'sentiment_model': model,
        'sentiment_encoder': sentiment_encoder,
        'sentiment_trainer': trainer,
        'sentiment_metrics': metrics,
    }


def run_baseline(data: dict, config: dict) -> None:
    """LogisticRegression baseline on the same embeddings for comparison."""
    print("\n" + "=" * 70)
    print("  BASELINE: LogisticRegression on DistilBERT embeddings")
    print("=" * 70)

    # Aspect baseline
    print("\n--- Aspect Classification (LogisticRegression) ---")
    aspect_encoder = LabelEncoder()
    train_aspects = aspect_encoder.fit_transform(data['train_df']['aspect'])
    test_aspects = aspect_encoder.transform(data['test_df']['aspect'])

    lr_aspect = LogisticRegression(max_iter=1000, multi_class='multinomial')
    lr_aspect.fit(data['train_embeddings'], train_aspects)
    lr_aspect_preds = lr_aspect.predict(data['test_embeddings'])

    aspect_metrics = compute_metrics(
        test_aspects, lr_aspect_preds,
        class_names=aspect_encoder.classes_,
        task='multiclass'
    )
    print(f"Test Accuracy: {aspect_metrics['accuracy']:.4f}")
    print(f"Macro-F1:      {aspect_metrics['macro_f1']:.4f}")

    # Sentiment baseline
    print("\n--- Sentiment Classification (LogisticRegression) ---")
    train_binary = data['train_df'][data['train_df']['sentiment'] != 'neutral']
    test_binary = data['test_df'][data['test_df']['sentiment'] != 'neutral']

    train_emb = data['train_embeddings'][data['train_df'].index.get_indexer(train_binary.index)]
    test_emb = data['test_embeddings'][data['test_df'].index.get_indexer(test_binary.index)]

    sentiment_encoder = LabelEncoder()
    train_sentiments = sentiment_encoder.fit_transform(train_binary['sentiment'])
    test_sentiments = sentiment_encoder.transform(test_binary['sentiment'])

    lr_sent = LogisticRegression(max_iter=1000)
    lr_sent.fit(train_emb, train_sentiments)
    lr_sent_preds = lr_sent.predict(test_emb)

    sent_metrics = compute_metrics(
        test_sentiments, lr_sent_preds,
        class_names=sentiment_encoder.classes_,
        task='binary'
    )
    print(f"Test Accuracy: {sent_metrics['accuracy']:.4f}")
    print(f"Macro-F1:      {sent_metrics['macro_f1']:.4f}")

    # Comparison summary
    print("\n" + "-" * 70)
    print("BASELINE VS DNN COMPARISON")
    print("-" * 70)
    print(f"{'Task':<15} {'Metric':<12} {'DNN':<10} {'LR Baseline':<12} {'Delta':<10}")
    print(f"{'Aspect':<15} {'Accuracy':<12} {data['aspect_metrics']['accuracy']:<10.4f} {aspect_metrics['accuracy']:<12.4f} {data['aspect_metrics']['accuracy'] - aspect_metrics['accuracy']:<+10.4f}")
    print(f"{'Aspect':<15} {'Macro-F1':<12} {data['aspect_metrics']['macro_f1']:<10.4f} {aspect_metrics['macro_f1']:<12.4f} {data['aspect_metrics']['macro_f1'] - aspect_metrics['macro_f1']:<+10.4f}")
    print(f"{'Sentiment':<15} {'Accuracy':<12} {data['sentiment_metrics']['accuracy']:<10.4f} {sent_metrics['accuracy']:<12.4f} {data['sentiment_metrics']['accuracy'] - sent_metrics['accuracy']:<+10.4f}")
    print(f"{'Sentiment':<15} {'Macro-F1':<12} {data['sentiment_metrics']['macro_f1']:<10.4f} {sent_metrics['macro_f1']:<12.4f} {data['sentiment_metrics']['macro_f1'] - sent_metrics['macro_f1']:<+10.4f}")


def run_full_pipeline(config_path: str = 'configs/config.yaml') -> dict:
    """Run the complete ABSA pipeline."""
    print("=" * 70)
    print("  ASPECT-BASED SENTIMENT ANALYSIS PIPELINE")
    print("=" * 70)

    config = load_config(config_path)

    data = prepare_data(config)
    data = extract_embeddings(data, config)
    data = train_aspect_model(data, config)
    data = train_sentiment_model(data, config)
    run_baseline(data, config)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    return data


if __name__ == '__main__':
    results = run_full_pipeline()
