"""Evaluation metrics: confusion matrix, macro-F1, per-class reports."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)
from typing import Dict, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    task: str = 'multiclass'
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names for reporting
        task: 'multiclass' or 'binary'

    Returns:
        Dictionary with accuracy, macro-F1, confusion matrix, classification report
    """
    accuracy = accuracy_score(y_true, y_pred)

    if task == 'multiclass':
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        macro_f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=False
    )

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'class_names': class_names.tolist() if isinstance(class_names, np.ndarray) else (class_names or np.unique(y_true).tolist())
    }


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None
) -> pd.DataFrame:
    """
    Compute per-class accuracy (recall).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        DataFrame with per-class accuracy
    """
    unique_classes = np.unique(y_true)
    class_names = class_names or unique_classes.tolist()

    accuracies = []
    for cls in unique_classes:
        mask = y_true == cls
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == cls).mean()
            accuracies.append({
                'class': class_names[cls] if isinstance(cls, (int, np.integer)) else cls,
                'accuracy': class_acc,
                'count': mask.sum()
            })

    return pd.DataFrame(accuracies)


def confusion_matrix_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate confusion matrix with normalized values.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Tuple of (confusion_matrix_array, normalized_dataframe)
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by true class
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    if class_names is None:
        class_names = np.unique(y_true).tolist()

    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

    return cm, df_cm


def macro_f1_by_subgroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    group_names: list = None
) -> pd.DataFrame:
    """
    Compute macro-F1 broken down by subgroup.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        groups: Group assignments for each sample
        group_names: Names of groups

    Returns:
        DataFrame with macro-F1 per group
    """
    unique_groups = np.unique(groups)
    group_names = group_names or unique_groups.tolist()

    results = []
    for grp in unique_groups:
        mask = groups == grp
        if mask.sum() > 0:
            macro_f1 = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
            results.append({
                'group': group_names[grp] if isinstance(grp, (int, np.integer)) else grp,
                'macro_f1': macro_f1,
                'count': mask.sum()
            })

    return pd.DataFrame(results)
