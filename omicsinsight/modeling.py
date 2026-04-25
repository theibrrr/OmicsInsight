"""Supervised classification with Leave-One-Out cross-validation."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger("omicsinsight")


def run_loo_classification(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run Leave-One-Out CV with Logistic Regression and Random Forest.

    Parameters
    ----------
    X : DataFrame
        Samples × features, **unscaled** (scaling is applied inside each fold).
    y : Series
        Categorical target labels.

    Returns
    -------
    dict
        Per-model accuracy, macro-F1, confusion matrix, and classification report.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)

    model_templates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            solver="lbfgs",
            C=1.0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
    }

    loo = LeaveOneOut()
    X_arr = X.values
    predictions = {name: np.zeros(len(y_encoded), dtype=int) for name in model_templates}

    for train_idx, test_idx in loo.split(X_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train = y_encoded[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, template in model_templates.items():
            model = clone(template)
            model.fit(X_train_s, y_train)
            predictions[name][test_idx[0]] = model.predict(X_test_s)[0]

    results: Dict[str, Any] = {}
    for name, preds in predictions.items():
        acc = accuracy_score(y_encoded, preds)
        f1 = f1_score(y_encoded, preds, average="macro")
        cm = confusion_matrix(y_encoded, preds)
        report = classification_report(
            y_encoded, preds, target_names=class_names, output_dict=True,
        )
        results[name] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "class_names": class_names,
        }
        logger.info("%s LOO-CV — Accuracy: %.4f, Macro F1: %.4f", name, acc, f1)

    return results


def fit_final_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], StandardScaler]:
    """Fit final models on the full dataset (for persistence and feature ranking).

    Returns (models_dict, fitted_scaler).
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    lr = LogisticRegression(
        max_iter=2000, random_state=random_state,
        solver="lbfgs", C=1.0,
    )
    lr.fit(X_scaled, y_encoded)

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_scaled, y_encoded)

    models = {
        "LogisticRegression": lr,
        "RandomForest": rf,
        "_label_encoder": le,
    }
    return models, scaler
