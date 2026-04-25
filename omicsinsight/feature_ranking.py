"""Feature ranking by variance, logistic regression coefficients, and RF importance."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("omicsinsight")


def rank_by_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Rank features (columns) by variance, descending."""
    variances = df.var(axis=0).sort_values(ascending=False)
    return pd.DataFrame({
        "feature": variances.index,
        "variance": variances.values,
        "variance_rank": range(1, len(variances) + 1),
    })


def rank_by_logreg_coef(
    model: Any, feature_names: List[str]
) -> pd.DataFrame:
    """Rank features by mean absolute logistic-regression coefficient."""
    mean_coefs = np.abs(model.coef_).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "logreg_importance": mean_coefs,
    })
    ranking = ranking.sort_values("logreg_importance", ascending=False)
    ranking["logreg_rank"] = range(1, len(ranking) + 1)
    return ranking.reset_index(drop=True)


def rank_by_rf_importance(
    model: Any, feature_names: List[str]
) -> pd.DataFrame:
    """Rank features by Random Forest Gini importance."""
    ranking = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": model.feature_importances_,
    })
    ranking = ranking.sort_values("rf_importance", ascending=False)
    ranking["rf_rank"] = range(1, len(ranking) + 1)
    return ranking.reset_index(drop=True)


def combine_rankings(
    variance_rank: pd.DataFrame,
    logreg_rank: Optional[pd.DataFrame] = None,
    rf_rank: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge multiple rankings and compute an average rank."""
    combined = variance_rank.copy()

    if logreg_rank is not None:
        combined = combined.merge(
            logreg_rank[["feature", "logreg_importance", "logreg_rank"]],
            on="feature", how="left",
        )
    if rf_rank is not None:
        combined = combined.merge(
            rf_rank[["feature", "rf_importance", "rf_rank"]],
            on="feature", how="left",
        )

    rank_cols = [c for c in combined.columns if c.endswith("_rank")]
    combined["avg_rank"] = combined[rank_cols].mean(axis=1)
    combined = combined.sort_values("avg_rank")
    return combined.reset_index(drop=True)


def get_top_features(
    combined_ranking: pd.DataFrame, top_n: int = 20
) -> Dict[str, Any]:
    """Return a JSON-serializable dict of the top-ranked features."""
    top = combined_ranking.head(top_n)
    features = []
    for _, row in top.iterrows():
        entry: Dict[str, Any] = {"feature": row["feature"]}
        for col in top.columns:
            if col == "feature":
                continue
            val = row[col]
            if pd.notna(val):
                entry[col] = round(float(val), 6) if isinstance(val, float) else int(val)
        features.append(entry)
    return {"top_n": top_n, "features": features}
