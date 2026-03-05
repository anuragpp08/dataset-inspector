from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def detect_class_imbalance(
    df: pd.DataFrame, categorical_columns: List[str], target_column: Optional[str] = None
) -> Dict[str, Any]:
    if df.empty or not categorical_columns:
        return {"target_column": None, "distributions": {}, "severe_imbalance": False}

    # Prefer explicit target if provided and valid, otherwise use first categorical column
    target = target_column if target_column in df.columns else categorical_columns[0]
    series = df[target].astype("category")
    value_counts = series.value_counts(dropna=False)
    percentages = (value_counts / len(series)) * 100

    distributions = {
        str(cls): {"count": int(count), "percentage": float(percentages[cls])}
        for cls, count in value_counts.items()
    }

    max_pct = float(percentages.max()) if not percentages.empty else 0.0
    severe_imbalance = max_pct > 80.0

    return {"target_column": target, "distributions": distributions, "severe_imbalance": severe_imbalance}

