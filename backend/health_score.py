from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def calculate_health_score(
    df: pd.DataFrame,
    missing_report: pd.DataFrame,
    duplicates_info: Dict[str, Any],
    outlier_report: pd.DataFrame,
    imbalance_info: Dict[str, Any],
    correlation_info: Dict[str, Any],
) -> Dict[str, Any]:
    score = 100
    details = []

    # Missing values > 10% anywhere
    if not missing_report.empty and (missing_report["missing_percentage"] > 10.0).any():
        score -= 20
        details.append("Missing values exceed 10% in at least one column (-20).")

    # Duplicates > 2%
    dup_pct = float(duplicates_info.get("percentage_duplicates", 0.0))
    if dup_pct > 2.0:
        score -= 10
        details.append("Duplicate rows exceed 2% of dataset (-10).")

    # Outliers > 5% in any numeric column
    if not outlier_report.empty and (outlier_report["outlier_percentage"] > 5.0).any():
        score -= 20
        details.append("Outliers exceed 5% in at least one numeric column (-20).")

    # Severe class imbalance
    if imbalance_info.get("severe_imbalance", False):
        score -= 30
        details.append("Severe class imbalance detected (-30).")

    # High correlation features
    high_corr = correlation_info.get("high_correlation_pairs", []) or []
    if high_corr:
        score -= 20
        details.append("Highly correlated feature pairs with |corr| > 0.9 detected (-20).")

    score = max(0, min(100, score))

    if score >= 80:
        label = "Good"
        color = "green"
    elif score >= 60:
        label = "Moderate"
        color = "yellow"
    else:
        label = "Poor"
        color = "red"

    return {"score": score, "label": label, "color": color, "details": details}

