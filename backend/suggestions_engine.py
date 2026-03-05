from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _missing_values_suggestion(missing_report: pd.DataFrame) -> List[Dict[str, Any]]:
    if missing_report.empty or not (missing_report["missing_percentage"] > 0).any():
        return []

    high_missing = missing_report[missing_report["missing_percentage"] > 10.0]
    explanation = (
        "Missing data can bias model training, reduce effective sample size, "
        "and break algorithms that cannot handle null values."
    )
    strategies = [
        "Impute numeric columns using median values.",
        "Impute numeric columns using mean values when distribution is symmetric.",
        "For categorical columns, impute with the most frequent category or a dedicated 'Missing' category.",
        "Drop rows or columns with extremely high missing rates if they are not critical.",
        "Consider advanced techniques such as k-NN or model-based imputation for important features.",
    ]
    impacted_columns = high_missing["column"].tolist()

    return [
        {
            "issue": "Missing Values",
            "severity": "high" if not high_missing.empty else "moderate",
            "explanation": explanation,
            "affected_columns": impacted_columns,
            "suggested_fixes": strategies,
        }
    ]


def _duplicates_suggestion(duplicates_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    total = int(duplicates_info.get("total_duplicates", 0))
    pct = float(duplicates_info.get("percentage_duplicates", 0.0))
    if total == 0:
        return []

    explanation = (
        "Duplicate rows can cause models to overfit specific patterns and distort the true distribution "
        "of the data, especially for rare classes."
    )
    strategies = [
        "Remove exact duplicate rows before training.",
        "Investigate whether duplicates indicate data collection issues.",
        "If duplicates are meaningful (e.g. repeated measurements), aggregate them appropriately.",
    ]
    severity = "high" if pct > 5.0 else "moderate"

    return [
        {
            "issue": "Duplicate Rows",
            "severity": severity,
            "explanation": explanation,
            "affected_columns": [],
            "suggested_fixes": strategies,
        }
    ]


def _outliers_suggestion(outlier_report: pd.DataFrame) -> List[Dict[str, Any]]:
    if outlier_report.empty or not (outlier_report["outlier_percentage"] > 0).any():
        return []

    high_outliers = outlier_report[outlier_report["outlier_percentage"] > 5.0]
    explanation = (
        "Outliers can skew feature distributions, dominate loss functions, and lead models to learn "
        "unstable decision boundaries."
    )
    strategies = [
        "Cap extreme values using IQR-based lower and upper bounds.",
        "Apply log or other monotonic transformations for heavily skewed features.",
        "Remove rows with extreme outliers if they are likely to be data errors.",
        "Use robust models or loss functions that are less sensitive to outliers.",
    ]
    impacted_columns = high_outliers["column"].tolist()
    severity = "high" if not high_outliers.empty else "moderate"

    return [
        {
            "issue": "Outliers",
            "severity": severity,
            "explanation": explanation,
            "affected_columns": impacted_columns,
            "suggested_fixes": strategies,
        }
    ]


def _imbalance_suggestion(imbalance_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not imbalance_info.get("distributions"):
        return []

    if not imbalance_info.get("severe_imbalance", False):
        return []

    target = imbalance_info.get("target_column")
    explanation = (
        "Severe class imbalance can cause models to be biased toward the majority class, leading to poor "
        "recall and precision on minority classes."
    )
    strategies = [
        "Use oversampling techniques such as SMOTE or random oversampling for minority classes.",
        "Use undersampling for majority classes when dataset is large enough.",
        "Apply class weights or cost-sensitive learning to penalize misclassification of minority classes.",
        "Evaluate models using metrics that are robust to imbalance, such as F1-score, ROC-AUC, and PR-AUC.",
    ]

    return [
        {
            "issue": "Class Imbalance",
            "severity": "high",
            "explanation": explanation,
            "affected_columns": [target] if target else [],
            "suggested_fixes": strategies,
        }
    ]


def _correlation_suggestion(correlation_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = correlation_info.get("high_correlation_pairs", []) or []
    if not pairs:
        return []

    explanation = (
        "Highly correlated features can introduce multicollinearity, making model coefficients unstable "
        "and inflating variance without adding new information."
    )
    strategies = [
        "Drop one feature from each highly correlated pair based on domain knowledge or feature importance.",
        "Apply dimensionality reduction techniques such as PCA to combine correlated features.",
        "Use regularization methods (L1/L2) to reduce the impact of redundant features.",
    ]
    impacted_columns = sorted({col for a, b, _ in pairs for col in (a, b)})

    return [
        {
            "issue": "Highly Correlated Features",
            "severity": "moderate",
            "explanation": explanation,
            "affected_columns": impacted_columns,
            "suggested_fixes": strategies,
        }
    ]


def generate_suggestions(
    missing_report: pd.DataFrame,
    duplicates_info: Dict[str, Any],
    outlier_report: pd.DataFrame,
    imbalance_info: Dict[str, Any],
    correlation_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    suggestions.extend(_missing_values_suggestion(missing_report))
    suggestions.extend(_duplicates_suggestion(duplicates_info))
    suggestions.extend(_outliers_suggestion(outlier_report))
    suggestions.extend(_imbalance_suggestion(imbalance_info))
    suggestions.extend(_correlation_suggestion(correlation_info))
    return suggestions

