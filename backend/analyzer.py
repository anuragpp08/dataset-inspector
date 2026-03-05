from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import outlier_detection, imbalance_detector, health_score, suggestions_engine


def compute_dataset_summary(df: pd.DataFrame, dtypes: Dict[str, str]) -> Dict[str, Any]:
    numeric_cols = [col for col, t in dtypes.items() if t == "numeric"]
    categorical_cols = [col for col, t in dtypes.items() if t == "categorical"]
    text_cols = [col for col, t in dtypes.items() if t == "text"]

    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "text_columns": text_cols,
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
    }


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_pct = df.isna().mean() * 100
    result = pd.DataFrame({"column": missing_pct.index, "missing_percentage": missing_pct.values})
    result["warning"] = result["missing_percentage"] > 10.0
    return result.sort_values("missing_percentage", ascending=False).reset_index(drop=True)


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    total_duplicates = int(df.duplicated().sum())
    pct_duplicates = float(total_duplicates / len(df) * 100) if len(df) > 0 else 0.0
    duplicate_rows_df = df[df.duplicated(keep=False)]
    return {
        "total_duplicates": total_duplicates,
        "percentage_duplicates": pct_duplicates,
        "duplicate_rows": duplicate_rows_df,
    }


def analyze_outliers(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    return outlier_detection.detect_outliers_iqr(df[numeric_columns])


def analyze_class_imbalance(
    df: pd.DataFrame, categorical_columns: List[str], target_column: Optional[str] = None
) -> Dict[str, Any]:
    return imbalance_detector.detect_class_imbalance(df, categorical_columns, target_column)


def analyze_correlation(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    if not numeric_columns:
        return {"correlation_matrix": pd.DataFrame(), "high_correlation_pairs": []}

    corr_matrix = df[numeric_columns].corr()
    high_pairs: List[Tuple[str, str, float]] = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr_matrix.iloc[i, j]
            if abs(value) > 0.9:
                high_pairs.append((cols[i], cols[j], float(value)))

    return {
        "correlation_matrix": corr_matrix,
        "high_correlation_pairs": high_pairs,
    }


def compute_health_score(
    df: pd.DataFrame,
    missing_report: pd.DataFrame,
    duplicates_info: Dict[str, Any],
    outlier_report: pd.DataFrame,
    imbalance_info: Dict[str, Any],
    correlation_info: Dict[str, Any],
) -> Dict[str, Any]:
    return health_score.calculate_health_score(
        df=df,
        missing_report=missing_report,
        duplicates_info=duplicates_info,
        outlier_report=outlier_report,
        imbalance_info=imbalance_info,
        correlation_info=correlation_info,
    )


def generate_suggestions(
    missing_report: pd.DataFrame,
    duplicates_info: Dict[str, Any],
    outlier_report: pd.DataFrame,
    imbalance_info: Dict[str, Any],
    correlation_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return suggestions_engine.generate_suggestions(
        missing_report=missing_report,
        duplicates_info=duplicates_info,
        outlier_report=outlier_report,
        imbalance_info=imbalance_info,
        correlation_info=correlation_info,
    )


def auto_fix_dataset(
    df: pd.DataFrame,
    numeric_columns: List[str],
) -> pd.DataFrame:
    fixed = df.copy()

    # Remove duplicate rows
    fixed = fixed.drop_duplicates()

    # Fill missing numeric values with median
    for col in numeric_columns:
        if col in fixed.columns:
            median_value = fixed[col].median()
            fixed[col] = fixed[col].fillna(median_value)

    # Cap outliers using IQR limits
    fixed[numeric_columns] = outlier_detection.cap_outliers_iqr(fixed[numeric_columns])

    return fixed


def full_analysis(
    df: pd.DataFrame,
    dtypes: Dict[str, str],
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    summary = compute_dataset_summary(df, dtypes)
    missing_report = analyze_missing_values(df)
    duplicates_info = analyze_duplicates(df)

    numeric_columns = summary["numeric_columns"]
    categorical_columns = summary["categorical_columns"]

    outlier_report = analyze_outliers(df, numeric_columns) if numeric_columns else pd.DataFrame()
    imbalance_info = analyze_class_imbalance(df, categorical_columns, target_column)
    correlation_info = analyze_correlation(df, numeric_columns)

    health = compute_health_score(
        df=df,
        missing_report=missing_report,
        duplicates_info=duplicates_info,
        outlier_report=outlier_report,
        imbalance_info=imbalance_info,
        correlation_info=correlation_info,
    )

    suggestions = generate_suggestions(
        missing_report=missing_report,
        duplicates_info=duplicates_info,
        outlier_report=outlier_report,
        imbalance_info=imbalance_info,
        correlation_info=correlation_info,
    )

    return {
        "summary": summary,
        "missing_report": missing_report,
        "duplicates_info": duplicates_info,
        "outlier_report": outlier_report,
        "imbalance_info": imbalance_info,
        "correlation_info": correlation_info,
        "health": health,
        "suggestions": suggestions,
    }

