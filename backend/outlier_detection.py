from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def detect_outliers_iqr(df_numeric: pd.DataFrame) -> pd.DataFrame:
    if df_numeric.empty:
        return pd.DataFrame(columns=["column", "n_outliers", "outlier_percentage"])

    q1 = df_numeric.quantile(0.25)
    q3 = df_numeric.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_flags = (df_numeric.lt(lower_bound)) | (df_numeric.gt(upper_bound))
    n_outliers = outlier_flags.sum()
    pct_outliers = (n_outliers / len(df_numeric)) * 100

    result = pd.DataFrame(
        {
            "column": n_outliers.index,
            "n_outliers": n_outliers.values,
            "outlier_percentage": pct_outliers.values,
        }
    )
    return result.sort_values("outlier_percentage", ascending=False).reset_index(drop=True)


def cap_outliers_iqr(df_numeric: pd.DataFrame) -> pd.DataFrame:
    if df_numeric.empty:
        return df_numeric

    q1 = df_numeric.quantile(0.25)
    q3 = df_numeric.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    capped = df_numeric.copy()
    for col in df_numeric.columns:
        lb = lower_bound[col]
        ub = upper_bound[col]
        capped[col] = np.where(capped[col] < lb, lb, capped[col])
        capped[col] = np.where(capped[col] > ub, ub, capped[col])
    return capped

