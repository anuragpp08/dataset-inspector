from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify columns into numeric, categorical, or text.
    """
    types: Dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "categorical"
        else:
            # Heuristic: treat low-cardinality non-numeric as categorical, otherwise text
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
            if unique_ratio < 0.3:
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types

