from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import io

import pandas as pd


MAX_FILE_SIZE_MB = 20
MAX_ROWS = 100_000
MAX_COLUMNS = 100


@dataclass
class DatasetValidationResult:
    df: Optional[pd.DataFrame]
    error: Optional[str]
    warning: Optional[str]


def _validate_shape(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    rows, cols = df.shape
    if rows > MAX_ROWS or cols > MAX_COLUMNS:
        return (
            False,
            f"Dataset has shape ({rows} rows, {cols} columns), which exceeds the limit "
            f"of {MAX_ROWS} rows and {MAX_COLUMNS} columns.",
        )
    return True, None


def load_dataset_from_bytes(filename: str, file_bytes: bytes) -> DatasetValidationResult:
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return DatasetValidationResult(
            df=None,
            error=(
                f"File size {size_mb:.2f} MB exceeds the maximum allowed {MAX_FILE_SIZE_MB} MB."
            ),
            warning=None,
        )

    ext = filename.lower().split(".")[-1]
    buffer = io.BytesIO(file_bytes)

    try:
        if ext == "csv":
            df = pd.read_csv(buffer)
        elif ext in ("xls", "xlsx"):
            df = pd.read_excel(buffer)
        else:
            return DatasetValidationResult(
                df=None,
                error="Unsupported file format. Please upload a CSV or XLSX file.",
                warning=None,
            )
    except Exception as exc:  # pragma: no cover - defensive
        return DatasetValidationResult(
            df=None,
            error=f"Failed to read file: {exc}",
            warning=None,
        )

    ok, warning = _validate_shape(df)
    return DatasetValidationResult(df=df if ok else df.head(MAX_ROWS), error=None, warning=warning)

