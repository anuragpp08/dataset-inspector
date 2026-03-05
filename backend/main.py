from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI

from ..utils.data_type_detector import detect_column_types
from ..utils.file_loader import load_dataset_from_bytes
from . import analyzer

app = FastAPI(title="AI Dataset Inspector API")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


# NOTE:
# For simplicity, the Streamlit frontend imports the analysis modules directly rather than
# calling this HTTP API. This FastAPI app is provided so the analysis capabilities are
# also available as a standalone backend if you wish to expose them as services.

