## AI Dataset Inspector

AI Dataset Inspector is a full-stack data quality dashboard built with **FastAPI**, **Pandas**, **NumPy**, **Scikit-learn**, **Plotly**, and **Streamlit**. It helps you quickly assess whether a tabular dataset is ready for machine learning and suggests concrete fixes.

### Features

- **Dataset upload** (CSV, XLSX) with limits:
  - Max file size: 10 MB
  - Max rows: 100,000
  - Max columns: 100
- **Dataset summary**: rows, columns, numeric/categorical/text columns, missing values, duplicate rows.
- **Missing value analysis**:
  - Per-column missing percentage.
  - Plotly bar chart.
  - Columns with >10% missing flagged as warnings.
- **Duplicate detection**:
  - Total and percentage of duplicate rows.
  - Download duplicate rows as CSV.
- **Outlier detection** (IQR method):
  - Outliers per numeric column.
  - Plotly boxplots for numeric columns.
- **Class imbalance detection**:
  - Class distribution for a categorical target column.
  - Pie chart visualization.
  - Warning when one class >80%.
- **Correlation analysis**:
  - Correlation matrix for numeric columns.
  - Plotly heatmap.
  - Warnings for feature pairs with |corr| > 0.9.
- **Data distributions**:
  - Histograms and boxplots for numeric columns.
  - Bar charts for categorical columns.
- **Dataset health score** (0–100) with penalties:
  - Missing values >10% → −20
  - Duplicates >2% → −10
  - Outliers >5% → −20
  - Severe class imbalance → −30
  - Highly correlated features → −20
- **AI-style explanations & fix suggestions** (rule-based).
- **Auto-fix button**:
  - Remove duplicates.
  - Impute numeric missing values with median.
  - Cap numeric outliers using IQR.
  - Download cleaned dataset.
- **Report generation**:
  - Downloadable HTML report containing all key analyses and suggestions.

### Project Structure

```text
ai_dataset_inspector/
  backend/
    main.py
    analyzer.py
    outlier_detection.py
    imbalance_detector.py
    health_score.py
    suggestions_engine.py
    report_generator.py
  frontend/
    .streamlit/
      config.toml
    app.py
  utils/
    file_loader.py
    chart_utils.py
    data_type_detector.py
  requirements.txt
  README.md
```

### Installation

1. Create and activate a virtual environment (recommended).

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   # source .venv/bin/activate  # on macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Streamlit Dashboard

From the project root (where `frontend/` lives), run:

```bash
streamlit run frontend/app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

### (Optional) Running the FastAPI Backend

The main analysis logic is implemented in reusable backend modules and used directly by Streamlit. An optional FastAPI app is also provided if you want to expose the analysis as an API.

To run the FastAPI app:

```bash
uvicorn backend.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` to see the interactive API docs.

### Notes

- The app processes datasets fully in memory and enforces size limits to avoid excessive resource usage.
- Temporary data is kept in memory only for the session; no files are written to disk by default.

