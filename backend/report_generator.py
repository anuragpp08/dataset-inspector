from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _df_to_html_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"<h3>{title}</h3><p>No data available.</p>"
    return f"<h3>{title}</h3>" + df.to_html(classes='table table-striped', border=0)


def _suggestions_to_html(suggestions: List[Dict[str, Any]]) -> str:
    if not suggestions:
        return "<h3>AI Suggestions</h3><p>No critical issues detected. Dataset looks healthy.</p>"

    parts = ["<h3>AI Suggestions</h3>"]
    for s in suggestions:
        parts.append("<div class='card'>")
        parts.append(f"<h4>{s.get('issue')} ({s.get('severity','')})</h4>")
        parts.append(f"<p>{s.get('explanation')}</p>")
        affected_cols = s.get("affected_columns") or []
        if affected_cols:
            parts.append("<p><b>Affected Columns:</b> " + ", ".join(map(str, affected_cols)) + "</p>")
        fixes = s.get("suggested_fixes") or []
        if fixes:
            parts.append("<ul>")
            for fix in fixes:
                parts.append(f"<li>{fix}</li>")
            parts.append("</ul>")
        parts.append("</div>")
    return "\n".join(parts)


def generate_html_report(analysis: Dict[str, Any]) -> str:
    summary = analysis["summary"]
    missing_report: pd.DataFrame = analysis["missing_report"]
    duplicates_info: Dict[str, Any] = analysis["duplicates_info"]
    outlier_report: pd.DataFrame = analysis["outlier_report"]
    correlation_info: Dict[str, Any] = analysis["correlation_info"]
    health = analysis["health"]
    suggestions = analysis["suggestions"]

    corr_matrix: pd.DataFrame = correlation_info.get("correlation_matrix", pd.DataFrame())
    high_corr_pairs = correlation_info.get("high_correlation_pairs", []) or []

    summary_items = [
        f"<li><b>Rows:</b> {summary['n_rows']}</li>",
        f"<li><b>Columns:</b> {summary['n_cols']}</li>",
        f"<li><b>Numeric Columns:</b> {len(summary['numeric_columns'])}</li>",
        f"<li><b>Categorical Columns:</b> {len(summary['categorical_columns'])}</li>",
        f"<li><b>Text Columns:</b> {len(summary['text_columns'])}</li>",
        f"<li><b>Total Missing Values:</b> {summary['total_missing_values']}</li>",
        f"<li><b>Duplicate Rows:</b> {summary['duplicate_rows']}</li>",
    ]

    duplicates_summary_html = (
        f"<p><b>Total Duplicates:</b> {duplicates_info.get('total_duplicates', 0)} "
        f"({duplicates_info.get('percentage_duplicates', 0.0):.2f}% of dataset)</p>"
    )

    high_corr_html = ""
    if high_corr_pairs:
        rows = "".join(
            f"<tr><td>{a}</td><td>{b}</td><td>{v:.3f}</td></tr>" for a, b, v in high_corr_pairs
        )
        high_corr_html = (
            "<h3>Highly Correlated Features (|corr| > 0.9)</h3>"
            "<table class='table table-striped'><thead><tr>"
            "<th>Feature A</th><th>Feature B</th><th>Correlation</th>"
            "</tr></thead><tbody>"
            f"{rows}"
            "</tbody></table>"
        )

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>AI Dataset Inspector Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #222; }}
        .health-score {{
            padding: 10px 15px;
            border-radius: 6px;
            display: inline-block;
            margin-bottom: 20px;
        }}
        .health-score.green {{ background-color: #d4edda; color: #155724; }}
        .health-score.yellow {{ background-color: #fff3cd; color: #856404; }}
        .health-score.red {{ background-color: #f8d7da; color: #721c24; }}
        .card {{ border: 1px solid #ddd; padding: 10px 15px; margin-bottom: 10px; border-radius: 6px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <h1>AI Dataset Inspector Report</h1>
    <div class="health-score {health['color']}">
        <b>Dataset Health Score:</b> {health['score']} / 100 ({health['label']})
    </div>
    <h2>Dataset Summary</h2>
    <ul>
        {''.join(summary_items)}
    </ul>

    <h2>Missing Values Analysis</h2>
    {_df_to_html_table(missing_report, "Missing Values per Column")}

    <h2>Duplicate Rows</h2>
    {duplicates_summary_html}

    <h2>Outlier Analysis</h2>
    {_df_to_html_table(outlier_report, "Outliers per Numeric Column")}

    <h2>Correlation Analysis</h2>
    {_df_to_html_table(corr_matrix, "Correlation Matrix")}
    {high_corr_html}

    <h2>Dataset Health Details</h2>
    <ul>
        {''.join(f'<li>{d}</li>' for d in health.get('details', []))}
    </ul>

    { _suggestions_to_html(suggestions) }
</body>
</html>
""".strip()

    return html

