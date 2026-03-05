
# import streamlit as st
from __future__ import annotations
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend import analyzer, report_generator

from typing import Any, Dict, Optional

import streamlit as st

from backend import analyzer, report_generator
from utils.chart_utils import (
    categorical_distribution_charts,
    class_distribution_pie,
    correlation_heatmap,
    duplicate_pie_chart,
    missing_values_bar_chart,
    numeric_distribution_charts,
    outlier_boxplots,
)
from utils.data_type_detector import detect_column_types
from utils.file_loader import DatasetValidationResult, load_dataset_from_bytes


st.set_page_config(
    page_title="AI Dataset Inspector",
    layout="wide",
)


def _render_summary_cards(summary: Dict[str, Any]):
    cols = st.columns(4)
    with cols[0]:
        st.metric("Rows", summary["n_rows"])
        st.metric("Numeric Columns", len(summary["numeric_columns"]))
    with cols[1]:
        st.metric("Columns", summary["n_cols"])
        st.metric("Categorical Columns", len(summary["categorical_columns"]))
    with cols[2]:
        st.metric("Text Columns", len(summary["text_columns"]))
        st.metric("Missing Values", summary["total_missing_values"])
    with cols[3]:
        st.metric("Duplicate Rows", summary["duplicate_rows"])


def main():
    st.title("AI Dataset Inspector")
    st.markdown(
        "Upload your dataset and automatically analyze its quality and readiness for machine learning."
    )

    st.markdown("### 1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

    if not uploaded_file:
        st.info("Please upload a dataset to begin.")
        return

    data_bytes = uploaded_file.read()
    load_result: DatasetValidationResult = load_dataset_from_bytes(
        filename=uploaded_file.name, file_bytes=data_bytes
    )

    if load_result.error:
        st.error(load_result.error)
        return
    if load_result.warning:
        st.warning(load_result.warning)

    df = load_result.df
    if df is None:
        st.error("Failed to load dataset.")
        return

    st.success("Dataset loaded successfully.")

    dtypes = detect_column_types(df)

    with st.expander("Preview Dataset", expanded=False):
        st.write(df.head())

    # Full analysis
    analysis = analyzer.full_analysis(df, dtypes)
    summary = analysis["summary"]
    missing_report = analysis["missing_report"]
    duplicates_info = analysis["duplicates_info"]
    outlier_report = analysis["outlier_report"]
    imbalance_info = analysis["imbalance_info"]
    correlation_info = analysis["correlation_info"]
    health = analysis["health"]
    suggestions = analysis["suggestions"]

    st.markdown("### 2. Dataset Summary")
    _render_summary_cards(summary)

    st.markdown("### 3. Missing Values Analysis")
    st.dataframe(missing_report)
    missing_fig = missing_values_bar_chart(missing_report)
    st.plotly_chart(missing_fig, use_container_width=True)

    st.markdown("### 4. Duplicate Detection")
    total_dups = duplicates_info["total_duplicates"]
    pct_dups = duplicates_info["percentage_duplicates"]
    st.write(f"Total duplicate rows: **{total_dups}** ({pct_dups:.2f}% of dataset)")

    if total_dups > 0:
        dup_df = duplicates_info["duplicate_rows"]
        st.dataframe(dup_df.head())
        pie_fig = duplicate_pie_chart(total_dups, len(df) - total_dups)
        st.plotly_chart(pie_fig, use_container_width=True)

        csv_bytes = dup_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Duplicate Rows as CSV",
            data=csv_bytes,
            file_name="duplicate_rows.csv",
            mime="text/csv",
        )
    else:
        st.info("No duplicate rows detected.")

    st.markdown("### 5. Outlier Analysis")
    if not outlier_report.empty and summary["numeric_columns"]:
        st.dataframe(outlier_report)
        boxplots = outlier_boxplots(df, summary["numeric_columns"])
        cols = st.columns(3)
        for i, fig in enumerate(boxplots):
            with cols[i % 3]:
                st.plotly_chart(fig, width="stretch")
        # for fig in boxplots:
        #     st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns or outliers detected.")

    st.markdown("### 6. Class Distribution")
    if imbalance_info.get("distributions"):
        target = imbalance_info.get("target_column")
        st.write(f"Target column for class distribution: **{target}**")
        dist_pie = class_distribution_pie(
            imbalance_info["distributions"], title="Class Distribution"
        )
        st.plotly_chart(dist_pie, use_container_width=True)
        if imbalance_info.get("severe_imbalance"):
            st.warning(
                "Severe class imbalance detected: one or more classes represent more than 80% of the data."
            )
    else:
        st.info("No categorical columns suitable for class imbalance analysis.")

    st.markdown("### 7. Correlation Heatmap")
    corr_matrix = correlation_info.get("correlation_matrix")
    if corr_matrix is not None and not corr_matrix.empty:
        heatmap_fig = correlation_heatmap(corr_matrix)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        high_pairs = correlation_info.get("high_correlation_pairs", [])
        if high_pairs:
            st.warning("Highly correlated feature pairs (|corr| > 0.9) detected.")
            st.table(
                {
                    "Feature A": [a for a, _, _ in high_pairs],
                    "Feature B": [b for _, b, _ in high_pairs],
                    "Correlation": [v for _, _, v in high_pairs],
                }
            )
        else:
            st.info("No highly correlated feature pairs detected (|corr| <= 0.9).")
    else:
        st.info("No numeric columns available for correlation analysis.")

    st.markdown("### 8. Health Score Indicator")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Dataset Health Score", f"{health['score']} / 100", help="\n".join(health["details"]))
    with col2:
        if health["color"] == "green":
            st.success(f"Overall data quality is {health['label']}.")
        elif health["color"] == "yellow":
            st.warning(f"Overall data quality is {health['label']}.")
        else:
            st.error(f"Overall data quality is {health['label']}.")

    st.markdown("### 9. AI Fix Suggestions")
    if suggestions:
        for s in suggestions:
            severity = s.get("severity", "info").capitalize()
            st.subheader(f"{s['issue']} ({severity})")
            st.write(s["explanation"])
            affected = s.get("affected_columns") or []
            if affected:
                st.write("**Affected columns:** " + ", ".join(map(str, affected)))
            fixes = s.get("suggested_fixes") or []
            if fixes:
                st.markdown("**Suggested fixes:**")
                for fix in fixes:
                    st.markdown(f"- {fix}")
    else:
        st.info("No major issues detected. Dataset appears healthy.")

    st.markdown("### 10. Auto Fix Dataset & Download Report")

    col_a, col_b = st.columns(2)
    cleaned_df = None

    with col_a:
        st.markdown("#### Auto Fix Dataset")
        if st.button("Auto Fix Dataset"):
            cleaned_df = analyzer.auto_fix_dataset(df, summary["numeric_columns"])
            st.success("Auto cleaning applied: duplicates removed, missing numeric values imputed, outliers capped.")
            st.dataframe(cleaned_df.head())
            cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Cleaned Dataset",
                data=cleaned_csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
            )

    with col_b:
        st.markdown("#### Download Analysis Report")
        html_report = report_generator.generate_html_report(analysis)
        st.download_button(
            "Download HTML Report",
            data=html_report.encode("utf-8"),
            file_name="dataset_report.html",
            mime="text/html",
        )

    # Additional distributions
    st.markdown("### Additional Data Distribution Visualizations")
    num_charts = numeric_distribution_charts(df, summary["numeric_columns"])
    cat_charts = categorical_distribution_charts(df, summary["categorical_columns"])

    if num_charts:
        st.markdown("#### Numeric Columns")
        for col, charts in num_charts.items():
            st.markdown(f"**{col}**")
            cols = st.columns(2)
            cols[0].plotly_chart(charts["hist"], width="stretch")
            cols[1].plotly_chart(charts["box"], width="stretch")
            # cols[2].plotly_chart(charts["density"], width="stretch")

    if cat_charts:
        st.markdown("#### Categorical Columns")
        cols = st.columns(3)
        
        for i,(col, fig) in enumerate( cat_charts.items()):
            with cols[i%3]:
                st.markdown(f"**{col}**")
                st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()

