from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def missing_values_bar_chart(missing_report: pd.DataFrame) -> go.Figure:
    if missing_report.empty:
        return go.Figure()
    fig = px.bar(
        missing_report,
        x="column",
        y="missing_percentage",
        title="Missing Values Percentage per Column",
    )
    fig.update_layout(xaxis_title="Column", yaxis_title="Missing %")
    return fig


def duplicate_pie_chart(total: int, non_duplicates: int) -> go.Figure:
    labels = ["Duplicate Rows", "Unique Rows"]
    values = [total, non_duplicates]
    fig = px.pie(
        names=labels,
        values=values,
        title="Duplicate vs Unique Rows",
    )
    return fig


def outlier_boxplots(df: pd.DataFrame, numeric_columns: List[str]) -> List[go.Figure]:
    figures: List[go.Figure] = []

    for col in numeric_columns:
        fig = px.box(
            df,
            y=col,
            title=f"Boxplot for {col}",
            color_discrete_sequence=["#4C78A8"]
        )

        figures.append(fig)

    return figures


def numeric_distribution_charts(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, go.Figure]:
    figs: Dict[str, go.Figure] = {}

    for col in numeric_columns:
        hist = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Histogram of {col}",
            color_discrete_sequence=["#80BF12"]  # blue
        )

        # Add border to bars
        hist.update_traces(
            marker=dict(
                line=dict(color="#ADF230", width=1)
            )
        )

        box = px.box(
            df,
            y=col,
            title=f"Boxplot of {col}",
            color_discrete_sequence=["#38D9A6"]  # same blue
        )

        figs[col] = {"hist": hist, "box": box}

    return figs


def categorical_distribution_charts(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, go.Figure]:
    figs: Dict[str, go.Figure] = {}

    for col in categorical_columns:
        vc = df[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, "count"]

        fig = px.bar(
            vc,
            x=col,
            y="count",
            title=f"Category Distribution for {col}",
            color_discrete_sequence=["#F58518"]  # orange
        )

        fig.update_traces(
            marker=dict(
                line=dict(color="#B0276D", width=1)
            )
        )

        figs[col] = fig

    return figs


def class_distribution_pie(distributions: Dict[str, Dict[str, float]], title: str) -> go.Figure:
    if not distributions:
        return go.Figure()
    labels = list(distributions.keys())
    values = [v["count"] for v in distributions.values()]
    fig = px.pie(names=labels, values=values, title=title)
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    if corr_matrix.empty:
        return go.Figure()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
        title="Correlation Heatmap",
    )
    return fig

