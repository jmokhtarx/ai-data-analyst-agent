"""
analysis_tools.py

Step 2 of the AI Data Analyst Agent project.

This module adds the first real "analysis" tool: analyze_dataset().
It is pure pandas — no LLM calls. It relies on detect_column_type()
from dataset_loader.py so we don't duplicate that logic.

The idea going forward:
    LLM decides "I need column stats" -> calls analyze_dataset()
    -> gets back facts -> explains them in natural language.
This file is the "Python/Pandas performs the calculation" layer.
"""

import pandas as pd

from tools.dataset_loader import detect_column_type


def _analyze_numerical(series: pd.Series) -> dict:
    """
    Descriptive stats for a numerical column.

    Handles columns that are detected as numerical but are still stored
    as strings (e.g. "1,000" or "$500") by cleaning and coercing them
    to actual numbers first, instead of assuming the dtype is already numeric.
    """
    non_null = series.dropna()

    if not pd.api.types.is_numeric_dtype(non_null):
        cleaned = (
            non_null.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        non_null = pd.to_numeric(cleaned, errors="coerce").dropna()

    if non_null.empty:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None, "sum": None}

    return {
        "mean": round(float(non_null.mean()), 2),
        "median": round(float(non_null.median()), 2),
        "min": round(float(non_null.min()), 2),
        "max": round(float(non_null.max()), 2),
        "std": round(float(non_null.std()), 2) if len(non_null) > 1 else 0.0,
        "sum": round(float(non_null.sum()), 2),
    }


def _analyze_categorical(series: pd.Series, top_n: int = 5) -> dict:
    """Value counts and mode for a categorical column."""
    non_null = series.dropna()
    value_counts = non_null.value_counts().head(top_n)
    return {
        "mode": value_counts.index[0] if not value_counts.empty else None,
        "top_values": {str(k): int(v) for k, v in value_counts.items()},
    }


def _analyze_datetime(series: pd.Series) -> dict:
    """Min, max, and range for a datetime column."""
    non_null = pd.to_datetime(series.dropna(), errors="coerce", format="mixed")
    non_null = non_null.dropna()

    if non_null.empty:
        return {"min_date": None, "max_date": None, "range_days": None}

    min_date = non_null.min()
    max_date = non_null.max()
    return {
        "min_date": str(min_date.date()),
        "max_date": str(max_date.date()),
        "range_days": int((max_date - min_date).days),
    }


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics for every column in the dataset,
    using the appropriate analysis based on each column's detected type.

    Returns a JSON-safe dictionary shaped like:
    {
        "quantity": {"type": "numerical", "stats": {...}},
        "region":   {"type": "categorical", "stats": {...}},
        "date":     {"type": "datetime", "stats": {...}},
        "notes":    {"type": "text", "stats": None},
    }
    """
    results = {}

    for column_name in df.columns:
        series = df[column_name]
        column_type = detect_column_type(series)

        if column_type == "numerical":
            stats = _analyze_numerical(series)
        elif column_type == "categorical":
            stats = _analyze_categorical(series)
        elif column_type == "datetime":
            stats = _analyze_datetime(series)
        else:
            # text / unknown: no detailed stats yet
            stats = None

        results[column_name] = {
            "type": column_type,
            "stats": stats,
        }

    return results