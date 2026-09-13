"""
dataset_loader.py

Step 1 of the AI Data Analyst Agent project.

This module is intentionally simple and dependency-light. It only uses
pandas. No LLM calls happen here — this is pure, deterministic Python
code that:

1. Loads a dataset from disk (CSV or Excel).
2. Detects the "logical" type of each column (numerical, datetime,
   categorical, text, unknown) — separate from pandas' own dtype.
3. Profiles the dataset: shape, per-column stats, missing values,
   duplicates.

Later steps (analysis planner, tool-calling LLM, chart generation, etc.)
will be built on top of this foundation.
"""

from pathlib import Path
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the dataset file (.csv, .xlsx, or .xls).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. Supported: .csv, .xlsx, .xls"
        )

    if df.empty:
        raise ValueError(f"The file '{file_path}' was loaded but contains no rows.")

    return df


def detect_column_type(series: pd.Series) -> str:
    """
    Detect the logical type of a single column.

    This goes beyond the pandas dtype (e.g. 'object') and tries to answer:
    is this column really numerical, datetime, categorical, or free text?

    Returns one of: "numerical", "datetime", "categorical", "text", "unknown"
    """
    # Drop missing values before inspecting — they shouldn't affect type detection.
    non_null = series.dropna()

    if non_null.empty:
        return "unknown"

    # 1. Already a numeric dtype (int, float)
    if pd.api.types.is_numeric_dtype(non_null):
        return "numerical"

    # 2. Already a datetime dtype
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return "datetime"

    # 3. Try to parse as datetime (covers string dates like "2026-01-01")
    #    Only trust this if a large majority of values actually parse.
    parsed_dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
    if parsed_dates.notna().mean() > 0.9:
        return "datetime"

    # 4. Try to parse as numeric (covers numbers stored as strings, e.g. "1,000")
    parsed_numbers = pd.to_numeric(
        non_null.astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    if parsed_numbers.notna().mean() > 0.9:
        return "numerical"

    # 5. Categorical vs free text: base this on how many unique values there are
    #    relative to the number of rows. Few unique values -> categorical.
    unique_ratio = non_null.nunique() / len(non_null)
    if unique_ratio <= 0.5 or non_null.nunique() <= 20:
        return "categorical"

    return "text"


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Build a structured profile (summary) of the dataset.

    Returns a dictionary that is safe to convert to JSON, e.g. for
    later use by an LLM planner.
    """
    columns_profile = []

    for column_name in df.columns:
        series = df[column_name]
        columns_profile.append(
            {
                "name": column_name,
                "pandas_dtype": str(series.dtype),
                "detected_type": detect_column_type(series),
                "missing_values": int(series.isna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_profile": columns_profile,
    }

    return profile