"""
chart_tools.py

Step 34 of the AI Data Analyst Agent project.

find_column() is now a public function (renamed from _find_column) so
app.py can use it to sanity-check the planner's column/chart-type choice
before drawing anything - specifically, catching cases where the LLM
picks "bar" for two numerical columns (which should be a scatter plot
instead), regardless of how the prompt is worded.

Supports 6 chart types: bar, line, histogram, scatter, pie, heatmap
(correlation grid across all numeric columns).
"""

from pathlib import Path
import uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"

MAX_BAR_CATEGORIES = 12
MAX_LINE_POINTS = 60
MAX_PIE_CATEGORIES = 8
MAX_SCATTER_POINTS = 2000

COLOR_PALETTE = ["#2dd4bf", "#38bdf8", "#818cf8", "#f472b6", "#fb923c", "#a3e635"]
GRID_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"
MUTED_COLOR = "#8b949e"
BG_COLOR = "#0d1117"

ID_NAME_HINTS = ("id", "code", "number", "no", "index", "key")


def _apply_base_style(fig, ax):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def find_column(df: pd.DataFrame, requested):
    """
    Find the best matching column for a requested name, forgiving case
    differences and minor typos/partial names. Returns None if nothing
    reasonable is found. Public so app.py can use it for validation.
    """
    if not requested:
        return None
    if requested in df.columns:
        return requested
    lower_map = {col.lower(): col for col in df.columns}
    if requested.lower() in lower_map:
        return lower_map[requested.lower()]
    for col in df.columns:
        if requested.lower() in col.lower() or col.lower() in requested.lower():
            return col
    return None


def _looks_like_id_column(df: pd.DataFrame, col: str) -> bool:
    name_lower = col.lower()
    if any(hint in name_lower for hint in ID_NAME_HINTS):
        return True
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return False
    uniqueness_ratio = non_null.nunique() / len(non_null)
    return uniqueness_ratio > 0.9


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _pick_default_column(df: pd.DataFrame, kind: str):
    from tools.dataset_loader import detect_column_type
    candidates = []
    for col in df.columns:
        col_type = detect_column_type(df[col])
        if kind == "any" or col_type == kind:
            candidates.append(col)
    if not candidates:
        return df.columns[0] if len(df.columns) else None
    non_id_candidates = [c for c in candidates if not _looks_like_id_column(df, c)]
    return non_id_candidates[0] if non_id_candidates else candidates[0]


def _get_numeric_columns(df: pd.DataFrame, exclude_ids: bool = True) -> list:
    """Return all columns that are genuinely numeric (coercible), skipping ID-like ones."""
    from tools.dataset_loader import detect_column_type
    result = []
    for col in df.columns:
        if detect_column_type(df[col]) == "numerical":
            if exclude_ids and _looks_like_id_column(df, col):
                continue
            result.append(col)
    return result


def _limit_categories(grouped: pd.Series, max_categories: int) -> pd.Series:
    if len(grouped) <= max_categories:
        return grouped
    top = grouped.iloc[:max_categories]
    other_sum = grouped.iloc[max_categories:].sum()
    top = pd.concat([top, pd.Series({"Other": other_sum})])
    return top


def _resample_line_data(plot_df: pd.DataFrame, x_col: str, y_col: str, max_points: int = MAX_LINE_POINTS) -> pd.DataFrame:
    if len(plot_df) <= max_points:
        return plot_df.groupby(x_col, as_index=False)[y_col].sum()
    span_days = (plot_df[x_col].max() - plot_df[x_col].min()).days or 1
    bucket_days = max(1, span_days // max_points)
    plot_df = plot_df.set_index(x_col)
    resampled = plot_df[y_col].resample(f"{bucket_days}D").sum().reset_index()
    return resampled


def create_chart(
    df: pd.DataFrame,
    chart_type: str = "bar",
    x_column=None,
    y_column=None,
    title=None,
    filename=None,
) -> dict:
    """
    Create a chart from the dataset and save it as a PNG file.

    Supported chart_type values: "bar", "line", "histogram", "scatter",
    "pie", "heatmap". Heatmap ignores x_column/y_column and instead
    uses every numeric column in the dataset (excluding ID-like ones)
    to build a correlation grid.

    Returns
    -------
    dict
        {"path": str, "chart_type": str, "x_column": str or None, "y_column": str or None}
    """
    valid_types = ("bar", "line", "histogram", "scatter", "pie", "heatmap")
    if chart_type not in valid_types:
        chart_type = "bar"

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Heatmap is a special case: no single x/y, uses all numeric columns ---
    if chart_type == "heatmap":
        numeric_cols = _get_numeric_columns(df)
        if len(numeric_cols) < 2:
            raise ValueError("Need at least two numeric columns to build a correlation heatmap.")

        numeric_df = df[numeric_cols].apply(_coerce_numeric)
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(7 + 0.4 * len(numeric_cols), 6 + 0.4 * len(numeric_cols)), dpi=140)
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=40, ha="right", color=TEXT_COLOR, fontsize=9)
        ax.set_yticklabels(numeric_cols, color=TEXT_COLOR, fontsize=9)

        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                value = corr.values[i, j]
                text_color = "white" if abs(value) > 0.5 else TEXT_COLOR
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=MUTED_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=MUTED_COLOR)

        ax.set_title(title or "Correlation Heatmap", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=14)
        fig.tight_layout()

        unique_suffix = uuid.uuid4().hex[:8]
        output_name = filename or "heatmap_correlation"
        output_path = CHARTS_DIR / f"{output_name}_{unique_suffix}.png"
        fig.savefig(output_path, dpi=140, facecolor=BG_COLOR)
        plt.close(fig)

        return {"path": str(output_path), "chart_type": "heatmap", "x_column": None, "y_column": None}

    # --- All other chart types resolve x/y columns as before ---
    resolved_x = find_column(df, x_column) or _pick_default_column(
        df, "categorical" if chart_type in ("bar", "pie") else ("datetime" if chart_type == "line" else "numerical")
    )
    resolved_y = find_column(df, y_column) or _pick_default_column(df, "numerical")

    if chart_type == "histogram":
        target = resolved_y or resolved_x
        if target and _looks_like_id_column(df, target):
            better = _pick_default_column(df, "numerical")
            if better:
                resolved_y = better if resolved_y else None
                resolved_x = better if not resolved_y else resolved_x

    if resolved_x is None:
        raise ValueError("Could not find any usable column for the x-axis.")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    _apply_base_style(fig, ax)

    if chart_type == "bar":
        if resolved_y is None:
            raise ValueError("Could not find a numerical column to plot for a bar chart.")
        y_values = _coerce_numeric(df[resolved_y])
        grouped = y_values.groupby(df[resolved_x]).sum().sort_values(ascending=False)
        grouped = _limit_categories(grouped, MAX_BAR_CATEGORIES)

        colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(grouped))]
        bars = ax.bar(grouped.index.astype(str), grouped.values, color=colors, width=0.65, edgecolor="none")
        max_val = grouped.values.max()
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + max_val * 0.015,
                     f"{height:,.0f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)
        ax.set_xlabel(resolved_x)
        ax.set_ylabel(resolved_y)
        plt.xticks(rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    elif chart_type == "line":
        if resolved_y is None:
            raise ValueError("Could not find a numerical column to plot for a line chart.")
        plot_df = df[[resolved_x, resolved_y]].copy()
        plot_df[resolved_y] = _coerce_numeric(plot_df[resolved_y])
        parsed_x = pd.to_datetime(plot_df[resolved_x], errors="coerce", format="mixed")
        if parsed_x.notna().mean() > 0.9:
            plot_df[resolved_x] = parsed_x
            plot_df = plot_df.dropna(subset=[resolved_x]).sort_values(resolved_x)
            plot_df = _resample_line_data(plot_df, resolved_x, resolved_y)
        else:
            plot_df = plot_df.groupby(resolved_x, as_index=False)[resolved_y].sum()
        ax.plot(plot_df[resolved_x], plot_df[resolved_y], marker="o", markersize=4, linewidth=2, color=COLOR_PALETTE[0])
        ax.fill_between(plot_df[resolved_x], plot_df[resolved_y], color=COLOR_PALETTE[0], alpha=0.1)
        ax.set_xlabel(resolved_x)
        ax.set_ylabel(resolved_y)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig.autofmt_xdate()

    elif chart_type == "histogram":
        target_col = resolved_y or resolved_x
        values = _coerce_numeric(df[target_col]).dropna()
        ax.hist(values, bins=min(20, max(5, len(values) // 5 or 5)), color=COLOR_PALETTE[3], edgecolor=BG_COLOR)
        ax.set_xlabel(target_col)
        ax.set_ylabel("Frequency")
        resolved_x = target_col
        resolved_y = None

    elif chart_type == "scatter":
        if resolved_y is None:
            raise ValueError("Could not find a second numerical column to plot for a scatter chart.")
        x_vals = _coerce_numeric(df[resolved_x])
        y_vals = _coerce_numeric(df[resolved_y])
        plot_df = pd.DataFrame({resolved_x: x_vals, resolved_y: y_vals}).dropna()
        if len(plot_df) > MAX_SCATTER_POINTS:
            plot_df = plot_df.sample(MAX_SCATTER_POINTS, random_state=42)
        ax.scatter(plot_df[resolved_x], plot_df[resolved_y], color=COLOR_PALETTE[1], alpha=0.5, s=18, edgecolors="none")
        ax.set_xlabel(resolved_x)
        ax.set_ylabel(resolved_y)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    elif chart_type == "pie":
        if resolved_y is None:
            raise ValueError("Could not find a numerical column to plot for a pie chart.")
        y_values = _coerce_numeric(df[resolved_y])
        grouped = y_values.groupby(df[resolved_x]).sum().sort_values(ascending=False)
        grouped = _limit_categories(grouped, MAX_PIE_CATEGORIES)
        colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(grouped))]
        wedges, _, autotexts = ax.pie(
            grouped.values, labels=grouped.index.astype(str), autopct="%1.1f%%",
            colors=colors, textprops={"color": TEXT_COLOR, "fontsize": 9},
        )
        for autotext in autotexts:
            autotext.set_color(BG_COLOR)
            autotext.set_fontweight("bold")
        ax.set_facecolor(BG_COLOR)

    if chart_type != "pie":
        ax.set_title(
            title or (f"{resolved_y} by {resolved_x}" if resolved_y and chart_type != "histogram" else f"Distribution of {resolved_x}"),
            fontsize=13, fontweight="bold", pad=14,
        )
    else:
        ax.set_title(title or f"{resolved_y} share by {resolved_x}", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=14)

    fig.tight_layout()

    unique_suffix = uuid.uuid4().hex[:8]
    output_name = filename or f"{chart_type}_{resolved_x}" + (
        f"_{resolved_y}" if resolved_y and chart_type != "histogram" else ""
    )
    output_path = CHARTS_DIR / f"{output_name}_{unique_suffix}.png"
    fig.savefig(output_path, dpi=140, facecolor=BG_COLOR)
    plt.close(fig)

    return {
        "path": str(output_path),
        "chart_type": chart_type,
        "x_column": resolved_x,
        "y_column": resolved_y,
    }