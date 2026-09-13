"""
test_loader.py

Simple manual test script for dataset_loader.py.
Not using pytest yet — keeping this step simple. We can add proper
unit tests later once the project structure stabilizes.
"""

import sys
from pathlib import Path


# Allow running this file directly (python tools/test_loader.py) by adding
# the project root to the path, so "from tools.dataset_loader import ..." works
# even if you're not running it as a package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tools.dataset_loader import load_dataset, profile_dataset
from tools.analysis_tools import analyze_dataset
from tools.chart_tools import create_chart


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "sales.csv"

    print(f"Loading dataset from: {data_path}\n")
    df = load_dataset(str(data_path))

    print("First 5 rows:")
    print(df.head())
    print()

    profile = profile_dataset(df)

    print(f"Rows: {profile['rows']}")
    print(f"Columns: {profile['columns']}")
    print(f"Duplicate rows: {profile['duplicate_rows']}")
    print()
    print("Column profile:")
    for col in profile["columns_profile"]:
        print(
            f"  - {col['name']:<10} "
            f"dtype={col['pandas_dtype']:<10} "
            f"detected={col['detected_type']:<12} "
            f"missing={col['missing_values']:<3} "
            f"unique={col['unique_values']}"
        )

    print()
    print("Column analysis (analyze_dataset):")
    analysis = analyze_dataset(df)
    for column_name, result in analysis.items():
        print(f"  - {column_name} ({result['type']}):")
        print(f"      {result['stats']}")

        print()
    print("Creating charts (create_chart):")
    chart1 = create_chart(df, chart_type="bar", x_column="product", y_column="revenue", title="Total Revenue by Product")
    print(f"  - Saved: {chart1}")

    chart2 = create_chart(df, chart_type="line", x_column="date", y_column="revenue", title="Revenue Over Time")
    print(f"  - Saved: {chart2}")

    chart3 = create_chart(df, chart_type="histogram", x_column="revenue", title="Revenue Distribution")
    print(f"  - Saved: {chart3}")


if __name__ == "__main__":
    main()