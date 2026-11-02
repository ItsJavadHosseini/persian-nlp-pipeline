from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from digikala_review.data.auditor import (
    DataProfiler,
    DataQualityAuditor,
    TargetAuditor,
    DuplicateAuditor,
    DistributionAuditor,
    RelationshipAuditor,
    load_audit_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "data_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data audit for the Digikala review dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("DIGIKALA_DATA_PATH", str(DEFAULT_INPUT))),
        help="Directory containing digikala-comments.csv and digikala-products.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where audit reports will be written.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, pl.LazyFrame):
        return value.collect(engine="streaming").to_dicts()
    if isinstance(value, pl.DataFrame):
        return value.to_dicts()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(data: Any, path: Path) -> None:
    path.write_text(
        json.dumps(
            to_jsonable(data),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def save_csv(data: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
    if isinstance(data, pl.LazyFrame):
        data = data.collect(engine="streaming")
    if not isinstance(data, pl.DataFrame):
        raise TypeError(f"Expected Polars DataFrame, got {type(data).__name__}")
    data.write_csv(path)


def flatten_dict(data: dict[str, Any], prefix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_dict(value, name))
        elif isinstance(value, pl.DataFrame):
            rows.append({"metric": name, "value": json.dumps(
                to_jsonable(value), ensure_ascii=False, default=str
            )})
        else:
            rows.append({"metric": name, "value": str(value)})
    return rows


def build_column_profile(comments: pl.LazyFrame) -> pl.DataFrame:
    schema = comments.collect_schema()
    names = schema.names()
    expressions: list[pl.Expr] = [pl.len().alias("__rows")]
    for index, name in enumerate(names):
        col = pl.col(name)
        if schema[name] == pl.String:
            normalized = col.str.strip_chars().str.to_lowercase()
            missing = (
                col.is_null()
                | normalized.is_in(["", "nan", "null", "none", "n/a"]).fill_null(False)
            )
        else:
            missing = col.is_null()
        expressions.extend([
            missing.sum().alias(f"missing_{index}"),
            col.n_unique().alias(f"unique_{index}"),
        ])
    stats = comments.select(expressions).collect(engine="streaming")
    row_count = int(stats["__rows"][0])
    rows = []
    for index, name in enumerate(names):
        missing = int(stats[f"missing_{index}"][0] or 0)
        rows.append({
            "column": name,
            "dtype": str(schema[name]),
            "missing_count": missing,
            "missing_percentage": round(missing / row_count * 100, 4) if row_count else 0,
            "unique_count": int(stats[f"unique_{index}"][0] or 0),
        })
    return pl.DataFrame(rows)


def build_label_distribution(target: TargetAuditor) -> pl.DataFrame:
    recommendation = target.analyze_recommendation().with_columns(
        pl.lit("recommendation").alias("target")
    )
    rating = target.analyze_rating_band().with_columns(
        pl.lit("rating_band").alias("target")
    )
    return pl.concat([recommendation, rating], how="diagonal_relaxed")


def write_report(
    output_dir: Path,
    overview: dict[str, Any],
    row_count: int,
    execution_time: str,
) -> None:
    report = f"""# Digikala Dataset Audit Report

## 1. Dataset overview

- Comments rows: {row_count:,}
- Audit execution time (UTC): {execution_time}

## 2. Dataset metadata

```json
{json.dumps(to_jsonable(overview), ensure_ascii=False, indent=2, default=str)}
```

## 3. Generated files

- `schema.csv`
- `column_profile.csv`
- `missing_values.csv`
- `data_quality_details.json`
- `label_distribution.csv`
- `rating_analysis.json`
- `target_comparison.csv`
- `duplicate_summary.csv`
- `duplicate_conflicting_labels.csv`
- `text_length_summary.csv`
- `short_long_reviews.csv`
- `text_length_by_class.csv`
- `product_concentration.csv`
- `product_concentration_details.json`
- `seller_distribution.csv`
- `relationship_analysis.json`
- `unmatched_product_ids.csv`
- `audit_metadata.json`

## 4. Interpretation notes

- This audit does not modify the raw CSV files.
- String placeholders such as `nan`, `null`, `none`, and `n/a` are treated as missing for audit calculations.
- `recommendation_status` is normalized for reporting; unexpected non-missing labels remain visible.
- `rating_band` uses: low [0, 2.5), medium [2.5, 4.0), high [4.0, 5.0].
- Values outside the rating range are not assigned a rating band.
- Audit findings are descriptive and do not establish causality.
- Data cleaning, feature engineering, split construction and model fitting belong to later project stages.
"""
    (output_dir / "data_audit_report.md").write_text(report, encoding="utf-8")


def run_audit(input_dir: Path, output_dir: Path) -> None:
    comments_path = input_dir / "digikala-comments.csv"
    products_path = input_dir / "digikala-products.csv"
    if not comments_path.is_file():
        raise FileNotFoundError(f"Comments CSV not found: {comments_path}")
    if not products_path.is_file():
        raise FileNotFoundError(f"Products CSV not found: {products_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    execution_time = datetime.now(timezone.utc).isoformat()

    print("Loading datasets...")
    comments, products = load_audit_frames(comments_path, products_path)

    print("1/7 Profiling datasets...")
    profiler = DataProfiler(
        comments=comments,
        products=products,
        comments_path=comments_path,
        products_path=products_path,
    )
    overview = profiler.get_overview()
    row_count = profiler.get_row_count()
    save_csv(profiler.get_schema(), output_dir / "schema.csv")
    save_csv(build_column_profile(comments), output_dir / "column_profile.csv")
    save_csv(profiler.get_unique_counts(), output_dir / "unique_counts.csv")

    print("2/7 Checking data quality...")
    quality = DataQualityAuditor(comments)
    save_csv(quality.check_missing(), output_dir / "missing_values.csv")
    save_json({
        "ranges": quality.check_ranges(),
        "categories": quality.check_categories(),
        "missing_label_patterns": quality.analyze_missing_label_patterns(),
    }, output_dir / "data_quality_details.json")

    print("3/7 Analyzing targets...")
    target = TargetAuditor(comments)
    save_csv(build_label_distribution(target), output_dir / "label_distribution.csv")
    save_json(target.analyze_rating(), output_dir / "rating_analysis.json")
    save_csv(target.compare_targets(), output_dir / "target_comparison.csv")

    print("4/7 Auditing duplicates...")
    duplicate = DuplicateAuditor(comments)
    duplicate_results = duplicate.run_full_audit()
    summary = {
        key: value for key, value in duplicate_results.items()
        if not isinstance(value, pl.DataFrame)
    }
    save_csv(pl.DataFrame(flatten_dict(summary)), output_dir / "duplicate_summary.csv")
    conflicting_labels = duplicate_results["conflicting_labels"]

    if "labels" in conflicting_labels.columns:
        conflicting_labels = conflicting_labels.with_columns(
            pl.col("labels").list.join(" | ").alias("labels")
        )

    save_csv(
        conflicting_labels,
        output_dir / "duplicate_conflicting_labels.csv",
    )
    print("5/7 Analyzing text and distributions...")
    distribution = DistributionAuditor(comments)
    save_csv(distribution.analyze_text_length(), output_dir / "text_length_summary.csv")
    save_csv(distribution.analyze_short_long_reviews(), output_dir / "short_long_reviews.csv")
    save_csv(distribution.analyze_text_length_by_class(), output_dir / "text_length_by_class.csv")
    save_csv(distribution.product_review_counts(), output_dir / "product_concentration.csv")
    save_json(
        distribution.analyze_product_concentration(),
        output_dir / "product_concentration_details.json",
    )
    save_csv(distribution.analyze_seller_distribution(), output_dir / "seller_distribution.csv")

    print("6/7 Checking comments-products relationship...")
    relationship = RelationshipAuditor(comments, products)
    save_json({
        "product_id_uniqueness": relationship.check_product_id_uniqueness(),
        "join_coverage": relationship.analyze_join_coverage(),
    }, output_dir / "relationship_analysis.json")
    save_csv(relationship.get_unmatched_product_ids(), output_dir / "unmatched_product_ids.csv")

    print("7/7 Writing metadata and report...")
    metadata = {
        "execution_time_utc": execution_time,
        "comments_path": comments_path,
        "products_path": products_path,
        "comments_file_size_bytes": comments_path.stat().st_size,
        "products_file_size_bytes": products_path.stat().st_size,
        "comments_rows": row_count,
        "products_rows": overview["products_row_count"],
        "polars_version": pl.__version__,
        "output_directory": output_dir,
        "missing_string_placeholders": ["", "nan", "null", "none", "n/a"],
    }
    save_json(metadata, output_dir / "audit_metadata.json")
    write_report(output_dir, overview, row_count, execution_time)

    print("\nData audit completed.")
    print(f"Comments rows: {row_count:,}")
    print(f"Output directory: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    run_audit(args.input, args.output)


if __name__ == "__main__":
    main()