"""Prepare UCI Online Retail transactions for demand forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/raw/Online Retail.xlsx")
DEFAULT_OUTPUT = Path("data/processed/daily_sales.csv")


def build_daily_sales(transactions: pd.DataFrame, top_items: int = 100) -> pd.DataFrame:
    """Return a regular daily sales panel for the highest-volume products."""
    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if top_items < 1:
        raise ValueError("top_items must be at least 1")

    clean = transactions.loc[:, list(required)].copy()
    clean["InvoiceNo"] = clean["InvoiceNo"].astype("string")
    clean["StockCode"] = clean["StockCode"].astype("string")
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"], errors="coerce")
    clean = clean[
        clean["InvoiceDate"].notna()
        & clean["StockCode"].notna()
        & ~clean["InvoiceNo"].str.upper().str.startswith("C", na=False)
        & (clean["Quantity"] > 0)
        & (clean["UnitPrice"] > 0)
    ]

    popular_items = (
        clean.groupby("StockCode", observed=True)["Quantity"]
        .sum()
        .nlargest(top_items)
        .index
    )
    clean = clean[clean["StockCode"].isin(popular_items)].copy()
    clean["timestamp"] = clean["InvoiceDate"].dt.normalize()

    daily = clean.groupby(["StockCode", "timestamp"], as_index=False)["Quantity"].sum()
    daily = daily.rename(columns={"StockCode": "item_id", "Quantity": "sales"})

    dates = pd.date_range(daily["timestamp"].min(), daily["timestamp"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [popular_items.astype(str), dates], names=["item_id", "timestamp"]
    )
    return (
        daily.assign(item_id=daily["item_id"].astype(str))
        .set_index(["item_id", "timestamp"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["item_id", "timestamp"], ignore_index=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-items", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Raw dataset not found: {args.input}")

    transactions = pd.read_excel(args.input, engine="openpyxl")
    daily_sales = build_daily_sales(transactions, top_items=args.top_items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily_sales.to_csv(args.output, index=False)
    print(f"Wrote {len(daily_sales):,} rows to {args.output}")


if __name__ == "__main__":
    main()
