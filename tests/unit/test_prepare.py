import pandas as pd

from mlops_project.data.prepare import build_daily_sales


def test_build_daily_sales_filters_and_fills_missing_days() -> None:
    transactions = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "C3", "4"],
            "StockCode": ["A", "A", "A", "B"],
            "Quantity": [2, 3, 100, 1],
            "InvoiceDate": ["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-01"],
            "UnitPrice": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = build_daily_sales(transactions, top_items=1)

    assert result.to_dict("records") == [
        {"item_id": "A", "timestamp": pd.Timestamp("2024-01-01"), "sales": 2},
        {"item_id": "A", "timestamp": pd.Timestamp("2024-01-02"), "sales": 0},
        {"item_id": "A", "timestamp": pd.Timestamp("2024-01-03"), "sales": 3},
    ]
