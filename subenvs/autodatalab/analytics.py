from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class CleaningSummary:
    df: pd.DataFrame
    duplicates_removed: int
    imputed_prices: int


def _rounded_mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").dropna().mean()
    if pd.isna(value):
        return 0.0
    return round(float(value), 2)


def clean_orders(df: pd.DataFrame) -> CleaningSummary:
    working = df.copy()
    before = len(working)
    working = working.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(working)
    prices = pd.to_numeric(working["Price"], errors="coerce")
    imputed_prices = int(prices.isna().sum())
    if imputed_prices:
        fill = _rounded_mean(prices)
        working["Price"] = prices.fillna(fill).round(2)
    else:
        working["Price"] = prices.round(2)
    return CleaningSummary(df=working, duplicates_removed=duplicates_removed, imputed_prices=imputed_prices)


def derive_revenue(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["Price"] = pd.to_numeric(working["Price"], errors="coerce").round(2)
    working["Quantity"] = pd.to_numeric(working["Quantity"], errors="coerce")
    working["Revenue"] = (working["Price"] * working["Quantity"]).round(2)
    return working


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    working = derive_revenue(df)
    total_revenue = round(float(working["Revenue"].sum()), 2)
    order_count = int(working["OrderID"].nunique()) if "OrderID" in working.columns else len(working)
    avg_order_value = round(total_revenue / order_count, 2) if order_count else 0.0
    return {
        "total_revenue": total_revenue,
        "avg_order_value": avg_order_value,
        "order_count": float(order_count),
    }


def compute_revenue_share(df: pd.DataFrame) -> pd.DataFrame:
    working = derive_revenue(df)
    grouped = (
        working.groupby("Category", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["Revenue"].sum())
    grouped["RevenueShare"] = grouped["Revenue"].map(lambda v: round((float(v) / total) * 100, 2) if total else 0.0)
    return grouped


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    working = derive_revenue(df)
    dates = pd.to_datetime(working["OrderDate"], errors="coerce")
    working = working.loc[dates.notna()].copy()
    working["OrderDate"] = dates.loc[dates.notna()]
    working["Month"] = working["OrderDate"].dt.to_period("M").astype(str)
    monthly = working.groupby("Month", as_index=False)["Revenue"].sum()
    return monthly.sort_values("Month").reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    working = df.copy()
    prices = pd.to_numeric(working.get("Price"), errors="coerce")
    qty = pd.to_numeric(working.get("Quantity"), errors="coerce")
    dates = pd.to_datetime(working.get("OrderDate"), errors="coerce")
    invalid_price_rows = int((prices.isna() | (prices <= 0)).sum())
    invalid_quantity_rows = int((qty.isna() | (qty < 1)).sum())
    invalid_date_rows = int(dates.isna().sum())
    risk_flags = []
    if invalid_price_rows:
        risk_flags.append("price")
    if invalid_quantity_rows:
        risk_flags.append("quantity")
    if invalid_date_rows:
        risk_flags.append("order_date")
    return {
        "invalid_price_rows": invalid_price_rows,
        "invalid_quantity_rows": invalid_quantity_rows,
        "invalid_date_rows": invalid_date_rows,
        "risk_flags": risk_flags,
    }


def data_quality_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    checks = pd.Series([True] * len(df), index=df.index)
    prices = pd.to_numeric(df.get("Price"), errors="coerce")
    qty = pd.to_numeric(df.get("Quantity"), errors="coerce")
    dates = pd.to_datetime(df.get("OrderDate"), errors="coerce")
    checks &= prices.notna() & (prices > 0)
    checks &= qty.notna() & (qty >= 1)
    checks &= dates.notna()
    return round(float(checks.mean()), 4)
