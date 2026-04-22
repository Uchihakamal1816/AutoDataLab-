from __future__ import annotations

import numpy as np
import pandas as pd


def project_next_quarter(revenue_by_month: pd.DataFrame) -> dict[str, float]:
    if revenue_by_month.empty:
        return {"projection_next_quarter": 0.0, "confidence_band": 0.0}
    y = revenue_by_month["Revenue"].astype(float).to_numpy()
    x = np.arange(1, len(y) + 1, dtype=float)
    if len(y) == 1:
        next_values = np.array([y[-1], y[-1], y[-1]])
        residual = 0.0
    else:
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(y) + 1, len(y) + 4, dtype=float)
        next_values = slope * future_x + intercept
        predicted = slope * x + intercept
        residual = float(np.std(y - predicted))
    return {
        "projection_next_quarter": round(float(next_values.sum()), 2),
        "confidence_band": round(residual * 3, 2),
    }


def compute_variance(actual: float, plan: float) -> dict[str, float | str]:
    variance_abs = round(float(actual) - float(plan), 2)
    variance_pct = round((variance_abs / float(plan)) * 100, 2) if plan else 0.0
    flag = "ahead" if variance_abs >= 0 else "behind"
    return {
        "variance_abs": variance_abs,
        "variance_pct": variance_pct,
        "variance_flag": flag,
    }


def break_even(fixed_cost: float, unit_margin: float) -> dict[str, float]:
    if unit_margin <= 0:
        return {"break_even_units": 0.0, "break_even_revenue": 0.0}
    units = float(fixed_cost) / float(unit_margin)
    return {
        "break_even_units": round(units, 2),
        "break_even_revenue": round(units * float(unit_margin), 2),
    }
