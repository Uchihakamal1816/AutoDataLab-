from __future__ import annotations

import pandas as pd

from subenvs.autodatalab.analytics import clean_orders, monthly_revenue

from ..models import ExpertReport
from .finance_tools import break_even, compute_variance, project_next_quarter


class FinanceExpert:
    expert_id = "finance"

    def run(self, task_name: str, question: str, raw_df: pd.DataFrame, analyst_metrics: dict, task_meta: dict, focused: bool = False) -> ExpertReport:
        cleaned = clean_orders(raw_df).df
        monthly = monthly_revenue(cleaned)
        projection = project_next_quarter(monthly)
        variance = compute_variance(float(analyst_metrics.get("total_revenue", 0.0)), float(task_meta.get("plan_value", 1.0)))
        be = break_even(float(task_meta.get("fixed_cost", 10000.0)), float(task_meta.get("unit_margin", 500.0)))
        metrics = {
            **projection,
            **variance,
            **be,
        }
        bullets = [
            f"Next-quarter revenue projection is {metrics['projection_next_quarter']:.2f} with +/- {metrics['confidence_band']:.2f} band.",
            f"Variance versus plan is {metrics['variance_abs']:.2f} ({metrics['variance_pct']:.2f}%).",
            f"Break-even sits at {metrics['break_even_units']:.2f} units.",
        ]
        return ExpertReport(
            expert_id="finance",
            title="Finance Forecast",
            summary=(
                f"Finance projects next quarter at {metrics['projection_next_quarter']:.2f} and "
                f"marks performance as {metrics['variance_flag']} plan."
            ),
            metrics=metrics,
            bullet_points=bullets,
            citations=list(monthly['Month'].astype(str).tail(3)),
        )
