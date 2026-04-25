from __future__ import annotations

import pandas as pd

from subenvs.autodatalab.analytics import (
    clean_orders,
    compute_kpis,
    compute_revenue_share,
    data_quality_score,
    derive_revenue,
    validate_schema,
)

from ..models import ExpertReport

_ANALYST_QUERY = (
    "data quality cleaning duplicates imputation category median "
    "KPIs total revenue top category data_quality_score"
)


class DataAnalystExpert:
    expert_id = "analyst"

    def run(
        self,
        task_name: str,
        question: str,
        raw_df: pd.DataFrame,
        focused: bool = False,
        use_rag: bool = False,
    ) -> ExpertReport:
        cleaned = clean_orders(raw_df)
        enriched = derive_revenue(cleaned.df)
        kpis = compute_kpis(cleaned.df)
        share = compute_revenue_share(cleaned.df)
        schema = validate_schema(enriched)
        top_row = share.iloc[0]
        summary = (
            f"Cleaned {len(raw_df)} raw rows into {len(cleaned.df)} trusted rows. "
            f"Top category is {top_row['Category']} and total revenue is {kpis['total_revenue']:.2f}."
        )
        bullets = [
            f"Removed {cleaned.duplicates_removed} duplicate rows and imputed {cleaned.imputed_prices} missing prices.",
            f"Data quality score is {data_quality_score(cleaned.df):.2f} with {schema['invalid_date_rows']} invalid date rows.",
            f"{top_row['Category']} leads revenue at {float(top_row['Revenue']):.2f}.",
        ]
        metrics = {
            "duplicates_removed": cleaned.duplicates_removed,
            "imputed_prices": cleaned.imputed_prices,
            "data_quality_score": data_quality_score(cleaned.df),
            "total_revenue": kpis["total_revenue"],
            "avg_order_value": kpis["avg_order_value"],
            "top_category": str(top_row["Category"]),
            "top_category_revenue": round(float(top_row["Revenue"]), 2),
        }
        issues = [f"risk:{name}" for name in schema["risk_flags"]]
        citations = share["Category"].astype(str).head(3).tolist()

        memory_citations: list[str] = []
        memory_snippets: list[str] = []
        if use_rag:
            from memory import get_retriever

            hits = get_retriever().query(_ANALYST_QUERY, k=2)
            memory_citations = [h.as_citation() for h in hits]
            memory_snippets = [h.snippet for h in hits]
            if hits:
                summary = summary + f" Grounded in SOP {hits[0].source.split('#')[0]}."
                bullets.append(
                    f"Grounded against SOP: {hits[0].source.split('#')[0]} (score {hits[0].score:.2f})."
                )

        return ExpertReport(
            expert_id="analyst",
            title="Data Analyst Report",
            summary=summary,
            metrics=metrics,
            bullet_points=bullets,
            issues=issues,
            citations=citations,
            memory_citations=memory_citations,
            memory_snippets=memory_snippets,
        )
