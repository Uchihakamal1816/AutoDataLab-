from __future__ import annotations

from ..models import ExpertReport


class StrategyExpert:
    expert_id = "strategy"

    def run(self, task_name: str, task_meta: dict, analyst_report: ExpertReport, finance_report: ExpertReport, focused: bool = False) -> ExpertReport:
        top_category = str(analyst_report.metrics.get('top_category', 'the best category'))
        total_revenue = analyst_report.metrics.get('total_revenue', 0.0)
        projection = finance_report.metrics.get('projection_next_quarter', 0.0)
        variance_pct = finance_report.metrics.get('variance_pct', 0.0)
        bullets = [
            f"Prioritize {top_category} with a targeted campaign to defend {total_revenue:.2f} in current revenue.",
            f"Use the projection of {projection:.2f} to set weekly operating targets and tighten execution if variance reaches {variance_pct:.2f}%.",
            f"Protect margin by linking staffing and promotions to the break-even threshold and projection outlook.",
        ]
        return ExpertReport(
            expert_id='strategy',
            title='Strategy Recommendations',
            summary='Converted analyst and finance findings into a 3-bullet operating plan.',
            metrics={'recommendation_count': len(bullets)},
            bullet_points=bullets,
            citations=[str(analyst_report.metrics.get('top_category', '')), 'projection', 'variance'],
        )
