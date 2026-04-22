from __future__ import annotations

from ..models import ExpertReport
from subenvs.email.hr_tools import build_hr_memo, score_memo


class HRExpert:
    expert_id = "hr"

    def run(self, task_name: str, task_meta: dict, analyst_report: ExpertReport | None, finance_report: ExpertReport | None, strategy_report: ExpertReport | None = None, focused: bool = False) -> ExpertReport:
        highlights = []
        if analyst_report:
            highlights.extend(analyst_report.bullet_points[:2])
        if finance_report:
            highlights.extend(finance_report.bullet_points[:2])
        if strategy_report:
            highlights.extend(strategy_report.bullet_points[:2])
        memo = build_hr_memo(str(task_meta.get('memo_audience', 'team')), str(task_meta.get('title', task_name)), highlights)
        score = score_memo(memo, task_meta.get('hr_required_terms', []))
        return ExpertReport(
            expert_id='hr',
            title='HR / Communications Memo',
            summary='Prepared the internal communication for stakeholders.',
            metrics={'memo_score': score},
            bullet_points=['Internal memo drafted and checked for professional tone.'],
            memo=memo,
            score=score,
        )
