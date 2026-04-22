from __future__ import annotations

from ..models import ExpertReport
from subenvs.email.hr_tools import build_hr_memo, score_memo
from subenvs.email.graders import grade_response


# Maps CoS brief tasks to the Round 1 email-env task IDs so the right
# task-specific rubric fires inside grade_response.
_BRIEF_TO_EMAIL_TASK = {
    "easy_brief": "task_1",
    "medium_brief": "task_2",
    "hard_brief": "task_3",
}


def _clamp(score: float) -> float:
    """Match ceo_brief_env.graders._clamp so every emitted score is in (0.001, 0.999)."""
    return max(0.001, min(0.999, round(float(score), 4)))


class HRExpert:
    expert_id = "hr"

    def run(
        self,
        task_name: str,
        task_meta: dict,
        analyst_report: ExpertReport | None,
        finance_report: ExpertReport | None,
        strategy_report: ExpertReport | None = None,
        focused: bool = False,
    ) -> ExpertReport:
        # --- 1. Build the memo from upstream reports -----------------------
        highlights: list[str] = []
        if analyst_report:
            highlights.extend(analyst_report.bullet_points[:2])
        if finance_report:
            highlights.extend(finance_report.bullet_points[:2])
        if strategy_report:
            highlights.extend(strategy_report.bullet_points[:2])

        audience = str(task_meta.get("memo_audience", "team"))
        title = str(task_meta.get("title", task_name))
        memo = build_hr_memo(audience, title, highlights)

        # --- 2. Three-part scoring ----------------------------------------
        # (a) Structure + required-term coverage (Kamal's stub).
        structure_score = score_memo(memo, task_meta.get("hr_required_terms", []))

        # (b) Professional tone + task-specific rubric (Round 1 grader).
        email_task_id = _BRIEF_TO_EMAIL_TASK.get(task_name, "task_1")
        tone_score = grade_response(
            email=task_meta.get("instruction", ""),
            response=memo,
            task_id=email_task_id,
        )

        # (c) Audience mention bonus.
        audience_hit = 1.0 if audience.lower() in memo.lower() else 0.0

        blended = 0.45 * structure_score + 0.45 * tone_score + 0.10 * audience_hit
        score = _clamp(blended)

        # --- 3. Return a typed ExpertReport -------------------------------
        issues: list[str] = []
        if structure_score < 0.4:
            issues.append("hr:weak_structure_or_missing_required_terms")
        if tone_score < 0.4:
            issues.append("hr:weak_professional_tone")
        if audience_hit == 0.0:
            issues.append(f"hr:audience_not_referenced:{audience}")

        return ExpertReport(
            expert_id="hr",
            title="HR / Communications Memo",
            summary=(
                "Drafted the internal memo and scored it on structure, "
                "professional tone, and audience relevance."
            ),
            metrics={
                "memo_structure_score": _clamp(structure_score),
                "memo_tone_score": _clamp(tone_score),
                "audience_reference": audience_hit,
                "memo_score": score,
            },
            bullet_points=[
                f"Memo addressed to {audience}.",
                f"Structure score {structure_score:.3f}, tone score {tone_score:.3f}.",
                "Blended score uses 45% structure, 45% tone, 10% audience bonus.",
            ],
            issues=issues,
            memo=memo,
            score=score,
        )