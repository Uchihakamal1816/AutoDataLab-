from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from .models import Brief, ExpertReport

_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(score: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, round(float(score), 4)))


def load_metric_ground_truth(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    return {str(row["metric"]): str(row["value"]) for _, row in df.iterrows()}


def _try_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_match_score(expected: Dict[str, str], actual: Dict[str, object]) -> float:
    if not expected:
        return 1.0
    matched = 0.0
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        ev = _try_float(expected_value)
        av = _try_float(actual_value)
        if ev is not None and av is not None:
            tolerance = max(0.5, abs(ev) * 0.02)
            if abs(ev - av) <= tolerance:
                matched += 1.0
        elif str(actual_value) == str(expected_value):
            matched += 1.0
    return matched / max(len(expected), 1)


def strategy_rubric_score(recommendations: List[str], evidence_numbers: Iterable[str], categories: Iterable[str]) -> float:
    if not recommendations:
        return 0.001
    score = 0.0
    if len(recommendations) == 3:
        score += 0.25
    lowered = [r.lower() for r in recommendations]
    nums = [str(n) for n in evidence_numbers]
    if nums and all(any(n in rec for n in nums) for rec in lowered[: min(3, len(lowered))]):
        score += 0.25
    cat_hits = 0
    for cat in set(str(c).lower() for c in categories if c):
        if any(cat in rec for rec in lowered):
            cat_hits += 1
    if cat_hits >= 2:
        score += 0.25
    if any("projection" in rec or "variance" in rec or "break-even" in rec for rec in lowered):
        score += 0.25
    return _clamp(score)


def grade_episode(
    expected_metrics: Dict[str, str],
    task_meta: Dict[str, object],
    brief: Brief,
    reports: Dict[str, ExpertReport],
) -> float:
    required_experts = list(task_meta.get("required_experts", []))
    coverage = 0.0
    if required_experts:
        covered = sum(1 for expert in required_experts if expert in brief.consulted_experts)
        coverage = covered / len(required_experts)

    metric_score = metric_match_score(expected_metrics, brief.metrics)
    hr_score = reports.get("hr").score if reports.get("hr") and reports.get("hr").score is not None else 0.001
    analyst_cats = []
    if reports.get("analyst"):
        analyst_cats = [c for c in reports["analyst"].citations if c]
    evidence_numbers = [str(v) for v in brief.metrics.values()]
    strategy_needed = "strategy" in required_experts
    strategy_score = 1.0 if not strategy_needed else strategy_rubric_score(brief.recommendations, evidence_numbers, analyst_cats)

    weights = {
        "coverage": 0.2,
        "metrics": 0.5,
        "hr": 0.15,
        "strategy": 0.15,
    }
    if not strategy_needed:
        weights["metrics"] += weights["strategy"]
        weights["strategy"] = 0.0

    total = (
        weights["coverage"] * coverage
        + weights["metrics"] * metric_score
        + weights["hr"] * hr_score
        + weights["strategy"] * strategy_score
    )
    return _clamp(total)
