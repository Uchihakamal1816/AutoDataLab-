from __future__ import annotations

from typing import Iterable, Sequence


def build_hr_memo(audience: str, task_title: str, highlights: Sequence[str]) -> str:
    opening = f"Hello {audience},"
    body = [
        "Thank you for moving quickly on this update.",
        f"Here is the latest business update for {task_title}.",
    ]
    body.extend(f"- {line}" for line in highlights[:4])
    close = [
        "Please review the actions above and reply with any blockers today.",
        "Best,",
        "HR Operations",
    ]
    return "\n".join([opening, "", *body, "", *close])


def score_memo(memo: str, required_terms: Iterable[str]) -> float:
    text = memo.lower()
    score = 0.2
    if memo.startswith("Hello "):
        score += 0.15
    if "thank" in text or "appreciate" in text:
        score += 0.15
    if "please review" in text or "reply with any blockers" in text:
        score += 0.2
    if "best," in text:
        score += 0.1
    matched = 0
    terms = [t.lower() for t in required_terms if t]
    for term in terms:
        if term in text:
            matched += 1
    if terms:
        score += 0.2 * (matched / len(terms))
    return max(0.001, min(0.999, round(score, 4)))
