#!/usr/bin/env python3
"""
Print full expert + brief text from one AutoDataLab++ episode (Kaggle / local).
Does not emphasize scores — use for demos, reports, and qualitative inspection.

  python3 -m training.kaggle_agent_answers --task expert_brief --use-rag

If run from Kaggle, clone the repo and `pip install -e .` first, then:
  !python3 training/kaggle_agent_answers.py --task risk_brief
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ceo_brief_env.environment import oracle_action_for_observation
from inference import _roundrobin_baseline, _single_baseline, _trained_action, run_episode_collect


def _line(title: str, body: str, width: int = 88) -> str:
    sep = "=" * width
    return f"\n{sep}\n{title}\n{sep}\n{body.rstrip()}\n"


def format_episode_answers(data: dict, *, show_scores: bool = False) -> str:
    """Format expert reports and CEO brief for human reading."""
    out: list[str] = []
    out.append(
        f"Task: {data.get('task')}\n"
        f"Policy: {data.get('policy_label')}\n"
        f"RAG: {data.get('use_rag')}\n"
    )
    if show_scores:
        out.append(
            f"Terminal score (grader): {data.get('terminal_score')}  "
            f"success: {data.get('success')}\n"
        )

    inst = data.get("final_instruction") or ""
    if inst:
        out.append(_line("INSTRUCTION (from metadata)", inst))

    reports: dict = data.get("expert_reports") or {}
    order = ("analyst", "finance", "strategy", "hr")
    labels = {
        "analyst": "DATA ANALYST",
        "finance": "FINANCE",
        "strategy": "STRATEGIST",
        "hr": "HR / COMMS",
    }
    for eid in order:
        r = reports.get(eid)
        if not r:
            continue
        title = f"{labels.get(eid, eid).upper()} — {r.get('title', eid)}"
        chunks = [r.get("summary", "").strip() or "(no summary)"]
        bps = r.get("bullet_points") or []
        if bps:
            chunks.append("\nBullets:\n" + "\n".join(f"  • {b}" for b in bps))
        issues = r.get("issues") or []
        if issues and issues != ["(none)"]:
            chunks.append("\nIssues:\n" + "\n".join(f"  - {i}" for i in issues))
        m_c = r.get("memory_citations") or []
        m_s = r.get("memory_snippets") or []
        n = min(len(m_c), len(m_s), 5)
        if n:
            tape = [f"\nTape & citations (from strategist / RAG) — first {n}:"]
            for i in range(n):
                tape.append(f"  [{m_c[i]}] {m_s[i][:500]}{'…' if len(m_s[i]) > 500 else ''}")
            chunks.append("\n".join(tape))
        if eid == "hr" and r.get("memo"):
            chunks.append("\nHR memo field:\n" + str(r["memo"]))
        out.append(_line(title, "\n".join(chunks)))

    brief = data.get("current_brief")
    if brief:
        parts = [brief.get("summary", "") or ""]
        recs = brief.get("recommendations") or []
        if recs:
            parts.append("\nRecommendations:\n" + "\n".join(f"  • {x}" for x in recs))
        m = brief.get("hr_memo")
        if m:
            parts.append(f"\nHR memo (in brief object):\n{m}")
        cons = brief.get("consulted_experts")
        if cons is not None:
            parts.append(f"\nConsulted in brief object: {', '.join(cons)}")
        out.append(_line("COMPOSED BRIEF (to CEO — merged from reports)", "\n".join(parts)))

    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="expert_brief", help="e.g. expert_brief, risk_brief, crisis_brief, easy_brief")
    p.add_argument("--use-rag", action="store_true", help="enable RAG in experts + grader grounding")
    p.add_argument(
        "--policy",
        choices=("oracle", "single", "roundrobin", "trained"),
        default="oracle",
        help="oracle = consult all required experts in order; best for a full 'office' readout",
    )
    p.add_argument("--json-out", type=Path, help="write full run_episode_collect() dict to this path")
    p.add_argument("--show-scores", action="store_true", help="include terminal grader line in header")
    args = p.parse_args()

    if args.policy == "oracle":
        picker, label = oracle_action_for_observation, "oracle"
    elif args.policy == "single":
        picker, label = _single_baseline, "single-baseline"
    elif args.policy == "roundrobin":
        picker, label = _roundrobin_baseline, "roundrobin-baseline"
    else:
        picker, label = _trained_action, "trained-cos"

    data = run_episode_collect(args.task, picker, label, use_rag=bool(args.use_rag), quiet=True)
    print(format_episode_answers(data, show_scores=bool(args.show_scores)))

    if args.json_out:
        args.json_out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"\n[written] {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
