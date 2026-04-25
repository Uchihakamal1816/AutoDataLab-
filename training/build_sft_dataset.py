"""
Build a small SFT dataset for the Chief of Staff from oracle rollouts.

Output: ``training/sft_data/cos_sft.jsonl`` (one JSON object per line).

Format (Hugging Face chat-template friendly):
    {"messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "<observation snapshot>"},
        {"role": "assistant", "content": "<JSON action>"}
    ]}

No GPU and no API key required. Run before SFT:

    python3 training/build_sft_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ceo_brief_env.environment import (  # noqa: E402
    CEOBriefEnvironment,
    oracle_action_for_observation,
)
from ceo_brief_env.models import CoSAction, CoSObservation  # noqa: E402

OUT_DIR = REPO / "training" / "sft_data"
OUT_PATH = OUT_DIR / "cos_sft.jsonl"

TASKS = ["easy_brief", "medium_brief", "hard_brief", "expert_brief"]
RAG_MODES = [False, True]
ROLLOUTS_PER_COMBO = 2

SYSTEM_PROMPT = (
    "You are the Chief of Staff (CoS) in AutoDataLab++. You orchestrate four "
    "specialists: analyst, finance, strategy, hr. At each step, decide ONE "
    "action and reply with STRICT JSON only.\n\n"
    "Schema: {\"action_type\": one of [\"consult\", \"ask\", \"summarize\", \"submit\", \"noop\"], "
    "\"expert_id\": one of [\"analyst\", \"finance\", \"hr\", \"strategy\"] or null, "
    "\"sub_question_id\": string or null, \"notes\": string or null}.\n"
    "Rules: consult each required expert at most once before summarizing; "
    "summarize before submitting; submit only when the brief is composed."
)


def render_observation(obs: CoSObservation) -> str:
    parts = [
        f"task_name: {obs.task_name}",
        f"task_difficulty: {obs.task_difficulty}",
        f"step_count: {obs.step_count}",
        f"max_steps: {obs.max_steps}",
        f"rag_enabled: {obs.rag_enabled}",
        f"available_experts: {obs.available_experts}",
        f"consulted_experts: {obs.consulted_experts}",
        f"current_brief_present: {obs.current_brief is not None}",
        f"data_quality_score: {round(float(obs.data_quality_score or 0.0), 4)}",
        f"recent_issues: {obs.issues[-3:]}",
        f"instruction: {obs.instruction}",
    ]
    return "\n".join(parts)


def action_to_json(action: CoSAction) -> str:
    payload: dict[str, Any] = action.model_dump(exclude_none=True)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def collect_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in TASKS:
        for use_rag in RAG_MODES:
            for _ in range(ROLLOUTS_PER_COMBO):
                env = CEOBriefEnvironment()
                obs = env.reset(task=task, use_rag=use_rag)
                while not obs.done and obs.step_count < obs.max_steps:
                    action = oracle_action_for_observation(obs)
                    user_msg = render_observation(obs)
                    assistant_msg = action_to_json(action)
                    records.append(
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": assistant_msg},
                            ]
                        }
                    )
                    obs = env.step(action)
    return records


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[sft-data] wrote {len(records)} examples to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
