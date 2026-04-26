"""
Build a small SFT dataset for the Chief of Staff from oracle rollouts.

Output:
    - ``training/sft_data/cos_sft.jsonl`` for chat/message based trainers.
    - ``training/sft_data/cos_sft_autotrain.jsonl`` for AutoTrain SFT, with a
      single ``text`` column.

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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ceo_brief_env.environment import (  # noqa: E402
    CEOBriefEnvironment,
    oracle_action_for_observation,
    required_experts_for_task,
)
from ceo_brief_env.models import CoSAction, CoSObservation  # noqa: E402

OUT_DIR = REPO / "training" / "sft_data"
OUT_PATH = OUT_DIR / "cos_sft.jsonl"
AUTOTRAIN_OUT_PATH = OUT_DIR / "cos_sft_autotrain.jsonl"

TASKS = ["easy_brief", "medium_brief", "hard_brief", "expert_brief"]
RAG_MODES = [False, True]

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

SYSTEM_PROMPTS = [
    SYSTEM_PROMPT,
    (
        "Act as AutoDataLab++ Chief of Staff. Choose the next orchestration step. "
        "Return strict JSON only with keys action_type, expert_id, sub_question_id, notes. "
        "Valid actions: consult, ask, summarize, submit, noop. Valid experts: analyst, finance, hr, strategy."
    ),
    (
        "You are a routing policy for a CEO brief environment. Consult missing required experts first, "
        "then summarize, then submit. Output one JSON object only; no markdown and no explanation."
    ),
]

OBSERVATION_VARIANTS = range(6)


def render_observation(obs: CoSObservation, variant: int = 0) -> str:
    required = required_experts_for_task(obs.task_name)
    missing = [e for e in required if e not in obs.consulted_experts]
    parts = [
        f"task_name: {obs.task_name}",
        f"task_difficulty: {obs.task_difficulty}",
        f"step_count: {obs.step_count}",
        f"max_steps: {obs.max_steps}",
        f"rag_enabled: {obs.rag_enabled}",
        f"available_experts: {obs.available_experts}",
        f"required_experts: {required}",
        f"consulted_experts: {obs.consulted_experts}",
        f"missing_required_experts: {missing}",
        f"current_brief_present: {obs.current_brief is not None}",
        f"data_quality_score: {round(float(obs.data_quality_score or 0.0), 4)}",
        f"recent_issues: {obs.issues[-3:]}",
        f"instruction: {obs.instruction}",
    ]
    if variant == 0:
        return "\n".join(parts)
    if variant == 1:
        return (
            f"Task={obs.task_name}; difficulty={obs.task_difficulty}; "
            f"step={obs.step_count}/{obs.max_steps}; rag={obs.rag_enabled}; "
            f"required={required}; consulted={obs.consulted_experts}; missing={missing}; "
            f"brief_ready={obs.current_brief is not None}; dq={round(float(obs.data_quality_score or 0.0), 4)}"
        )
    if variant == 2:
        return json.dumps(
            {
                "task": obs.task_name,
                "difficulty": obs.task_difficulty,
                "step": obs.step_count,
                "max_steps": obs.max_steps,
                "rag_enabled": obs.rag_enabled,
                "required_experts": required,
                "consulted_experts": obs.consulted_experts,
                "missing_required_experts": missing,
                "brief_ready": obs.current_brief is not None,
                "data_quality_score": round(float(obs.data_quality_score or 0.0), 4),
            },
            separators=(",", ":"),
        )
    if variant == 3:
        return (
            "Decision checklist:\n"
            f"- task: {obs.task_name}\n"
            f"- required experts: {', '.join(required)}\n"
            f"- already consulted: {', '.join(obs.consulted_experts) or 'none'}\n"
            f"- still missing: {', '.join(missing) or 'none'}\n"
            f"- brief composed: {obs.current_brief is not None}\n"
            f"- choose the single next action"
        )
    if variant == 4:
        if missing:
            status = f"The next priority is one of the missing experts: {missing}."
        elif obs.current_brief is None:
            status = "All required experts are consulted, but the brief is not summarized yet."
        else:
            status = "The brief is summarized and ready for final submission."
        return (
            f"You are at step {obs.step_count} for {obs.task_name}. "
            f"Consulted experts are {obs.consulted_experts}. {status} "
            f"RAG mode is {obs.rag_enabled}."
        )
    return (
        f"state: task={obs.task_name} | required={required} | consulted={obs.consulted_experts} | "
        f"missing={missing} | has_brief={obs.current_brief is not None} | action?"
    )


def action_to_json(action: CoSAction) -> str:
    payload: dict[str, Any] = action.model_dump(exclude_none=True)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def collect_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in TASKS:
        for use_rag in RAG_MODES:
            env = CEOBriefEnvironment()
            obs = env.reset(task=task, use_rag=use_rag)
            while not obs.done and obs.step_count < obs.max_steps:
                action = oracle_action_for_observation(obs)
                assistant_msg = action_to_json(action)
                for system_prompt in SYSTEM_PROMPTS:
                    for variant in OBSERVATION_VARIANTS:
                        user_msg = render_observation(obs, variant=variant)
                        records.append(
                            {
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_msg},
                                    {"role": "assistant", "content": assistant_msg},
                                ]
                            }
                        )
                obs = env.step(action)
    return records


def render_autotrain_text(record: dict[str, Any]) -> str:
    messages = record["messages"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    assistant = messages[2]["content"]
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant}<|im_end|>"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with AUTOTRAIN_OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"text": render_autotrain_text(rec)}, ensure_ascii=False) + "\n")
    print(f"[sft-data] wrote {len(records)} examples to {OUT_PATH}")
    print(f"[sft-data] wrote AutoTrain text dataset to {AUTOTRAIN_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
