#!/usr/bin/env python3
"""
Build full contextual text results from saved training evidence.

This script does NOT load LoRA adapters. That is intentional: it works even
when a run such as GRPO+RLVR has no `adapter_config.json` or adapter weights in
the exported folder. It reads `evidence.json`, replays the recorded CoS action
sequence in AutoDataLab++, and prints/saves the full expert reports plus CEO
brief for SFT, DPO, SFT+DPO, and GRPO+RLVR.

Kaggle example:
    !python3 training/kaggle_context_results_from_evidence.py \\
      --roots /kaggle/working /kaggle/input/results-filtered \\
      --tasks expert_brief,risk_brief,crisis_brief \\
      --rag-modes false,true
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ceo_brief_env.environment import CEOBriefEnvironment, required_experts_for_task
from ceo_brief_env.models import CoSAction
from kaggle_agent_answers import format_episode_answers


DEFAULT_RUN_PATTERNS: dict[str, list[str]] = {
    "sft": [
        "training/evidence/sft/evidence.json",
        "**/training/evidence/sft/evidence.json",
        "**/qwen15b_sft_all/eval/evidence.json",
        "**/qwen15b_sft_v1/eval/evidence.json",
    ],
    "dpo": [
        "training/evidence/dpo/evidence.json",
        "**/training/evidence/dpo/evidence.json",
        "**/qwen15b_dpo_all/eval/evidence.json",
        "**/qwen15b_dpo_v1/eval/evidence.json",
    ],
    "sft_dpo": [
        "training/evidence/sft_dpo/evidence.json",
        "**/training/evidence/sft_dpo/evidence.json",
        "**/qwen15b_sft_then_dpo_all/eval/evidence.json",
        "**/qwen15b_sft_then_dpo_v1/eval/evidence.json",
    ],
    "grpo_rlvr": [
        "training/evidence/grpo_rlvr/evidence.json",
        "**/training/evidence/grpo_rlvr/evidence.json",
        "**/qwen15b_grpo_rlvr_safe_all/eval/evidence.json",
        "**/qwen15b_grpo_rlvr_safe_all/eval_after/evidence.json",
        "**/qwen15b_grpo_rlvr_safe_all/eval_before/evidence.json",
        "**/qwen15b_grpo_rlvr_safe_v1/eval/evidence.json",
        "**/qwen15b_grpo_rlvr_safe_v1/eval_after/evidence.json",
        "**/qwen15b_grpo_rlvr_safe_v1/eval_before/evidence.json",
        "**/qwen15b_grpo_rlvr*/eval/evidence.json",
        "**/qwen15b_grpo_rlvr*/eval_after/evidence.json",
        "**/qwen15b_grpo_rlvr*/eval_before/evidence.json",
    ],
}


def parse_bool_text(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y", "rag"}


def action_from_label(label: str) -> CoSAction:
    label = (label or "").strip()
    if ":" in label:
        action_type, expert_id = label.split(":", 1)
        expert_id = expert_id.strip() or None
        if expert_id in {"none", "null"}:
            expert_id = None
        return CoSAction(action_type=action_type.strip(), expert_id=expert_id)
    return CoSAction(action_type=label)


def action_from_any(item: Any) -> CoSAction:
    if isinstance(item, dict):
        if "action" in item and isinstance(item["action"], dict):
            return CoSAction.model_validate(item["action"])
        return CoSAction.model_validate(item)
    return action_from_label(str(item))


def action_label(action: CoSAction) -> str:
    if action.action_type in {"consult", "ask"}:
        return f"{action.action_type}:{action.expert_id or 'null'}"
    return action.action_type


def discover_evidence(roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for label, patterns in DEFAULT_RUN_PATTERNS.items():
        for root in roots:
            if not root.exists():
                continue
            for pattern in patterns:
                matches = sorted(root.glob(pattern), key=lambda p: (len(str(p)), str(p)))
                if matches:
                    found[label] = matches[0]
                    break
            if label in found:
                break
    return found


def load_evidence(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    if not isinstance(data, list):
        raise ValueError(f"Expected list evidence in {path}")
    return [x for x in data if isinstance(x, dict)]


def deterministic_finish(obs, task: str) -> CoSAction:
    for expert in required_experts_for_task(task):
        if expert not in obs.consulted_experts:
            return CoSAction(action_type="consult", expert_id=expert)
    if obs.current_brief is None:
        return CoSAction(action_type="summarize")
    return CoSAction(action_type="submit")


def replay_row(method: str, row: dict[str, Any], complete_if_needed: bool) -> dict[str, Any]:
    task = str(row.get("task") or "expert_brief")
    use_rag = bool(row.get("rag", row.get("use_rag", False)))
    env = CEOBriefEnvironment(shaping="strict", auto_fill_required=False)
    obs = env.reset(task=task, use_rag=use_rag)

    model_actions = [action_from_any(x) for x in (row.get("action_sequence") or [])]
    fallback_actions = [action_from_any(x) for x in (row.get("fallback") or [])]
    all_actions = model_actions + fallback_actions

    trace: list[dict[str, Any]] = []
    rewards: list[float] = []
    for action in all_actions:
        if obs.done:
            break
        obs = env.step(action)
        rewards.append(float(obs.reward))
        trace.append(
            {
                "step": obs.step_count,
                "action": action.model_dump(exclude_none=True),
                "action_label": action_label(action),
                "reward": round(float(obs.reward), 4),
                "done": bool(obs.done),
                "consulted_experts": list(obs.consulted_experts),
                "source": "model" if len(trace) < len(model_actions) else "fallback",
            }
        )

    auto_finish: list[str] = []
    while complete_if_needed and not obs.done and obs.step_count < obs.max_steps:
        action = deterministic_finish(obs, task)
        auto_finish.append(action_label(action))
        obs = env.step(action)
        rewards.append(float(obs.reward))
        trace.append(
            {
                "step": obs.step_count,
                "action": action.model_dump(exclude_none=True),
                "action_label": action_label(action),
                "reward": round(float(obs.reward), 4),
                "done": bool(obs.done),
                "consulted_experts": list(obs.consulted_experts),
                "source": "auto_finish",
            }
        )

    score = max(0.001, min(0.999, float(obs.terminal_grader_score or 0.001)))
    return {
        "task": task,
        "policy_label": method,
        "use_rag": use_rag,
        "success": score >= 0.5,
        "steps": obs.step_count,
        "terminal_score": round(score, 4),
        "cumulative_reward": round(sum(rewards), 4),
        "step_rewards": [round(x, 4) for x in rewards],
        "trace": trace,
        "error": None,
        "final_instruction": obs.instruction,
        "task_difficulty": obs.task_difficulty,
        "max_steps": obs.max_steps,
        "consulted_experts": list(obs.consulted_experts),
        "current_brief": obs.current_brief.model_dump() if obs.current_brief is not None else None,
        "expert_reports": {k: v.model_dump() for k, v in obs.expert_reports.items()},
        "evidence": {
            "recorded_model_actions": [action_label(a) for a in model_actions],
            "recorded_fallback_actions": [action_label(a) for a in fallback_actions],
            "auto_finish_actions": auto_finish,
            "recorded_policy_reward": row.get("policy_reward"),
            "recorded_terminal_score": row.get("terminal_score"),
            "needed_fallback": row.get("needed_fallback"),
            "model_routed_required": row.get("model_routed_required") or [],
            "required_experts": row.get("required_experts") or required_experts_for_task(task),
            "trace_completion_previews": [
                t.get("completion_preview")
                for t in (row.get("trace") or [])
                if isinstance(t, dict) and t.get("completion_preview")
            ],
        },
    }


def evidence_header(data: dict[str, Any]) -> str:
    ev = data.get("evidence") or {}
    lines = [
        "TRAINING EVIDENCE CONTEXT",
        f"Method: {data.get('policy_label')}",
        f"Task: {data.get('task')} | RAG: {data.get('use_rag')}",
        f"Recorded model route: {' -> '.join(ev.get('recorded_model_actions') or []) or '-'}",
        f"Recorded fallback: {' -> '.join(ev.get('recorded_fallback_actions') or []) or '-'}",
        f"Auto-finish used by this report: {' -> '.join(ev.get('auto_finish_actions') or []) or '-'}",
        f"Required routed by model: {', '.join(ev.get('model_routed_required') or []) or '-'}",
        f"Recorded policy reward: {ev.get('recorded_policy_reward')} | recorded terminal: {ev.get('recorded_terminal_score')}",
        f"Replay terminal: {data.get('terminal_score')} | replay cumulative: {data.get('cumulative_reward')}",
    ]
    previews = ev.get("trace_completion_previews") or []
    if previews:
        lines.append("\nRaw action completions / previews:")
        lines.extend(f"  {i + 1}. {p}" for i, p in enumerate(previews[:8]))
    return "\n".join(lines)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=[
            Path("/kaggle/working"),
            Path("/kaggle/input"),
            Path.cwd(),
            REPO / "results",
        ],
        help="Folders to search recursively for evidence.json files.",
    )
    ap.add_argument(
        "--evidence",
        action="append",
        default=[],
        help="Manual mapping: label=/path/to/evidence.json. Can be repeated.",
    )
    ap.add_argument("--tasks", default="expert_brief,risk_brief,crisis_brief")
    ap.add_argument("--rag-modes", default="false,true")
    ap.add_argument("--complete-if-needed", action="store_true", default=True)
    ap.add_argument("--no-complete", dest="complete_if_needed", action="store_false")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/kaggle/working/context_results_all_methods")
        if Path("/kaggle/working").is_dir()
        else Path("context_results_all_methods"),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    evidence_map = discover_evidence(args.roots)
    for spec in args.evidence:
        if "=" not in spec:
            raise ValueError("--evidence must be label=/path/to/evidence.json")
        label, path = spec.split("=", 1)
        evidence_map[label.strip()] = Path(path.strip())

    wanted_tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    wanted_rag = {parse_bool_text(x) for x in args.rag_modes.split(",") if x.strip()}

    if not evidence_map:
        print("[error] No evidence.json files found. Pass --evidence label=/path/to/evidence.json", file=sys.stderr)
        return 2

    summary_rows: list[dict[str, Any]] = []
    print("# AutoDataLab++ Context Results From Evidence\n")
    print("Evidence files:")
    for label, path in evidence_map.items():
        print(f"- {label}: {path}")
    print()

    for method in ["sft", "dpo", "sft_dpo", "grpo_rlvr"]:
        path = evidence_map.get(method)
        if not path:
            print(f"[skip] {method}: evidence.json not found")
            continue
        rows = load_evidence(path)
        for row in rows:
            task = str(row.get("task") or "")
            rag = bool(row.get("rag", row.get("use_rag", False)))
            if task not in wanted_tasks or rag not in wanted_rag:
                continue
            data = replay_row(method, row, complete_if_needed=bool(args.complete_if_needed))
            text = evidence_header(data) + "\n\n" + format_episode_answers(data, show_scores=True)
            stem = safe_name(f"{method}__{task}__rag_{rag}")
            (args.out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
            (args.out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            print("\n" + "#" * 96)
            print(f"RESULT: {method} | task={task} | rag={rag}")
            print("#" * 96)
            print(text)
            summary_rows.append(
                {
                    "method": method,
                    "task": task,
                    "rag": rag,
                    "model_route": " -> ".join(data["evidence"]["recorded_model_actions"]),
                    "fallback": " -> ".join(data["evidence"]["recorded_fallback_actions"]),
                    "auto_finish": " -> ".join(data["evidence"]["auto_finish_actions"]),
                    "recorded_policy_reward": data["evidence"]["recorded_policy_reward"],
                    "recorded_terminal": data["evidence"]["recorded_terminal_score"],
                    "replay_terminal": data["terminal_score"],
                    "consulted": ", ".join(data["consulted_experts"]),
                }
            )

    md = [
        "# AutoDataLab++ Method Context Summary",
        "",
        "| Method | Task | RAG | Model route | Fallback | Auto-finish | Policy reward | Terminal | Consulted |",
        "|---|---|---:|---|---|---|---:|---:|---|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['method']} | {row['task']} | {row['rag']} | `{row['model_route']}` | "
            f"`{row['fallback'] or '-'}` | `{row['auto_finish'] or '-'}` | "
            f"{row['recorded_policy_reward']} | {row['replay_terminal']} | {row['consulted']} |"
        )
    summary = "\n".join(md)
    (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print("\n" + summary)
    print(f"\n[saved] {args.out_dir}")
    print("[note] No adapters were loaded, so missing GRPO+RLVR adapter_config.json is not a problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
