#!/usr/bin/env python3
"""
Print actual AutoDataLab++ agent text for locally saved Kaggle adapters.

Use this after running training scripts that save adapters under:
  /kaggle/working/cos_1p5b_runs/<run>/adapter
  /kaggle/working/cos_1p5b_rl_runs/<run>/adapter

Example:
    !python3 training/kaggle_compare_local_adapters_text.py \\
      --task expert_brief \\
      --base-model-id Qwen/Qwen2.5-1.5B-Instruct \\
      --include-base \\
      --adapters "sftdpo=/kaggle/working/cos_1p5b_runs/qwen15b_sft_then_dpo_v1/adapter,grpo_rlvr=/kaggle/working/cos_1p5b_rl_runs/qwen15b_grpo_rlvr_safe_v1/adapter"

Outputs:
  - prints each run to notebook output
  - writes JSON/text artifacts under --out-dir

This is for qualitative storytelling: it shows the real model-controlled CoS
trajectory and the actual expert reports / final CEO brief.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kaggle_three_llms_text import (  # noqa: E402
    _format_episode,
    _load_env,
    _load_llm,
    _run_with_policy_trace,
)


def parse_adapter_specs(spec: str) -> list[dict[str, str]]:
    """Parse `name=path[:subfolder],name2=path2` style adapter specs."""
    out: list[dict[str, str]] = []
    for raw in [x.strip() for x in spec.split(",") if x.strip()]:
        if "=" in raw:
            name, rest = raw.split("=", 1)
        else:
            rest = raw
            name = Path(raw).parent.name or Path(raw).name
        # Optional `path::subfolder` syntax avoids ambiguity with HF repo ids.
        if "::" in rest:
            adapter, subfolder = rest.split("::", 1)
        else:
            adapter, subfolder = rest, ""
        out.append({"name": name.strip(), "adapter": adapter.strip(), "subfolder": subfolder.strip()})
    return out


def safe_filename(s: str) -> str:
    keep = []
    for ch in s:
        keep.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "run"


def _has_adapter_weights(path: Path) -> bool:
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def resolve_adapter_id(adapter: str | None) -> str | None:
    """Resolve local Kaggle adapter folders and give useful errors for bad exports."""
    if not adapter:
        return None

    path = Path(adapter).expanduser()
    looks_local = adapter.startswith("/") or adapter.startswith(".") or str(path).startswith("/kaggle/")
    if not looks_local:
        return adapter

    if not path.exists():
        raise FileNotFoundError(
            f"Local adapter path does not exist: {path}\n"
            "Check the run name. For your recent files it may be `qwen15b_sft_then_dpo_all`, "
            "not `qwen15b_sft_then_dpo_v1`."
        )

    candidates = [path]
    if (path / "adapter").is_dir():
        candidates.append(path / "adapter")
    candidates.extend(p.parent for p in path.rglob("adapter_config.json"))

    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / "adapter_config.json").exists() and _has_adapter_weights(cand):
            if cand != path.resolve():
                print(f"[adapter] resolved {path} -> {cand}", flush=True)
            return str(cand)

    configs = [str(p.parent) for p in path.rglob("adapter_config.json")]
    raise FileNotFoundError(
        f"Invalid adapter folder: {path}\n"
        "A loadable LoRA adapter must contain `adapter_config.json` and "
        "`adapter_model.safetensors` or `adapter_model.bin` in the same folder.\n"
        f"Found adapter_config folders: {configs or 'none'}\n"
        "If this came from a filtered zip/export, include the missing `adapter_model.safetensors` file "
        "or point to the original `/kaggle/working/.../adapter` folder before filtering."
    )


def summarize_rows(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Adapter Text Comparison",
        "",
        "| Run | Task | RAG | Action sequence | Routed required | Fallback | Policy reward | Terminal |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        ev = row.get("policy_evidence") or {}
        lines.append(
            f"| {row.get('policy_label')} | {row.get('task')} | {row.get('use_rag')} | "
            f"`{' -> '.join(ev.get('action_sequence') or [])}` | "
            f"{', '.join(ev.get('covered_required_experts_in_policy_steps') or []) or '-'} | "
            f"{ev.get('needed_fallback')} | {ev.get('policy_reward_sum')} | {row.get('terminal_score')} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="expert_brief")
    ap.add_argument("--use-rag", action="store_true")
    ap.add_argument("--base-model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument(
        "--adapters",
        required=True,
        help='comma list: "name=/local/adapter,name2=/local/adapter2" or "name=repo::subfolder"',
    )
    ap.add_argument("--include-base", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--policy-steps", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/kaggle/working/adapter_text_outputs") if Path("/kaggle/working").is_dir() else Path("adapter_text_outputs"),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    CEOBriefEnvironment, _oracle, required_experts_for_task, CoSAction, _CoSObservation = _load_env()
    use_4bit = not args.no_4bit
    specs = parse_adapter_specs(args.adapters)
    if args.include_base:
        specs = [{"name": "base", "adapter": "", "subfolder": ""}] + specs

    all_rows: list[dict[str, Any]] = []
    big_sep = "#" * 100
    for spec in specs:
        label = spec["name"]
        adapter = spec["adapter"] or None
        subfolder = spec["subfolder"] or None
        tok = None
        model = None
        print(
            f"\n{big_sep}\n# RUN: {label}\n# base={args.base_model_id}\n# adapter={adapter or '(none)'}"
            f"{(' subfolder=' + subfolder) if subfolder else ''}\n{big_sep}",
            flush=True,
        )
        try:
            adapter = resolve_adapter_id(adapter)
            tok, model = _load_llm(
                args.base_model_id,
                adapter,
                subfolder,
                use_4bit=use_4bit,
                hf_token=args.hf_token or None,
            )
            data = _run_with_policy_trace(
                tok,
                model,
                CoSAction,
                CEOBriefEnvironment,
                required_experts_for_task,
                label=label,
                task=args.task,
                use_rag=bool(args.use_rag),
                max_new_tokens=args.max_new_tokens,
                policy_steps=args.policy_steps,
                shaping="strict",
            )
        except Exception as exc:
            print(f"[failed] {label}: {exc}", flush=True)
            continue
        finally:
            try:
                if model is not None:
                    del model
                if tok is not None:
                    del tok
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        text = _format_episode(data)
        print(text, flush=True)
        stem = safe_filename(f"{label}__{args.task}__rag_{args.use_rag}")
        (args.out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        (args.out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        all_rows.append(data)
        print(f"[saved] {args.out_dir / f'{stem}.json'}", flush=True)
        print(f"[saved] {args.out_dir / f'{stem}.txt'}", flush=True)

    summary = summarize_rows(all_rows)
    (args.out_dir / f"summary__{safe_filename(args.task)}__rag_{args.use_rag}.md").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    print(f"\n[summary] {args.out_dir / f'summary__{safe_filename(args.task)}__rag_{args.use_rag}.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
