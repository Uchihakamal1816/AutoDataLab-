#!/usr/bin/env python3
"""
One-command Kaggle experiment runner for AutoDataLab++ Qwen2.5-1.5B.

It runs, sequentially:
  1. SFT
  2. DPO
  3. SFT -> DPO
  4. GRPO+RLVR continuation from SFT->DPO
  5. GRPO continuation from SFT->DPO
  6. PPO continuation from SFT->DPO

Every experiment evaluates both:
  - non-RAG mode
  - RAG mode

It writes one leaderboard:
  /kaggle/working/cos_1p5b_all_runs/leaderboard.md
  /kaggle/working/cos_1p5b_all_runs/leaderboard.csv
  /kaggle/working/cos_1p5b_all_runs/leaderboard.json

Recommended Kaggle cell:
    !python3 training/kaggle_run_all_1p5b_experiments.py --quick

If the best RL method beats SFT->DPO, keep it. If not, keep SFT->DPO and
report RL attempts as ablations.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 100, flush=True)
    print("[cmd] " + " ".join(str(x) for x in cmd), flush=True)
    print("=" * 100, flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def evidence_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return -999.0
    vals = []
    for row in rows:
        required = set(row.get("required_experts") or [])
        routed = set(row.get("model_routed_required") or [])
        coverage = len(required & routed) / max(len(required), 1)
        no_fallback = 1.0 if not row.get("needed_fallback") else 0.0
        policy_reward = float(row.get("policy_reward") or 0.0)
        terminal = float(row.get("terminal_score") or 0.0)
        vals.append(coverage + no_fallback + 0.1 * policy_reward + 0.05 * terminal)
    return round(sum(vals) / len(vals), 4)


def collect_rows(root: Path, run_type: str, run_name: str, adapter: Path, eval_dir: Path) -> dict[str, Any]:
    evidence_path = eval_dir / "evidence.json"
    if not evidence_path.is_file():
        return {
            "run_type": run_type,
            "run_name": run_name,
            "adapter": str(adapter),
            "eval_dir": str(eval_dir),
            "score": -999.0,
            "ok": False,
            "rows": [],
        }
    rows = json.loads(evidence_path.read_text(encoding="utf-8"))
    all_no_fallback = all(not r.get("needed_fallback") for r in rows)
    all_full_coverage = all(
        set(r.get("required_experts") or []).issubset(set(r.get("model_routed_required") or []))
        for r in rows
    )
    return {
        "run_type": run_type,
        "run_name": run_name,
        "adapter": str(adapter),
        "eval_dir": str(eval_dir),
        "score": evidence_score(rows),
        "ok": bool(all_no_fallback and all_full_coverage),
        "rows": rows,
    }


def write_leaderboard(out_root: Path, summaries: list[dict[str, Any]]) -> None:
    summaries = sorted(summaries, key=lambda x: x["score"], reverse=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "leaderboard.json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    with (out_root / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "run_type", "run_name", "score", "ok", "adapter", "eval_dir"],
        )
        writer.writeheader()
        for i, row in enumerate(summaries, start=1):
            writer.writerow(
                {
                    "rank": i,
                    "run_type": row["run_type"],
                    "run_name": row["run_name"],
                    "score": row["score"],
                    "ok": row["ok"],
                    "adapter": row["adapter"],
                    "eval_dir": row["eval_dir"],
                }
            )
    md = [
        "# AutoDataLab++ 1.5B Training Leaderboard",
        "",
        "| Rank | Method | Run | Score | Full routing / no fallback | Adapter |",
        "|---:|---|---|---:|---:|---|",
    ]
    for i, row in enumerate(summaries, start=1):
        md.append(
            f"| {i} | {row['run_type']} | `{row['run_name']}` | {row['score']} | "
            f"{row['ok']} | `{row['adapter']}` |"
        )
    if summaries:
        best = summaries[0]
        md += [
            "",
            "## Selected Best",
            "",
            f"- **Method:** {best['run_type']}",
            f"- **Run:** `{best['run_name']}`",
            f"- **Score:** {best['score']}",
            f"- **Adapter:** `{best['adapter']}`",
            f"- **Eval:** `{best['eval_dir']}`",
        ]
    (out_root / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/kaggle/working/cos_1p5b_all_runs") if Path("/kaggle/working").is_dir() else Path("cos_1p5b_all_runs"))
    ap.add_argument("--base-out-root", type=Path, default=Path("/kaggle/working/cos_1p5b_runs") if Path("/kaggle/working").is_dir() else Path("cos_1p5b_runs"))
    ap.add_argument("--rl-out-root", type=Path, default=Path("/kaggle/working/cos_1p5b_rl_runs") if Path("/kaggle/working").is_dir() else Path("cos_1p5b_rl_runs"))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval-tasks", default="expert_brief,risk_brief,crisis_brief")
    ap.add_argument("--eval-rag-modes", default="false,true")
    ap.add_argument("--quick", action="store_true", help="short but useful hackathon run")
    ap.add_argument("--skip-sft", action="store_true")
    ap.add_argument("--skip-dpo", action="store_true")
    ap.add_argument("--skip-rl", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    common = [
        "--model-id",
        args.model_id,
        "--eval-tasks",
        args.eval_tasks,
        "--eval-rag-modes",
        args.eval_rag_modes,
        "--hf-token",
        args.hf_token,
    ]
    base_common = common + ["--out-root", str(args.base_out_root)]
    rl_common = common + ["--out-root", str(args.rl_out_root)]
    if args.quick:
        sft_epochs = "2"
        dpo_epochs = "1"
        rl_epochs = "1"
        max_train_examples = "0"
        max_train_states = "80"
    else:
        sft_epochs = "3"
        dpo_epochs = "2"
        rl_epochs = "2"
        max_train_examples = "0"
        max_train_states = "0"

    summaries: list[dict[str, Any]] = []

    if not args.skip_sft:
        run(
            [
                py,
                "training/kaggle_train_1p5b_methods.py",
                "--method",
                "sft",
                "--epochs",
                sft_epochs,
                "--run-name",
                "qwen15b_sft_all",
                "--max-train-examples",
                max_train_examples,
                *base_common,
            ],
            REPO,
        )
        summaries.append(
            collect_rows(
                args.out_root,
                "sft",
                "qwen15b_sft_all",
                args.base_out_root / "qwen15b_sft_all" / "adapter",
                args.base_out_root / "qwen15b_sft_all" / "eval",
            )
        )

    if not args.skip_dpo:
        run(
            [
                py,
                "training/kaggle_train_1p5b_methods.py",
                "--method",
                "dpo",
                "--epochs",
                dpo_epochs,
                "--run-name",
                "qwen15b_dpo_all",
                "--max-train-examples",
                max_train_examples,
                *base_common,
            ],
            REPO,
        )
        summaries.append(
            collect_rows(
                args.out_root,
                "dpo",
                "qwen15b_dpo_all",
                args.base_out_root / "qwen15b_dpo_all" / "adapter",
                args.base_out_root / "qwen15b_dpo_all" / "eval",
            )
        )

        run(
            [
                py,
                "training/kaggle_train_1p5b_methods.py",
                "--method",
                "sft_then_dpo",
                "--sft-epochs",
                sft_epochs,
                "--dpo-epochs",
                dpo_epochs,
                "--run-name",
                "qwen15b_sft_then_dpo_all",
                "--max-train-examples",
                max_train_examples,
                *base_common,
            ],
            REPO,
        )
        sftdpo_adapter = args.base_out_root / "qwen15b_sft_then_dpo_all" / "adapter"
        summaries.append(
            collect_rows(
                args.out_root,
                "sft_then_dpo",
                "qwen15b_sft_then_dpo_all",
                sftdpo_adapter,
                args.base_out_root / "qwen15b_sft_then_dpo_all" / "eval",
            )
        )
    else:
        sftdpo_adapter = args.base_out_root / "qwen15b_sft_then_dpo_all" / "adapter"

    if not args.skip_rl:
        for method, lr, anchor in [
            ("grpo_rlvr", "5e-6", "0.3"),
            ("grpo", "3e-6", "0.35"),
            ("ppo", "3e-6", "0.35"),
        ]:
            run_name = f"qwen15b_{method}_safe_all"
            run(
                [
                    py,
                    "training/kaggle_rl_1p5b_methods.py",
                    "--method",
                    method,
                    "--init-adapter",
                    str(sftdpo_adapter),
                    "--epochs",
                    rl_epochs,
                    "--lr",
                    lr,
                    "--sft-anchor",
                    anchor,
                    "--max-train-states",
                    max_train_states,
                    "--run-name",
                    run_name,
                    *rl_common,
                ],
                REPO,
            )
            summaries.append(
                collect_rows(
                    args.out_root,
                    method,
                    run_name,
                    args.rl_out_root / run_name / "adapter",
                    args.rl_out_root / run_name / "eval",
                )
            )

    write_leaderboard(args.out_root, summaries)
    print("\n" + "=" * 100, flush=True)
    print(f"[leaderboard] {args.out_root / 'leaderboard.md'}", flush=True)
    print(f"[leaderboard] {args.out_root / 'leaderboard.csv'}", flush=True)
    print(f"[leaderboard] {args.out_root / 'leaderboard.json'}", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
