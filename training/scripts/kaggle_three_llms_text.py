#!/usr/bin/env python3
"""
Kaggle-ready: run AutoDataLab++ once with 3 different LLM CoS pickers and print
the *full agent text* for each run. No scores, just answers.

Three pickers:
  1) base LLM            (BASE_MODEL_ID, no adapter)
  2) base LLM + SFT LoRA (BASE_MODEL_ID + SFT_ADAPTER_ID)
  3) base LLM + GRPO LoRA (BASE_MODEL_ID + GRPO_ADAPTER_ID, optional subfolder)

Usage on Kaggle (Settings → Internet: ON, GPU: T4 or better):
    !pip install -q "transformers>=4.45,<4.49" "peft>=0.13,<0.16" \
        "accelerate>=0.33,<1.1" bitsandbytes huggingface_hub pydantic
    !python3 training/kaggle_three_llms_text.py --task expert_brief

Set TASK and adapter ids below or via CLI flags.

The CoS LLM only chooses the *first* action. The env then runs deterministic
continuation to make sure all required experts (analyst, finance, strategy, hr)
report. That guarantees you always get a full text readout to compare.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Kaggle can trip torch.compile/triton paths when transformers imports optional
# packages. Keep inference on the plain eager path.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Make the env package importable both as a script and from a Kaggle notebook.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------- Config defaults (override via CLI) ----------
DEFAULTS = {
    "task": "expert_brief",
    "use_rag": False,
    "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "sft_adapter_id": "uchihakamal/qwen2.5-1.5b-autodatalab-sft",   # change to your repo
    "sft_subfolder": "",
    "grpo_adapter_id": "uchihakamal/qwen2.5-7b-autodatalab-grpo",   # change to your repo
    "grpo_subfolder": "final",
    "use_4bit": True,
    "max_new_tokens": 96,
    "shaping": "strict",
}

# ---------- Env imports (deferred so --help works without deps) ----------


def _load_env():
    from ceo_brief_env.environment import (  # noqa: F401
        CEOBriefEnvironment,
        oracle_action_for_observation,
        required_experts_for_task,
    )
    from ceo_brief_env.models import CoSAction, CoSObservation  # noqa: F401

    return (
        CEOBriefEnvironment,
        oracle_action_for_observation,
        required_experts_for_task,
        CoSAction,
        CoSObservation,
    )


# ---------- LLM action policy (LLM picks first action; rest is deterministic) ----------

_VALID_ACTIONS = {"consult", "ask", "summarize", "submit", "noop"}
_VALID_EXPERTS = {"analyst", "finance", "hr", "strategy"}
_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)

_SYSTEM_PROMPT = (
    "You are the Chief of Staff in AutoDataLab++. You orchestrate four "
    "specialists: analyst, finance, strategy, hr. Reply with STRICT JSON only.\n"
    'Schema: {"action_type": one of [consult, ask, summarize, submit, noop], '
    '"expert_id": one of [analyst, finance, hr, strategy] or null}.\n'
    "Rules: consult each required expert at most once -> summarize -> submit."
)


def _render_obs(obs) -> str:
    return (
        f"task={obs.task_name} step={obs.step_count}/{obs.max_steps} "
        f"rag={obs.rag_enabled} consulted={obs.consulted_experts} "
        f"brief_done={obs.current_brief is not None} available={obs.available_experts}"
    )


def _parse_action(text: str, CoSAction):
    m = _JSON_RE.search(text or "")
    if not m:
        return CoSAction(action_type="noop")
    try:
        a = json.loads(m.group(0))
    except Exception:
        return CoSAction(action_type="noop")
    at = a.get("action_type")
    if at not in _VALID_ACTIONS:
        return CoSAction(action_type="noop")
    eid = a.get("expert_id")
    if eid is not None and eid not in _VALID_EXPERTS:
        eid = None
    return CoSAction(action_type=at, expert_id=eid)


# ---------- Model loading ----------


def _load_llm(base_model_id: str, adapter_id: Optional[str], adapter_subfolder: Optional[str], use_4bit: bool, hf_token: Optional[str]):
    """Load base model and (optionally) attach a PEFT/LoRA adapter.

    When an adapter is present we:
      * load the *adapter's* tokenizer (training-time chat template + special
        tokens; avoids vocab drift),
      * resize base model embeddings to that tokenizer **before** PEFT attach,
      * load adapter and ``merge_and_unload`` for a clean inference graph
        (also fixes ``set_input_embeddings``-style layer mismatch errors).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = adapter_id or base_model_id
    tok_kw: dict[str, Any] = {"token": hf_token}
    if adapter_id and adapter_subfolder:
        tok_kw["subfolder"] = adapter_subfolder
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_source, **tok_kw)
    except Exception as e:
        if not adapter_id:
            raise
        print(
            f"[load] warn: adapter tokenizer failed ({e}); falling back to base tokenizer.",
            flush=True,
        )
        tok = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            bnb_cfg = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            token=hf_token,
            device_map="auto",
            quantization_config=bnb_cfg,
            torch_dtype=torch.float16,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    if adapter_id:
        try:
            model.resize_token_embeddings(len(tok))
        except Exception as e:
            print(f"[load] warn: resize_token_embeddings failed: {e}", flush=True)

    model.eval()
    # Greedy generation; clear sampling defaults from model generation_config to
    # avoid noisy warnings (`temperature/top_p/top_k` are ignored when
    # do_sample=False).
    try:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    except Exception:
        pass

    if adapter_id:
        from peft import PeftModel

        kw: dict[str, Any] = {"token": hf_token}
        if adapter_subfolder:
            kw["subfolder"] = adapter_subfolder
        model = PeftModel.from_pretrained(model, adapter_id, **kw)
        try:
            model = model.merge_and_unload()
        except Exception as e:
            print(
                f"[load] warn: merge_and_unload failed ({e}); keeping PEFT wrapper.",
                flush=True,
            )
        model.eval()

    return tok, model


def _action_label(action) -> str:
    if action.action_type in {"consult", "ask"}:
        return f"{action.action_type}:{action.expert_id or 'null'}"
    return action.action_type


def _generate_action_and_text(tok, model, CoSAction, obs, max_new_tokens: int = 96):
    """Ask the LLM for one CoS action and keep the raw completion for evidence."""
    import torch

    with torch.no_grad():
        msgs = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _render_obs(obs)},
        ]
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = _SYSTEM_PROMPT + "\n\n" + _render_obs(obs) + "\n"
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        action = _parse_action(text, CoSAction)
        return action, text.strip()


def _make_llm_picker(tok, model, CoSAction, max_new_tokens: int = 96):
    """Returns a CoS picker that asks the LLM, parses JSON, falls back to noop."""

    def picker(obs):
        action, _completion = _generate_action_and_text(
            tok, model, CoSAction, obs, max_new_tokens=max_new_tokens
        )
        return action

    return picker


# ---------- Run one episode and pretty-print agent text ----------


def _format_episode(data: dict) -> str:
    """Mirror training/kaggle_agent_answers.format_episode_answers but stand-alone."""
    width = 88
    sep = "=" * width
    out: list[str] = []
    out.append(
        f"Task: {data.get('task')}\nPolicy: {data.get('policy_label')}\nRAG: {data.get('use_rag')}\n"
    )
    inst = data.get("final_instruction") or ""
    if inst:
        out += [sep, "INSTRUCTION (from metadata)", sep, inst, ""]

    policy_trace = data.get("policy_trace") or []
    fallback_trace = data.get("fallback_trace") or []
    if policy_trace:
        out += [sep, "TRAINING EVIDENCE - MODEL-CONTROLLED POLICY TRAJECTORY", sep]
        out.append(
            "These are the actions chosen directly by the LLM/adapter before any "
            "deterministic completion logic. This is the part to compare across "
            "base vs SFT vs GRPO."
        )
        for row in policy_trace:
            out.append(
                f"  step {row['step']:02d} | action={row['action_label']:<18} "
                f"reward={row['reward']:+.4f} done={row['done']} "
                f"model_routed={row['model_routed_required']} "
                f"env_consulted={row['consulted_after']}"
            )
            preview = str(row.get("completion_preview") or "").replace("\n", " ")
            if preview:
                out.append(f"       raw: {preview}")
        if fallback_trace:
            out.append(
                "\nFallback used after model-controlled steps to complete the report "
                "for qualitative comparison:"
            )
            out.append(
                "  " + " -> ".join(row["action_label"] for row in fallback_trace)
            )
        out.append("")

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
        out += [sep, f"{labels.get(eid, eid).upper()} — {r.get('title', eid)}", sep]
        out.append(r.get("summary", "").strip() or "(no summary)")
        bps = r.get("bullet_points") or []
        if bps:
            out.append("\nBullets:")
            out += [f"  • {b}" for b in bps]
        m_c = r.get("memory_citations") or []
        m_s = r.get("memory_snippets") or []
        n = min(len(m_c), len(m_s), 3)
        if n:
            out.append(f"\nTape & citations — first {n}:")
            for i in range(n):
                snip = m_s[i]
                snip = (snip[:400] + "…") if len(snip) > 400 else snip
                out.append(f"  [{m_c[i]}] {snip}")
        if eid == "hr" and r.get("memo"):
            out.append("\nHR memo:\n" + str(r["memo"]))
        out.append("")

    brief = data.get("current_brief")
    if brief:
        out += [sep, "COMPOSED BRIEF (to CEO — merged from reports)", sep]
        out.append(brief.get("summary", "") or "")
        recs = brief.get("recommendations") or []
        if recs:
            out.append("\nRecommendations:")
            out += [f"  • {x}" for x in recs]
        if brief.get("hr_memo"):
            out.append("\nHR memo (in brief object):\n" + str(brief["hr_memo"]))
        out.append("")
    return "\n".join(out)


def _run(picker, label: str, task: str, use_rag: bool) -> dict:
    """Run one episode with the given picker. Uses inference.run_episode_collect for parity."""
    from inference import run_episode_collect

    return run_episode_collect(task, picker, label, use_rag=use_rag, quiet=True)


def _deterministic_next_action(obs, task: str, required_experts_for_task, CoSAction):
    missing = [e for e in required_experts_for_task(task) if e not in obs.consulted_experts]
    if missing:
        return CoSAction(action_type="consult", expert_id=missing[0])
    if obs.current_brief is None:
        return CoSAction(action_type="summarize")
    return CoSAction(action_type="submit")


def _run_with_policy_trace(
    tok,
    model,
    CoSAction,
    CEOBriefEnvironment,
    required_experts_for_task,
    *,
    label: str,
    task: str,
    use_rag: bool,
    max_new_tokens: int,
    policy_steps: int,
    shaping: str,
) -> dict:
    """Run N model-controlled steps, then fallback only to finish the report.

    This exposes the actual learned policy behavior. The final expert reports are
    still included so you can quote agent answers in the presentation.
    """
    env = CEOBriefEnvironment(shaping=shaping, auto_fill_required=False)
    obs = env.reset(task=task, use_rag=use_rag)
    rewards: list[float] = []
    policy_trace: list[dict[str, Any]] = []
    fallback_trace: list[dict[str, Any]] = []
    required = required_experts_for_task(task)
    model_routed_required: list[str] = []

    for _ in range(max(1, policy_steps)):
        if obs.done:
            break
        before_consulted = list(obs.consulted_experts)
        action, completion = _generate_action_and_text(
            tok, model, CoSAction, obs, max_new_tokens=max_new_tokens
        )
        obs = env.step(action)
        if action.expert_id in required and action.expert_id not in model_routed_required:
            model_routed_required.append(action.expert_id)
        rewards.append(float(obs.reward))
        policy_trace.append(
            {
                "step": obs.step_count,
                "obs_before_consulted": before_consulted,
                "action": action.model_dump(exclude_none=True),
                "action_label": _action_label(action),
                "completion_preview": completion[:400],
                "reward": round(float(obs.reward), 4),
                "done": bool(obs.done),
                "consulted_after": list(obs.consulted_experts),
                "model_routed_required": list(model_routed_required),
            }
        )

    while not obs.done and obs.step_count < obs.max_steps:
        action = _deterministic_next_action(obs, task, required_experts_for_task, CoSAction)
        obs = env.step(action)
        rewards.append(float(obs.reward))
        fallback_trace.append(
            {
                "step": obs.step_count,
                "action": action.model_dump(exclude_none=True),
                "action_label": _action_label(action),
                "reward": round(float(obs.reward), 4),
                "done": bool(obs.done),
                "consulted_after": list(obs.consulted_experts),
            }
        )

    action_sequence = [row["action_label"] for row in policy_trace]
    policy_covered_required = list(model_routed_required)
    data = {
        "task": task,
        "policy_label": label,
        "use_rag": use_rag,
        "shaping": shaping,
        "success": bool((obs.terminal_grader_score or 0.0) >= 0.5),
        "steps": int(obs.step_count),
        "terminal_score": round(float(obs.terminal_grader_score or 0.0), 4),
        "cumulative_reward": round(sum(rewards), 4),
        "step_rewards": [round(x, 4) for x in rewards],
        "final_instruction": obs.instruction,
        "task_difficulty": obs.task_difficulty,
        "max_steps": obs.max_steps,
        "consulted_experts": list(obs.consulted_experts),
        "current_brief": obs.current_brief.model_dump() if obs.current_brief is not None else None,
        "expert_reports": {k: v.model_dump() for k, v in obs.expert_reports.items()},
        "policy_trace": policy_trace,
        "fallback_trace": fallback_trace,
        "policy_evidence": {
            "model_controlled_steps": len(policy_trace),
            "action_sequence": action_sequence,
            "unique_actions": len(set(action_sequence)),
            "covered_required_experts_in_policy_steps": policy_covered_required,
            "required_experts": required,
            "needed_fallback": bool(fallback_trace),
            "fallback_steps": len(fallback_trace),
            "policy_reward_sum": round(sum(row["reward"] for row in policy_trace), 4),
            "total_reward_sum": round(sum(rewards), 4),
        },
    }
    return data


def _write_evidence_artifacts(out_dir: Path, rows: list[dict[str, Any]], task: str) -> None:
    """Save compact JSON/CSV/Markdown and a reward plot for presentation use."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for row in rows:
        ev = row.get("policy_evidence") or {}
        summary.append(
            {
                "policy": row.get("policy_label"),
                "task": row.get("task"),
                "action_sequence": " -> ".join(ev.get("action_sequence") or []),
                "covered_required": ",".join(ev.get("covered_required_experts_in_policy_steps") or []),
                "needed_fallback": ev.get("needed_fallback"),
                "policy_reward_sum": ev.get("policy_reward_sum"),
                "total_reward_sum": ev.get("total_reward_sum"),
                "terminal_score": row.get("terminal_score"),
                "steps": row.get("steps"),
            }
        )
    (out_dir / f"training_evidence__{task}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    csv_lines = [
        "policy,task,action_sequence,covered_required,needed_fallback,policy_reward_sum,total_reward_sum,terminal_score,steps"
    ]
    for item in summary:
        csv_lines.append(
            ",".join(
                json.dumps(str(item[k])) if k in {"action_sequence", "covered_required"} else str(item[k])
                for k in [
                    "policy",
                    "task",
                    "action_sequence",
                    "covered_required",
                    "needed_fallback",
                    "policy_reward_sum",
                    "total_reward_sum",
                    "terminal_score",
                    "steps",
                ]
            )
        )
    (out_dir / f"training_evidence__{task}.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    md = [
        "# Training Evidence",
        "",
        "| Policy | Model-controlled action sequence | Required experts covered | Needed fallback | Policy reward | Final score |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in summary:
        md.append(
            f"| {item['policy']} | `{item['action_sequence']}` | "
            f"{item['covered_required'] or '-'} | {item['needed_fallback']} | "
            f"{item['policy_reward_sum']} | {item['terminal_score']} |"
        )
    (out_dir / f"training_evidence__{task}.md").write_text("\n".join(md), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        for row in rows:
            rewards = row.get("step_rewards") or []
            if not rewards:
                continue
            xs = list(range(1, len(rewards) + 1))
            cum = []
            total = 0.0
            for r in rewards:
                total += float(r)
                cum.append(total)
            plt.plot(xs, cum, marker="o", label=str(row.get("policy_label")))
        plt.xlabel("environment step")
        plt.ylabel("cumulative reward")
        plt.title(f"Training evidence: policy trajectories on {task}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"training_evidence_curve__{task}.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[plot] skipped evidence curve: {e}", flush=True)


# ---------- Main ----------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULTS["task"])
    p.add_argument("--use-rag", action="store_true", default=DEFAULTS["use_rag"])
    p.add_argument("--base-model-id", default=DEFAULTS["base_model_id"])
    p.add_argument("--sft-adapter-id", default=DEFAULTS["sft_adapter_id"])
    p.add_argument("--sft-subfolder", default=DEFAULTS["sft_subfolder"])
    p.add_argument("--grpo-adapter-id", default=DEFAULTS["grpo_adapter_id"])
    p.add_argument("--grpo-subfolder", default=DEFAULTS["grpo_subfolder"])
    p.add_argument("--no-4bit", action="store_true", help="disable 4-bit (fallback to bf16)")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    p.add_argument(
        "--policy-steps",
        type=int,
        default=6,
        help="number of model-controlled CoS steps before deterministic fallback",
    )
    p.add_argument(
        "--shaping",
        choices=("default", "strict"),
        default=DEFAULTS["shaping"],
        help="env reward shaping for evidence run; strict makes lazy policies obvious",
    )
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    p.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/three_llms_text") if Path("/kaggle/working").is_dir() else Path("./three_llms_text"))
    args = p.parse_args()

    use_4bit = not args.no_4bit
    args.out_dir.mkdir(parents=True, exist_ok=True)

    (
        CEOBriefEnvironment,
        oracle_action_for_observation,
        required_experts_for_task,
        CoSAction,
        CoSObservation,
    ) = _load_env()

    runs = [
        ("base_llm", args.base_model_id, None, None),
        ("base_llm+sft", args.base_model_id, (args.sft_adapter_id or None), (args.sft_subfolder or None)),
        ("base_llm+grpo", args.base_model_id, (args.grpo_adapter_id or None), (args.grpo_subfolder or None)),
    ]

    big_sep = "#" * 88
    evidence_rows: list[dict[str, Any]] = []
    for label, base_id, adapter_id, sub in runs:
        print(f"\n{big_sep}\n# RUN: {label}\n# base={base_id}\n# adapter={adapter_id or '(none)'}{(' subfolder='+sub) if sub else ''}\n{big_sep}", flush=True)
        try:
            tok, model = _load_llm(
                base_id,
                adapter_id,
                sub,
                use_4bit=use_4bit,
                hf_token=args.hf_token or None,
            )
        except Exception as e:
            print(f"[load failed for {label}] {e}", flush=True)
            continue

        try:
            data = _run_with_policy_trace(
                tok,
                model,
                CoSAction,
                CEOBriefEnvironment,
                required_experts_for_task,
                label=label,
                task=args.task,
                use_rag=args.use_rag,
                max_new_tokens=args.max_new_tokens,
                policy_steps=args.policy_steps,
                shaping=args.shaping,
            )
        except Exception as e:
            print(f"[episode failed for {label}] {e}", flush=True)
            continue
        finally:
            # Free GPU memory between runs.
            try:
                import torch

                del model
                del tok
                torch.cuda.empty_cache()
            except Exception:
                pass

        print(_format_episode(data))
        out_path = args.out_dir / f"{label}__{args.task}.json"
        out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"[saved] {out_path}", flush=True)
        evidence_rows.append(data)

    if evidence_rows:
        _write_evidence_artifacts(args.out_dir, evidence_rows, args.task)
        print(
            f"[saved] training evidence artifacts under {args.out_dir} "
            f"(json/csv/md/png)",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
