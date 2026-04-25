#!/usr/bin/env python3
"""
Kaggle GPU RL trainer for AutoDataLab++ CoS routing on Qwen2.5-1.5B.

This script is intentionally practical for hackathon iteration:

Methods
-------
  grpo
      Group-relative preference optimization over candidate JSON actions.
      For each environment state, score multiple candidate actions, normalize
      rewards within the group, and update LoRA weights toward high-advantage
      actions.

  grpo_rlvr
      Same GRPO update, but the reward is "verifiable" from the environment:
      correct next oracle action + strict environment step reward + penalties
      for premature summarize/submit/repeats. This is the best RLVR-style
      option for our setup.

  ppo
      Lightweight PPO-style clipped policy update over candidate JSON actions.
      It uses old/new action log-prob ratios and clipped advantages. This is
      not a full value-head PPO trainer; it is a compact bandit-PPO variant
      tailored for CoS action routing.

Recommended flow
----------------
1. Train SFT/DPO first with:
     training/kaggle_train_1p5b_methods.py --method sft_then_dpo ...

2. Then run RL from that adapter:
     !python3 training/kaggle_rl_1p5b_methods.py \\
       --method grpo_rlvr \\
       --init-adapter /kaggle/working/cos_1p5b_runs/qwen15b_sft_then_dpo_v1/adapter \\
       --run-name qwen15b_grpo_rlvr_v1

3. Check:
     /kaggle/working/cos_1p5b_rl_runs/<run-name>/eval/evidence.md

If trajectory is:
  analyst -> finance -> strategy -> hr -> summarize -> submit
with fallback=False, keep it. Otherwise skip and try another method/seed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ceo_brief_env.environment import CEOBriefEnvironment, oracle_action_for_observation, required_experts_for_task
from ceo_brief_env.models import CoSAction, CoSObservation

TASKS = ["easy_brief", "medium_brief", "hard_brief", "expert_brief", "risk_brief", "crisis_brief"]
VALID_ACTIONS = {"consult", "ask", "summarize", "submit", "noop"}
VALID_EXPERTS = {"analyst", "finance", "hr", "strategy"}
JSON_RE = re.compile(r"\{[^{}]*\}", re.S)

SYSTEM_PROMPT = (
    "You are the Chief of Staff in AutoDataLab++. You orchestrate four specialists: "
    "analyst, finance, strategy, hr. Reply with STRICT JSON only.\n"
    'Schema: {"action_type": one of [consult, ask, summarize, submit, noop], '
    '"expert_id": one of [analyst, finance, hr, strategy] or null}.\n'
    "Rules: consult each required expert exactly once when required, then summarize, then submit. "
    "Never summarize while required experts are missing. Never repeat summarize."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def action_json(action: CoSAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"), sort_keys=True)


def action_label(action: CoSAction) -> str:
    if action.action_type in {"consult", "ask"}:
        return f"{action.action_type}:{action.expert_id or 'null'}"
    return action.action_type


def parse_action(text: str) -> CoSAction:
    m = JSON_RE.search(text or "")
    if not m:
        return CoSAction(action_type="noop")
    try:
        payload = json.loads(m.group(0))
    except Exception:
        return CoSAction(action_type="noop")
    action_type = payload.get("action_type")
    if action_type not in VALID_ACTIONS:
        return CoSAction(action_type="noop")
    expert_id = payload.get("expert_id")
    if expert_id is not None and expert_id not in VALID_EXPERTS:
        expert_id = None
    return CoSAction(action_type=action_type, expert_id=expert_id)


def render_obs(obs: CoSObservation, variant: int = 0) -> str:
    required = required_experts_for_task(obs.task_name)
    missing = [e for e in required if e not in obs.consulted_experts]
    if variant == 0:
        return json.dumps(
            {
                "task": obs.task_name,
                "step": obs.step_count,
                "max_steps": obs.max_steps,
                "required_experts": required,
                "consulted_experts": list(obs.consulted_experts),
                "missing_required_experts": missing,
                "brief_ready": obs.current_brief is not None,
                "rag_enabled": obs.rag_enabled,
            },
            separators=(",", ":"),
        )
    return (
        f"task={obs.task_name}; step={obs.step_count}/{obs.max_steps}; "
        f"required={required}; consulted={obs.consulted_experts}; missing={missing}; "
        f"brief_ready={obs.current_brief is not None}; return next strict JSON action."
    )


def format_chat(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    text = ""
    for m in messages:
        text += f"{m['role'].upper()}:\n{m['content']}\n"
    if add_generation_prompt:
        text += "ASSISTANT:\n"
    return text


def prompt_for_obs(tokenizer, obs: CoSObservation, variant: int = 0) -> str:
    return format_chat(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_obs(obs, variant=variant)},
        ],
        add_generation_prompt=True,
    )


def candidate_actions(obs: CoSObservation) -> list[CoSAction]:
    """Fixed action set for candidate-action RL."""
    required = required_experts_for_task(obs.task_name)
    out: list[CoSAction] = [CoSAction(action_type="consult", expert_id=e) for e in ["analyst", "finance", "strategy", "hr"]]
    out += [
        CoSAction(action_type="summarize"),
        CoSAction(action_type="submit"),
        CoSAction(action_type="noop"),
    ]
    # Keep action set stable, but put oracle-ish actions first for easier inspection.
    oracle = oracle_action_for_observation(obs)
    oracle_s = action_json(oracle)
    ordered = [oracle] + [a for a in out if action_json(a) != oracle_s]
    # If task does not require strategy, strategy remains a candidate but will be
    # lower reward; useful for learning "don't over-consult".
    return ordered


def oracle_reward(obs: CoSObservation, action: CoSAction) -> float:
    oracle = oracle_action_for_observation(obs)
    if action.model_dump(exclude_none=True) == oracle.model_dump(exclude_none=True):
        return 1.0
    required = required_experts_for_task(obs.task_name)
    missing = [e for e in required if e not in obs.consulted_experts]
    if action.action_type == "consult" and action.expert_id in missing:
        return 0.65
    if missing and action.action_type in {"summarize", "submit"}:
        return -1.0
    if action.action_type == "noop":
        return -0.8
    if action.action_type == "consult" and action.expert_id in obs.consulted_experts:
        return -0.6
    return -0.2


def env_step_reward(obs: CoSObservation, action: CoSAction) -> float:
    """Verifiable one-step reward from strict env, without auto-fill."""
    env = CEOBriefEnvironment(shaping="strict", auto_fill_required=False)
    sim = env.reset(task=obs.task_name, use_rag=obs.rag_enabled)
    # Recreate state approximately by replaying history. Histories are action JSONs.
    for h in obs.history:
        try:
            payload = json.loads(h)
            sim = env.step(CoSAction.model_validate(payload))
        except Exception:
            pass
    sim = env.step(action)
    return float(sim.reward)


def rlvr_reward(obs: CoSObservation, action: CoSAction) -> float:
    """Combined RLVR reward: oracle correctness + strict env signal."""
    return oracle_reward(obs, action) + 2.0 * env_step_reward(obs, action)


def collect_states(tasks: list[str], rag_modes: list[bool], variants: int) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for task in tasks:
        for use_rag in rag_modes:
            env = CEOBriefEnvironment(auto_fill_required=False)
            obs = env.reset(task=task, use_rag=use_rag)
            while not obs.done and obs.step_count < obs.max_steps:
                for variant in range(variants):
                    states.append(
                        {
                            "task": task,
                            "use_rag": use_rag,
                            "variant": variant,
                            "obs": obs,
                        }
                    )
                obs = env.step(oracle_action_for_observation(obs))
    random.shuffle(states)
    return states


@dataclass
class Loaded:
    tokenizer: Any
    model: Any


def load_policy_model(args) -> Loaded:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok_source = args.init_adapter if args.init_adapter else args.model_id
    tok = AutoTokenizer.from_pretrained(tok_source, token=args.hf_token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = None
    if not args.no_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.hf_token or None,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    model.resize_token_embeddings(len(tok))
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, cfg)
    model.config.use_cache = False
    return Loaded(tok, model)


def sequence_logprob(model, tokenizer, prompt: str, completion: str, max_length: int):
    """Mean log-prob of completion tokens under CausalLM."""
    import torch

    full = prompt + completion + (tokenizer.eos_token or "")
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    prompt_len = min(prompt_ids["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    out = model(**enc)
    logits = out.logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target.ne(-100)
    logp = torch.log_softmax(logits, dim=-1)
    safe_target = target.masked_fill(~mask, 0)
    tok_logp = logp.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum() / mask.sum().clamp_min(1)


def train_candidate_rl(args, out_dir: Path) -> Path:
    import torch

    loaded = load_policy_model(args)
    model = loaded.model
    tok = loaded.tokenizer
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    states = collect_states(TASKS, [False, True], args.variants)
    if args.max_train_states:
        states = states[: args.max_train_states]
    print(f"[data] states={len(states)} method={args.method}", flush=True)

    metrics: list[dict[str, float]] = []
    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(states)
        for i, item in enumerate(states, start=1):
            obs: CoSObservation = item["obs"]
            prompt = prompt_for_obs(tok, obs, variant=int(item["variant"]))
            cands = candidate_actions(obs)
            rewards = []
            for action in cands:
                r = rlvr_reward(obs, action) if args.method == "grpo_rlvr" else oracle_reward(obs, action)
                rewards.append(float(r))
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=model.device)

            logps = []
            with torch.no_grad():
                old_logps = []
                for action in cands:
                    old_logps.append(sequence_logprob(model, tok, prompt, action_json(action), args.max_length).detach())
                old_logps_t = torch.stack(old_logps)

            for action in cands:
                logps.append(sequence_logprob(model, tok, prompt, action_json(action), args.max_length))
            logps_t = torch.stack(logps)

            if args.method in {"grpo", "grpo_rlvr"}:
                adv = (rewards_t - rewards_t.mean()) / rewards_t.std().clamp_min(1e-4)
                # Group-relative update: improve high-advantage candidates and
                # suppress low-advantage candidates. The detached log-softmax
                # normalization keeps this stable on tiny candidate groups.
                log_policy = torch.log_softmax(logps_t, dim=0)
                loss = -(adv.detach() * log_policy).mean()
            else:
                # PPO-style clipped update over candidate actions.
                adv = (rewards_t - rewards_t.mean()) / rewards_t.std().clamp_min(1e-4)
                ratio = torch.exp(logps_t - old_logps_t)
                clipped = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip)
                loss = -torch.minimum(ratio * adv.detach(), clipped * adv.detach()).mean()

            if args.sft_anchor > 0:
                oracle = oracle_action_for_observation(obs)
                oracle_s = action_json(oracle)
                oracle_logp = sequence_logprob(model, tok, prompt, oracle_s, args.max_length)
                # Positive anchor: keep the known-good routing behavior while
                # RL nudges preferences. This is critical when continuing from
                # an already-good SFT/DPO adapter.
                loss = loss - args.sft_anchor * oracle_logp

            loss = loss / args.grad_accum
            loss.backward()
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    best_idx = int(torch.argmax(logps_t.detach()).item())
                    reward_best = float(rewards_t[best_idx].item())
                    oracle_idx = int(torch.argmax(rewards_t).item())
                    chosen_ok = best_idx == oracle_idx
                    row = {
                        "step": global_step,
                        "epoch": float(epoch),
                        "loss": float(loss.detach().cpu().item() * args.grad_accum),
                        "mean_reward": float(rewards_t.mean().detach().cpu().item()),
                        "best_reward": reward_best,
                        "chosen_ok": float(chosen_ok),
                    }
                    metrics.append(row)
                    print(
                        f"[train] step={global_step} loss={row['loss']:.4f} "
                        f"mean_reward={row['mean_reward']:.3f} best_reward={row['best_reward']:.3f} "
                        f"chosen_ok={chosen_ok}",
                        flush=True,
                    )
        optimizer.zero_grad(set_to_none=True)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        if metrics:
            xs = [m["step"] for m in metrics]
            ys = [m["best_reward"] for m in metrics]
            ok = [m["chosen_ok"] for m in metrics]
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, marker="o", label="best candidate reward")
            plt.plot(xs, ok, marker=".", label="policy picks max-reward candidate")
            plt.xlabel("optimizer step")
            plt.ylabel("metric")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "train_curve.png", dpi=160)
            plt.close()
    except Exception as e:
        print(f"[plot] train curve skipped: {e}", flush=True)
    del loaded, model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir


def load_for_eval(args, adapter_dir: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(adapter_dir, token=args.hf_token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = None
    if not args.no_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.hf_token or None,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def generate_action(tok, model, obs: CoSObservation, max_new_tokens: int) -> tuple[CoSAction, str]:
    import torch

    prompt = prompt_for_obs(tok, obs, variant=0)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_action(text), text.strip()


def deterministic_next(obs: CoSObservation, task: str) -> CoSAction:
    missing = [e for e in required_experts_for_task(task) if e not in obs.consulted_experts]
    if missing:
        return CoSAction(action_type="consult", expert_id=missing[0])
    if obs.current_brief is None:
        return CoSAction(action_type="summarize")
    return CoSAction(action_type="submit")


def evaluate(args, adapter_dir: Path, eval_dir: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt
    import torch

    tok, model = load_for_eval(args, adapter_dir)
    rows = []
    rag_modes = [s.strip().lower() in {"1", "true", "yes", "rag"} for s in args.eval_rag_modes.split(",")]
    for use_rag in rag_modes:
        for task in [t.strip() for t in args.eval_tasks.split(",") if t.strip()]:
            env = CEOBriefEnvironment(shaping="strict", auto_fill_required=False)
            obs = env.reset(task=task, use_rag=use_rag)
            trace = []
            rewards = []
            routed = []
            for _ in range(args.policy_steps):
                if obs.done:
                    break
                action, completion = generate_action(tok, model, obs, args.eval_new_tokens)
                obs = env.step(action)
                rewards.append(float(obs.reward))
                if action.expert_id in required_experts_for_task(task) and action.expert_id not in routed:
                    routed.append(action.expert_id)
                trace.append(
                    {
                        "step": obs.step_count,
                        "action": action.model_dump(exclude_none=True),
                        "action_label": action_label(action),
                        "completion_preview": completion[:300],
                        "reward": round(float(obs.reward), 4),
                        "consulted_after": list(obs.consulted_experts),
                        "model_routed_required": list(routed),
                    }
                )
            fallback = []
            while not obs.done and obs.step_count < obs.max_steps:
                act = deterministic_next(obs, task)
                obs = env.step(act)
                rewards.append(float(obs.reward))
                fallback.append(action_label(act))
            rows.append(
                {
                    "task": task,
                    "rag": bool(use_rag),
                    "action_sequence": [t["action_label"] for t in trace],
                    "model_routed_required": routed,
                    "required_experts": required_experts_for_task(task),
                    "fallback": fallback,
                    "needed_fallback": bool(fallback),
                    "policy_reward": round(sum(t["reward"] for t in trace), 4),
                    "total_reward": round(sum(rewards), 4),
                    "terminal_score": round(float(obs.terminal_grader_score or 0.0), 4),
                    "trace": trace,
                }
            )
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "evidence.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    md = [
        "# AutoDataLab++ RL Evidence",
        "",
        "| Task | RAG | Action sequence | Routed required experts | Needed fallback | Policy reward | Terminal |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['task']} | {row['rag']} | `{' -> '.join(row['action_sequence'])}` | "
            f"{', '.join(row['model_routed_required']) or '-'} | {row['needed_fallback']} | "
            f"{row['policy_reward']} | {row['terminal_score']} |"
        )
    (eval_dir / "evidence.md").write_text("\n".join(md), encoding="utf-8")
    plt.figure(figsize=(8, 4))
    for row in rows:
        total = 0.0
        curve = []
        for t in row["trace"]:
            total += float(t["reward"])
            curve.append(total)
        if curve:
            plt.plot(range(1, len(curve) + 1), curve, marker="o", label=f"{row['task']} rag={row['rag']}")
    plt.title("RL policy reward on model-controlled steps")
    plt.xlabel("policy step")
    plt.ylabel("cumulative strict reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "reward_curve.png", dpi=160)
    plt.close()
    del tok, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def evidence_score(rows: list[dict[str, Any]]) -> float:
    """Single scalar for quick continue/skip decisions."""
    if not rows:
        return -999.0
    vals = []
    for row in rows:
        required = set(row.get("required_experts") or [])
        routed = set(row.get("model_routed_required") or [])
        coverage = len(required & routed) / max(len(required), 1)
        no_fallback = 1.0 if not row.get("needed_fallback") else 0.0
        policy_reward = float(row.get("policy_reward") or 0.0)
        vals.append(coverage + no_fallback + 0.1 * policy_reward)
    return sum(vals) / len(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("grpo", "grpo_rlvr", "ppo"), required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--init-adapter", default="", help="optional SFT/DPO adapter directory or HF repo")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out-root", type=Path, default=Path("/kaggle/working/cos_1p5b_rl_runs") if Path("/kaggle/working").is_dir() else Path("cos_1p5b_rl_runs"))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="conservative default; higher LR can collapse a good SFT/DPO adapter to noop",
    )
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--max-train-states", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--ppo-clip", type=float, default=0.2)
    ap.add_argument(
        "--sft-anchor",
        type=float,
        default=0.2,
        help="oracle logprob anchor; keep >0 when continuing from a good SFT/DPO adapter",
    )
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--policy-steps", type=int, default=6)
    ap.add_argument("--eval-new-tokens", type=int, default=48)
    ap.add_argument("--eval-tasks", default="expert_brief,risk_brief,crisis_brief")
    ap.add_argument("--eval-rag-modes", default="false,true", help="comma list: false,true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.max_train_states == 0:
        args.max_train_states = None
    set_seed(args.seed)

    run_name = args.run_name or f"{args.method}_qwen15b_seed{args.seed}"
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"[run] {run_name} -> {out_dir}", flush=True)

    before_rows = []
    if args.init_adapter:
        print("[baseline] evaluating init adapter before RL...", flush=True)
        before_rows = evaluate(args, Path(args.init_adapter), out_dir / "eval_before")
        print(f"[baseline] evidence_score={evidence_score(before_rows):.4f}", flush=True)

    adapter_dir = train_candidate_rl(args, out_dir)
    rows = evaluate(args, adapter_dir, out_dir / "eval")
    after_score = evidence_score(rows)
    before_score = evidence_score(before_rows) if before_rows else None
    print("\n=== RL EVIDENCE SUMMARY ===", flush=True)
    for row in rows:
        print(
            f"{row['task']}: {' -> '.join(row['action_sequence'])} | "
            f"routed={row['model_routed_required']} | fallback={row['needed_fallback']} | "
            f"policy_reward={row['policy_reward']} terminal={row['terminal_score']}",
            flush=True,
        )
    print(f"\n[evidence_score] after={after_score:.4f}", flush=True)
    if before_score is not None:
        print(f"[evidence_score] before={before_score:.4f}", flush=True)
        if after_score + 1e-6 < before_score:
            print(
                "[decision] SKIP this RL adapter: it regressed versus the init adapter.",
                flush=True,
            )
        else:
            print(
                "[decision] KEEP this RL adapter: it matched or improved the init adapter.",
                flush=True,
            )
    print(f"\n[adapter] {adapter_dir}", flush=True)
    print(f"[eval] {out_dir / 'eval'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
