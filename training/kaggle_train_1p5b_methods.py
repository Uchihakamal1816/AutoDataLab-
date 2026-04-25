#!/usr/bin/env python3
"""
Kaggle GPU trainer for AutoDataLab++ CoS routing policies on Qwen2.5-1.5B.

Goal
----
Try many lightweight training methods quickly, evaluate the *actual routing
trajectory*, and keep only methods that improve behavior.

Recommended Kaggle setup
------------------------
Settings:
  - Accelerator: GPU (T4 is enough for Qwen2.5-1.5B QLoRA)
  - Internet: On

Install cell:
    !pip install -q -U "transformers>=4.45,<4.49" "accelerate>=0.33,<1.1" \\
      "peft>=0.13,<0.16" "bitsandbytes>=0.45.0" "trl>=0.11,<0.13" \\
      "datasets>=2.20" "huggingface_hub>=0.24,<1.0" pydantic pandas matplotlib

Run examples:
    # 1) Fast supervised routing policy
    !python3 training/kaggle_train_1p5b_methods.py --method sft --epochs 2

    # 2) Preference optimization from base model
    !python3 training/kaggle_train_1p5b_methods.py --method dpo --epochs 1

    # 3) Best practical path: SFT first, then DPO on top
    !python3 training/kaggle_train_1p5b_methods.py --method sft_then_dpo --sft-epochs 2 --dpo-epochs 1

Outputs
-------
Each run writes an adapter plus evidence files under /kaggle/working/cos_1p5b_runs/<run_name>:
  - adapter/                         PEFT LoRA adapter
  - eval/evidence.json               per-task trajectory evidence
  - eval/evidence.md                 PPT/report-ready table
  - eval/reward_curve.png            cumulative reward plot

Important
---------
This is for *training evidence*. The evaluator runs with:
  - strict reward shaping
  - auto_fill_required=False during model-controlled policy steps

So if a policy tries `summarize -> summarize`, it gets punished and the output
shows the failure clearly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
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


def render_obs(obs: CoSObservation, variant: int = 0) -> str:
    required = required_experts_for_task(obs.task_name)
    missing = [e for e in required if e not in obs.consulted_experts]
    base = {
        "task": obs.task_name,
        "difficulty": obs.task_difficulty,
        "step": obs.step_count,
        "max_steps": obs.max_steps,
        "rag_enabled": obs.rag_enabled,
        "required_experts": required,
        "consulted_experts": list(obs.consulted_experts),
        "missing_required_experts": missing,
        "brief_ready": obs.current_brief is not None,
        "issues": obs.issues[-3:],
    }
    if variant == 0:
        return json.dumps(base, separators=(",", ":"))
    if variant == 1:
        return (
            f"task={obs.task_name}; step={obs.step_count}/{obs.max_steps}; "
            f"required={required}; consulted={obs.consulted_experts}; missing={missing}; "
            f"brief_ready={obs.current_brief is not None}; choose next JSON action."
        )
    return (
        "Decision checklist:\n"
        f"- task: {obs.task_name}\n"
        f"- required experts: {', '.join(required)}\n"
        f"- consulted experts: {', '.join(obs.consulted_experts) or 'none'}\n"
        f"- missing required experts: {', '.join(missing) or 'none'}\n"
        f"- brief ready: {obs.current_brief is not None}\n"
        "Return the single next strict JSON action."
    )


def messages_for(obs: CoSObservation, variant: int, assistant: str | None = None) -> list[dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_obs(obs, variant=variant)},
    ]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def rejected_actions_for(obs: CoSObservation, chosen: CoSAction) -> list[CoSAction]:
    required = required_experts_for_task(obs.task_name)
    missing = [e for e in required if e not in obs.consulted_experts]
    rejects: list[CoSAction] = []
    if missing:
        rejects.extend(
            [
                CoSAction(action_type="summarize"),
                CoSAction(action_type="submit"),
                CoSAction(action_type="noop"),
            ]
        )
        if obs.consulted_experts:
            rejects.append(CoSAction(action_type="consult", expert_id=obs.consulted_experts[0]))
    elif obs.current_brief is None:
        rejects.extend(
            [
                CoSAction(action_type="consult", expert_id=required[0] if required else "analyst"),
                CoSAction(action_type="submit"),
                CoSAction(action_type="noop"),
            ]
        )
    else:
        rejects.extend(
            [
                CoSAction(action_type="summarize"),
                CoSAction(action_type="consult", expert_id=required[0] if required else "analyst"),
                CoSAction(action_type="noop"),
            ]
        )
    chosen_s = action_json(chosen)
    out = []
    seen = {chosen_s}
    for r in rejects:
        s = action_json(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def collect_routing_records(tasks: list[str], rag_modes: list[bool], variants: int = 3) -> tuple[list[dict], list[dict]]:
    """Return (sft_records, dpo_records)."""
    sft_records: list[dict[str, Any]] = []
    dpo_records: list[dict[str, Any]] = []
    for task in tasks:
        for use_rag in rag_modes:
            env = CEOBriefEnvironment(auto_fill_required=False)
            obs = env.reset(task=task, use_rag=use_rag)
            while not obs.done and obs.step_count < obs.max_steps:
                chosen = oracle_action_for_observation(obs)
                chosen_s = action_json(chosen)
                for variant in range(variants):
                    prompt_msgs = messages_for(obs, variant=variant)
                    sft_records.append({"messages": prompt_msgs + [{"role": "assistant", "content": chosen_s}]})
                    for rejected in rejected_actions_for(obs, chosen)[:2]:
                        dpo_records.append(
                            {
                                "prompt": prompt_msgs,
                                "chosen": chosen_s,
                                "rejected": action_json(rejected),
                            }
                        )
                obs = env.step(chosen)
    return sft_records, dpo_records


def parse_action(text: str) -> CoSAction:
    match = JSON_RE.search(text or "")
    if not match:
        return CoSAction(action_type="noop")
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return CoSAction(action_type="noop")
    action_type = payload.get("action_type")
    if action_type not in VALID_ACTIONS:
        return CoSAction(action_type="noop")
    expert_id = payload.get("expert_id")
    if expert_id is not None and expert_id not in VALID_EXPERTS:
        expert_id = None
    return CoSAction(action_type=action_type, expert_id=expert_id)


def action_label(action: CoSAction) -> str:
    if action.action_type in {"consult", "ask"}:
        return f"{action.action_type}:{action.expert_id or 'null'}"
    return action.action_type


@dataclass
class LoadedModel:
    tokenizer: Any
    model: Any


def load_base_with_lora(model_id: str, use_4bit: bool, hf_token: str | None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    return LoadedModel(tok, model)


def format_chat(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    text = ""
    for m in messages:
        text += f"{m['role'].upper()}:\n{m['content']}\n"
    if add_generation_prompt:
        text += "ASSISTANT:\n"
    return text


class SFTTorchDataset:
    def __init__(self, records: list[dict], tokenizer, max_length: int = 1024):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        messages = rec["messages"]
        prompt = format_chat(self.tokenizer, messages[:-1], add_generation_prompt=True)
        answer = messages[-1]["content"] + (self.tokenizer.eos_token or "")
        full = prompt + answer
        enc = self.tokenizer(full, truncation=True, max_length=self.max_length)
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
        labels = list(enc["input_ids"])
        mask_n = min(len(prompt_ids), len(labels))
        labels[:mask_n] = [-100] * mask_n
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def train_sft(args, out_dir: Path) -> Path:
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    sft_records, _dpo_records = collect_routing_records(TASKS, [False, True], variants=args.variants)
    if args.max_train_examples:
        sft_records = sft_records[: args.max_train_examples]
    print(f"[data] SFT examples={len(sft_records)}", flush=True)

    loaded = load_base_with_lora(args.model_id, use_4bit=not args.no_4bit, hf_token=args.hf_token or None)
    train_ds = SFTTorchDataset(sft_records, loaded.tokenizer, max_length=args.max_length)
    collator = DataCollatorForSeq2Seq(loaded.tokenizer, model=loaded.model, padding=True)
    train_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=True,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=loaded.model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()
    adapter_dir = out_dir / "adapter"
    loaded.model.save_pretrained(adapter_dir)
    loaded.tokenizer.save_pretrained(adapter_dir)
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir


def train_dpo(args, out_dir: Path, base_adapter: Path | None = None) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    _sft_records, dpo_records = collect_routing_records(TASKS, [False, True], variants=args.variants)
    if args.max_train_examples:
        dpo_records = dpo_records[: args.max_train_examples]
    print(f"[data] DPO pairs={len(dpo_records)}", flush=True)

    tok_source = str(base_adapter) if base_adapter else args.model_id
    tok = AutoTokenizer.from_pretrained(tok_source, token=args.hf_token or None)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def to_row(rec: dict[str, Any]) -> dict[str, str]:
        return {
            "prompt": format_chat(tok, rec["prompt"], add_generation_prompt=True),
            "chosen": rec["chosen"] + (tok.eos_token or ""),
            "rejected": rec["rejected"] + (tok.eos_token or ""),
        }

    ds = Dataset.from_list([to_row(r) for r in dpo_records])

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
    model.config.use_cache = False
    if base_adapter:
        model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, str(base_adapter), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model.add_adapter(lora_cfg)

    cfg = DPOConfig(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.dpo_epochs if args.method == "sft_then_dpo" else args.epochs,
        learning_rate=args.dpo_lr,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=True,
        beta=args.dpo_beta,
        max_length=args.max_length,
        max_prompt_length=min(768, args.max_length - 64),
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    del trainer, model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir


def load_for_eval(model_id: str, adapter_dir: Path, use_4bit: bool, hf_token: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(adapter_dir, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def generate_action(tok, model, obs: CoSObservation, max_new_tokens: int = 48) -> tuple[CoSAction, str]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_obs(obs, variant=0)},
    ]
    prompt = format_chat(tok, messages, add_generation_prompt=True)
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


def evaluate_adapter(args, adapter_dir: Path, eval_dir: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt
    import torch

    tok, model = load_for_eval(args.model_id, adapter_dir, use_4bit=not args.no_4bit, hf_token=args.hf_token or None)
    rows: list[dict[str, Any]] = []
    for task in args.eval_tasks.split(","):
        task = task.strip()
        if not task:
            continue
        env = CEOBriefEnvironment(shaping="strict", auto_fill_required=False)
        obs = env.reset(task=task, use_rag=False)
        rewards: list[float] = []
        trace: list[dict[str, Any]] = []
        routed: list[str] = []
        for _ in range(args.policy_steps):
            if obs.done:
                break
            action, completion = generate_action(tok, model, obs, max_new_tokens=args.eval_new_tokens)
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
        fallback: list[str] = []
        while not obs.done and obs.step_count < obs.max_steps:
            act = deterministic_next(obs, task)
            obs = env.step(act)
            rewards.append(float(obs.reward))
            fallback.append(action_label(act))
        rows.append(
            {
                "task": task,
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
        "# AutoDataLab++ 1.5B Training Evidence",
        "",
        "| Task | Action sequence | Routed required experts | Needed fallback | Policy reward | Terminal |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['task']} | `{' -> '.join(row['action_sequence'])}` | "
            f"{', '.join(row['model_routed_required']) or '-'} | {row['needed_fallback']} | "
            f"{row['policy_reward']} | {row['terminal_score']} |"
        )
    (eval_dir / "evidence.md").write_text("\n".join(md), encoding="utf-8")
    plt.figure(figsize=(8, 4))
    for row in rows:
        cum = []
        total = 0.0
        # plot policy-step rewards only, because that is the learning evidence
        for t in row["trace"]:
            total += float(t["reward"])
            cum.append(total)
        if cum:
            plt.plot(range(1, len(cum) + 1), cum, marker="o", label=row["task"])
    plt.title("Model-controlled policy reward (strict shaping)")
    plt.xlabel("policy step")
    plt.ylabel("cumulative reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "reward_curve.png", dpi=160)
    plt.close()
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("sft", "dpo", "sft_then_dpo"), required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out-root", type=Path, default=Path("/kaggle/working/cos_1p5b_runs") if Path("/kaggle/working").is_dir() else Path("cos_1p5b_runs"))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--sft-epochs", type=float, default=2.0)
    ap.add_argument("--dpo-epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dpo-lr", type=float, default=5e-5)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--max-train-examples", type=int, default=0)
    ap.add_argument("--policy-steps", type=int, default=6)
    ap.add_argument("--eval-new-tokens", type=int, default=48)
    ap.add_argument("--eval-tasks", default="expert_brief,risk_brief,crisis_brief")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.max_train_examples == 0:
        args.max_train_examples = None
    set_seed(args.seed)

    run_name = args.run_name or f"{args.method}_qwen15b_seed{args.seed}"
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"[run] {run_name} -> {out_dir}", flush=True)

    if args.method == "sft":
        adapter = train_sft(args, out_dir)
    elif args.method == "dpo":
        adapter = train_dpo(args, out_dir)
    else:
        sft_dir = out_dir / "sft_stage"
        dpo_dir = out_dir / "dpo_stage"
        old_epochs = args.epochs
        args.epochs = args.sft_epochs
        sft_adapter = train_sft(args, sft_dir)
        args.epochs = old_epochs
        adapter = train_dpo(args, dpo_dir, base_adapter=sft_adapter)
        final_adapter = out_dir / "adapter"
        if final_adapter.exists():
            import shutil

            shutil.rmtree(final_adapter)
        import shutil

        shutil.copytree(adapter, final_adapter)
        adapter = final_adapter

    evidence = evaluate_adapter(args, adapter, out_dir / "eval")
    print("\n=== EVIDENCE SUMMARY ===", flush=True)
    for row in evidence:
        print(
            f"{row['task']}: {' -> '.join(row['action_sequence'])} | "
            f"routed={row['model_routed_required']} | fallback={row['needed_fallback']} | "
            f"policy_reward={row['policy_reward']} terminal={row['terminal_score']}",
            flush=True,
        )
    print(f"\n[adapter] {adapter}", flush=True)
    print(f"[eval] {out_dir / 'eval'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
