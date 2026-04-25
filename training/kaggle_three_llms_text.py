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
REPO = Path(__file__).resolve().parents[1]
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


def _make_llm_picker(tok, model, CoSAction, max_new_tokens: int = 96):
    """Returns a CoS picker that asks the LLM, parses JSON, falls back to noop."""
    import torch

    @torch.no_grad()
    def picker(obs):
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
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse_action(text, CoSAction)

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

        picker = _make_llm_picker(tok, model, CoSAction, max_new_tokens=args.max_new_tokens)
        try:
            data = _run(picker, label, args.task, args.use_rag)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
