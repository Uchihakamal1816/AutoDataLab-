"""
Evaluate one or more LoRA adapter repos against the AutoDataLab++ environment
and pick the best. Designed for a Colab/T4/L4 GPU.

Usage (in Colab/Notebook or shell):

    python3 training/eval_lora_adapters.py \
        --base Qwen/Qwen2.5-14B-Instruct \
        --adapters \
            kartik1230/run-A \
            kartik1230/run-B \
            kartik1230/run-C \
        --tasks easy_brief medium_brief hard_brief expert_brief \
        --episodes 3 \
        --hf-token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --out training/eval_lora_results.json

Notes:
- Each adapter is loaded ONE AT A TIME on top of the same base model in 4-bit.
- For each (adapter, task, episode) we let the LLM emit ONLY the FIRST action,
  then a deterministic continuation completes the episode (consult missing
  required experts -> summarize -> submit). Same scheme as the GRPO notebook.
- The score is the env's terminal_grader_score (verifiable reward).
- Results are saved as JSON; a summary table is printed.

Pick whichever adapter has the highest mean score across tasks.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ceo_brief_env.environment import CEOBriefEnvironment, required_experts_for_task  # noqa: E402
from ceo_brief_env.models import CoSAction  # noqa: E402

SYSTEM_PROMPT = (
    "You are the Chief of Staff in AutoDataLab++. You orchestrate four specialists: "
    "analyst, finance, strategy, hr. Reply with STRICT JSON only.\n"
    "Schema: {\"action_type\": one of [consult, ask, summarize, submit, noop], "
    "\"expert_id\": one of [analyst, finance, hr, strategy] or null}.\n"
    "Rules: consult each required expert at most once -> summarize -> submit."
)

VALID_ACTIONS = {"consult", "ask", "summarize", "submit", "noop"}
VALID_EXPERTS = {"analyst", "finance", "hr", "strategy"}
_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def render_obs(obs) -> str:
    return (
        f"task={obs.task_name} step={obs.step_count}/{obs.max_steps} "
        f"rag={obs.rag_enabled} "
        f"consulted={obs.consulted_experts} "
        f"brief_done={obs.current_brief is not None} "
        f"available={obs.available_experts}"
    )


def parse_action(text: str) -> CoSAction:
    m = _JSON_RE.search(text or "")
    if not m:
        return CoSAction(action_type="noop")
    try:
        a = json.loads(m.group(0))
    except Exception:
        return CoSAction(action_type="noop")
    at = a.get("action_type")
    if at not in VALID_ACTIONS:
        return CoSAction(action_type="noop")
    eid = a.get("expert_id")
    if eid is not None and eid not in VALID_EXPERTS:
        eid = None
    return CoSAction(action_type=at, expert_id=eid)


def deterministic_continuation(env: CEOBriefEnvironment, obs, task: str) -> float:
    while not obs.done and obs.step_count < obs.max_steps:
        missing = [e for e in required_experts_for_task(task) if e not in obs.consulted_experts]
        if missing:
            act = CoSAction(action_type="consult", expert_id=missing[0])
        elif obs.current_brief is None:
            act = CoSAction(action_type="summarize")
        else:
            act = CoSAction(action_type="submit")
        obs = env.step(act)
    return float(obs.terminal_grader_score or 0.0)


def evaluate_adapter(model, tok, adapter_repo: str, tasks: list[str], episodes: int, hf_token: str | None, subfolder: str | None = None) -> dict[str, Any]:
    import torch
    from peft import PeftModel

    print(f"[eval] loading adapter: {adapter_repo} subfolder={subfolder!r}")
    kwargs: dict[str, Any] = {"token": hf_token}
    if subfolder:
        kwargs["subfolder"] = subfolder
    peft_model = PeftModel.from_pretrained(model, adapter_repo, **kwargs)
    peft_model.eval()

    out: dict[str, Any] = {"adapter": adapter_repo, "per_task": {}, "raw": []}
    for task in tasks:
        scores: list[float] = []
        for ep in range(episodes):
            env = CEOBriefEnvironment()
            obs = env.reset(task=task, use_rag=False)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_obs(obs)},
            ]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(peft_model.device)
            with torch.no_grad():
                gen = peft_model.generate(
                    **ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id,
                )
            comp = tok.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
            action = parse_action(comp)
            obs = env.step(action)
            term = deterministic_continuation(env, obs, task)
            scores.append(term)
            out["raw"].append({"task": task, "episode": ep, "completion": comp.strip()[:200], "score": term})
        mean = round(sum(scores) / len(scores), 4)
        out["per_task"][task] = {"scores": scores, "mean": mean}
        print(f"[eval] {adapter_repo} | {task} | mean={mean} | scores={scores}")

    out["mean_overall"] = round(
        sum(v["mean"] for v in out["per_task"].values()) / max(1, len(out["per_task"])), 4
    )

    del peft_model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF id of the base model used during AutoTrain (e.g. Qwen/Qwen2.5-14B-Instruct)")
    ap.add_argument("--adapters", nargs="+", required=True, help="HF repos of LoRA adapters to evaluate")
    ap.add_argument("--tasks", nargs="+", default=["easy_brief", "medium_brief", "hard_brief", "expert_brief"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN env var)")
    ap.add_argument("--out", default="training/eval_lora_results.json")
    ap.add_argument(
        "--adapter-subfolder",
        default=None,
        help="PEFT subfolder inside the adapter repo, e.g. 'final' when weights live in repo/final/",
    )
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit (use only on big GPUs)")
    args = ap.parse_args()

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[eval] loading base model: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = None
    if not args.no_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        token=hf_token,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
    )
    base_model.eval()

    results: list[dict[str, Any]] = []
    for adapter in args.adapters:
        try:
            res = evaluate_adapter(
                base_model, tok, adapter, args.tasks, args.episodes, hf_token, subfolder=args.adapter_subfolder
            )
        except Exception as e:
            print(f"[eval] FAILED {adapter}: {e}")
            res = {"adapter": adapter, "error": str(e), "mean_overall": -1.0, "per_task": {}}
        results.append(res)

    results.sort(key=lambda r: r.get("mean_overall", -1.0), reverse=True)

    print("\n=== RANKING (higher is better) ===")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['adapter']}  mean_overall={r.get('mean_overall')}  per_task={ {k: v.get('mean') for k, v in r.get('per_task', {}).items()} }")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "base": args.base,
                "adapter_subfolder": args.adapter_subfolder,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n[eval] saved -> {out_path}")
    if results:
        print(f"[eval] WINNER: {results[0]['adapter']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
