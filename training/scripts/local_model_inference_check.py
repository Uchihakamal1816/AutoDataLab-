#!/usr/bin/env python3
"""
Smoke test: load Qwen (or any causal LM) from a local folder and run one generation.

Usage:
    python3 training/local_model_inference_check.py --model-dir ./model
    python3 training/local_model_inference_check.py --model-dir /path/to/model --device cpu
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        default="model",
        help="Path to local folder with config + weights (e.g. ./model)",
    )
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    did = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "auto":
        dev = did
    else:
        dev = args.device

    print(f"[load] {args.model_dir!r} | device={dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp16 on GPU is enough for 1.5B; CPU can stay fp32
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if dev == "cpu":
        model = model.to("cpu")
    else:
        model = model.to("cuda")
    model.eval()

    messages = [
        {"role": "system", "content": "You reply briefly."},
        {"role": "user", "content": "Say the capital of France in one line."},
    ]
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = messages[0]["content"] + "\n" + messages[1]["content"]

    inputs = tok(prompt, return_tensors="pt")
    if dev == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    text = tok.decode(new_tokens, skip_special_tokens=True)

    print("\n[ok] generation:\n" + text.strip() + "\n", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OSError as e:
        if "local_files_only" in str(e) or e.errno in (2, 20):
            print(
                "Hint: run from the parent of `model/`, e.g.:\n"
                "  cd autodatalab-plus && python3 training/local_model_inference_check.py --model-dir model",
                file=sys.stderr,
            )
        raise
