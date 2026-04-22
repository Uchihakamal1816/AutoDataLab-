#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

API_BASE_URL = os.getenv('API_BASE_URL', 'https://router.huggingface.co/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct')
API_KEY = os.getenv('API_KEY') or os.getenv('HF_TOKEN') or ''
BENCHMARK = 'autodatalab_plus'
TASKS = [t.strip() for t in os.getenv('AUTODATALAB_PLUS_TASKS', 'easy_brief,medium_brief,hard_brief').split(',') if t.strip()]

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ceo_brief_env.environment import CEOBriefEnvironment, oracle_action_for_observation
from ceo_brief_env.models import CoSAction


def _bool_str(v: bool) -> str:
    return str(bool(v)).lower()


def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] task={task} env={env} model={model}', flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else 'null'
    print(f'[STEP] step={step} action={action} reward={reward:.2f} done={_bool_str(done)} error={err}', flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ','.join(f'{r:.2f}' for r in rewards)
    print(f'[END] success={_bool_str(success)} steps={steps} score={score:.3f} rewards={rewards_str}', flush=True)


def _action_str(action: CoSAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(',', ':'), sort_keys=True)


def _single_baseline(obs) -> CoSAction:
    if 'analyst' not in obs.consulted_experts:
        return CoSAction(action_type='consult', expert_id='analyst')
    if 'hr' not in obs.consulted_experts:
        return CoSAction(action_type='consult', expert_id='hr')
    if obs.current_brief is None:
        return CoSAction(action_type='summarize')
    return CoSAction(action_type='submit')


def _roundrobin_baseline(obs) -> CoSAction:
    for expert in ['analyst', 'finance', 'strategy', 'hr']:
        if expert not in obs.consulted_experts:
            return CoSAction(action_type='consult', expert_id=expert)
    if obs.current_brief is None:
        return CoSAction(action_type='summarize')
    return CoSAction(action_type='submit')


_TRAINED_POLICY = {"model": None}


def _trained_action(obs) -> CoSAction:
    import numpy as np
    import torch

    from training.train_cos_local import ACTIONS, PolicyNet, featurize

    if _TRAINED_POLICY["model"] is None:
        ckpt = REPO / "training" / "checkpoints" / "cos_final.pt"
        if not ckpt.exists():
            ckpt = REPO / "training" / "checkpoints" / "cos_ckpt0.pt"
        model = PolicyNet()
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        _TRAINED_POLICY["model"] = model
    model = _TRAINED_POLICY["model"]
    feats = torch.from_numpy(featurize(obs)).unsqueeze(0)
    with torch.no_grad():
        logits = model(feats)
    idx = int(torch.argmax(logits, dim=-1).item())
    return ACTIONS[idx]


def _cache_path(task_name: str, step_count: int, prompt: str) -> Path:
    prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
    cache_dir = REPO / 'cache'
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f'{task_name}_step{step_count}_{prompt_hash}.json'


def _llm_action(obs) -> CoSAction:
    from openai import OpenAI

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    prompt = (
        'You are the Chief of Staff in a multi-agent company simulation. '

        f'Task: {obs.task_name}. Instruction: {obs.instruction}. '

        f'Consulted experts: {obs.consulted_experts}. Current issues: {obs.issues}. '

        'Return exactly one JSON object with keys action_type, expert_id, sub_question_id, notes. '

        'Valid experts: analyst, finance, hr, strategy. Valid actions: consult, ask, summarize, submit, noop.'
    )
    cache_path = _cache_path(obs.task_name, obs.step_count, prompt)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
        return CoSAction.model_validate(payload)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        messages=[{'role': 'system', 'content': 'Respond with strict JSON only.'}, {'role': 'user', 'content': prompt}],
    )
    text = completion.choices[0].message.content or '{}'
    start = text.find('{')
    end = text.rfind('}')
    payload = json.loads(text[start:end + 1]) if start != -1 and end != -1 else {}
    cache_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return CoSAction.model_validate(payload)


def run_episode(task: str, picker: Callable, label: str) -> float:
    env = CEOBriefEnvironment()
    obs = env.reset(task=task)
    rewards: list[float] = []
    steps = 0
    score = 0.001
    success = False
    log_start(task=task, env=BENCHMARK, model=label)
    try:
        while not obs.done and steps < obs.max_steps:
            steps += 1
            action = picker(obs)
            obs = env.step(action)
            rewards.append(float(obs.reward))
            log_step(steps, _action_str(action), float(obs.reward), bool(obs.done), None)
        score = float(obs.terminal_grader_score or 0.001)
        score = max(0.001, min(0.999, score))
        success = score >= 0.5
        return score
    except Exception as exc:
        log_step(max(steps, 1), 'exception', 0.0, True, str(exc).replace('\n', ' '))
        return 0.001
    finally:
        log_end(success, steps, score, rewards)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--baseline', choices=['single', 'roundrobin'])
    parser.add_argument('--trained', action='store_true',
                        help='use locally trained CoS policy (training/checkpoints/cos_final.pt)')
    parser.add_argument('--task')
    parser.add_argument('--ablation', action='store_true')
    args = parser.parse_args()

    if args.trained:
        picker = _trained_action
        label = 'trained-cos'
    elif args.baseline == 'single':
        picker = _single_baseline
        label = 'single-baseline'
    elif args.baseline == 'roundrobin':
        picker = _roundrobin_baseline
        label = 'roundrobin-baseline'
    elif args.oracle or not API_KEY:
        picker = oracle_action_for_observation
        label = 'oracle'
    else:
        picker = _llm_action
        label = MODEL_NAME

    tasks = [args.task] if args.task else TASKS
    results = {}
    for task in tasks:
        results[task] = run_episode(task, picker, label)

    if args.ablation:
        ablations = {
            'single': {task: run_episode(task, _single_baseline, 'single-baseline') for task in tasks},
            'roundrobin': {task: run_episode(task, _roundrobin_baseline, 'roundrobin-baseline') for task in tasks},
            'oracle_or_llm': results,
        }
        trained_ckpt = REPO / 'training' / 'checkpoints' / 'cos_final.pt'
        if trained_ckpt.exists():
            ablations['trained_cos'] = {
                task: run_episode(task, _trained_action, 'trained-cos') for task in tasks
            }
        cache_dir = REPO / 'cache'
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / 'ablation_results.json').write_text(json.dumps(ablations, indent=2), encoding='utf-8')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
