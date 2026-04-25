#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

API_BASE_URL = os.getenv('API_BASE_URL', 'https://router.huggingface.co/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct')
API_KEY = os.getenv('API_KEY') or os.getenv('HF_TOKEN') or ''
BENCHMARK = 'autodatalab_plus'
TASKS = [t.strip() for t in os.getenv('AUTODATALAB_PLUS_TASKS', 'easy_brief,medium_brief,hard_brief,expert_brief').split(',') if t.strip()]

_LOG_SCORE_MIN = 0.001
_LOG_SCORE_MAX = 0.999

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ceo_brief_env.environment import CEOBriefEnvironment, oracle_action_for_observation, required_experts_for_task
from ceo_brief_env.models import CoSAction


def _bool_str(v: bool) -> str:
    return str(bool(v)).lower()


def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] task={task} env={env} model={model}', flush=True)


def _clamp_log_reward(r: float) -> float:
    return max(_LOG_SCORE_MIN, min(_LOG_SCORE_MAX, float(r)))


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else 'null'
    r = _clamp_log_reward(reward)
    print(f'[STEP] step={step} action={action} reward={r:.2f} done={_bool_str(done)} error={err}', flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    s = _clamp_log_reward(score)
    rewards_str = ','.join(f'{_clamp_log_reward(r):.2f}' for r in rewards)
    print(f'[END] success={_bool_str(success)} steps={steps} score={s:.2f} rewards={rewards_str}', flush=True)


def _action_str(action: CoSAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(',', ':'), sort_keys=True)


def _single_baseline(obs) -> CoSAction:
    for e in required_experts_for_task(obs.task_name):
        if e not in obs.consulted_experts:
            return CoSAction(action_type='consult', expert_id=e)
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


_TRAINED_POLICY: dict = {"model": None, "load_status": None, "load_warned": False}


def _trained_action(obs) -> CoSAction:
    import numpy as np
    import torch

    from training.train_cos_local import (
        ACTIONS,
        PolicyNet,
        featurize,
        load_policy_state_dict_from_file,
    )

    if _TRAINED_POLICY["model"] is None:
        import sys

        ckpt = REPO / "training" / "checkpoints" / "cos_final.pt"
        if not ckpt.exists():
            ckpt = REPO / "training" / "checkpoints" / "cos_ckpt0.pt"
        model = PolicyNet()
        status = "random_init"
        if ckpt.exists():
            try:
                status = load_policy_state_dict_from_file(model, ckpt)
            except (OSError, RuntimeError, KeyError) as e:
                print(
                    f"[inference] warning: could not load {ckpt} ({e}); using random init.",
                    file=sys.stderr,
                )
                model = PolicyNet()
                status = f"load_failed: {e}"
        else:
            print(
                "[inference] warning: no checkpoint; using random PolicyNet. "
                "Run: python3 training/train_cos_local.py",
                file=sys.stderr,
            )
        if status != "ok" and "padded" in status and not _TRAINED_POLICY.get("load_warned"):
            print(
                f"[inference] loaded checkpoint with compat: {status}. "
                "Re-run training for best results: python3 training/train_cos_local.py",
                file=sys.stderr,
            )
            _TRAINED_POLICY["load_warned"] = True
        model.eval()
        _TRAINED_POLICY["model"] = model
        _TRAINED_POLICY["load_status"] = status
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
        try:
            payload = json.loads(cache_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            payload = {}
        if not payload or 'action_type' not in payload:
            payload = {'action_type': 'noop'}
        return CoSAction.model_validate(payload)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        messages=[{'role': 'system', 'content': 'Respond with strict JSON only.'}, {'role': 'user', 'content': prompt}],
    )
    text = completion.choices[0].message.content or '{}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end >= start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    if not payload or 'action_type' not in payload:
        payload = {'action_type': 'noop'}
    cache_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return CoSAction.model_validate(payload)


def run_episode_collect(
    task: str,
    picker: Callable,
    label: str,
    use_rag: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Run one episode and return structured data (for --export). Same dynamics as `run_episode`.
    """
    env = CEOBriefEnvironment()
    obs = env.reset(task=task, use_rag=use_rag)
    rewards: list[float] = []
    trace: list[dict[str, Any]] = []
    steps = 0
    score = 0.001
    success = False
    err: Optional[str] = None
    if not quiet:
        log_start(task=task, env=f'{BENCHMARK} rag={use_rag}', model=label)
    try:
        while not obs.done and steps < obs.max_steps:
            steps += 1
            action = picker(obs)
            obs = env.step(action)
            r = float(obs.reward)
            rewards.append(r)
            if not quiet:
                log_step(steps, _action_str(action), r, bool(obs.done), None)
            trace.append(
                {
                    "step": steps,
                    "action": action.model_dump(exclude_none=True),
                    "reward": round(r, 4),
                    "done": bool(obs.done),
                    "consulted_experts": list(obs.consulted_experts),
                }
            )
        score = float(obs.terminal_grader_score or 0.001)
        score = max(0.001, min(0.999, score))
        success = score >= 0.5
    except Exception as exc:  # noqa: BLE001
        err = str(exc).replace('\n', ' ')
        if not quiet:
            log_step(max(steps, 1), 'exception', _LOG_SCORE_MIN, True, err)
    finally:
        if not quiet:
            log_end(success, steps, score, rewards)
    return {
        "task": task,
        "policy_label": label,
        "use_rag": use_rag,
        "success": success,
        "steps": steps,
        "terminal_score": round(score, 4),
        "cumulative_reward": round(sum(rewards), 4),
        "step_rewards": [round(x, 4) for x in rewards],
        "trace": trace,
        "error": err,
        "final_instruction": obs.instruction,
        "task_difficulty": obs.task_difficulty,
        "max_steps": obs.max_steps,
        "consulted_experts": list(obs.consulted_experts),
        "current_brief": obs.current_brief.model_dump() if obs.current_brief is not None else None,
        "expert_reports": {k: v.model_dump() for k, v in obs.expert_reports.items()},
    }


def run_episode(task: str, picker: Callable, label: str, use_rag: bool = False) -> float:
    out = run_episode_collect(task, picker, label, use_rag=use_rag, quiet=False)
    return float(out["terminal_score"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--baseline', choices=['single', 'roundrobin'])
    parser.add_argument('--trained', action='store_true',
                        help='use locally trained CoS policy (training/checkpoints/cos_final.pt)')
    parser.add_argument('--task')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument(
        '--rag',
        action='store_true',
        help='enable organizational memory (RAG) in experts and grounding in the grader (default: off, team parity)',
    )
    parser.add_argument(
        '--export',
        metavar='DIR',
        help='save one JSON per task (plus summary.json) under DIR; uses all tasks from env unless --task is set',
    )
    args = parser.parse_args()
    use_rag = bool(args.rag)

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

    if args.export:
        out_dir = Path(args.export).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        by_task: dict[str, dict[str, Any]] = {}
        for task in tasks:
            row = run_episode_collect(task, picker, label, use_rag=use_rag, quiet=True)
            by_task[task] = row
            (out_dir / f'{task}.json').write_text(
                json.dumps(row, indent=2, default=str), encoding='utf-8'
            )
            print(
                f'[export] {task} terminal={row["terminal_score"]} steps={row["steps"]} success={row["success"]}',
                flush=True,
            )
        mean_term = float(sum(t['terminal_score'] for t in by_task.values()) / max(len(by_task), 1))
        summary: dict[str, Any] = {
            'policy_label': label,
            'use_rag': use_rag,
            'checkpoint_load': _TRAINED_POLICY.get('load_status') if label == 'trained-cos' else None,
            'tasks': {
                t: {
                    'terminal_score': by_task[t]['terminal_score'],
                    'success': by_task[t]['success'],
                    'steps': by_task[t]['steps'],
                    'cumulative_reward': by_task[t]['cumulative_reward'],
                }
                for t in by_task
            },
            'mean_terminal': round(mean_term, 4),
            'export_dir': str(out_dir),
        }
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'[export] wrote {len(tasks) + 1} files to {out_dir}', flush=True)
        return 0
    else:
        results = {}
        for task in tasks:
            results[task] = run_episode(task, picker, label, use_rag=use_rag)

    if args.ablation and not args.export:
        ablations: dict[str, dict[str, float]] = {}
        if picker is _single_baseline:
            ablations['single'] = dict(results)
        else:
            ablations['single'] = {
                task: run_episode(task, _single_baseline, 'single-baseline', use_rag=use_rag) for task in tasks
            }
        if picker is _roundrobin_baseline:
            ablations['roundrobin'] = dict(results)
        else:
            ablations['roundrobin'] = {
                task: run_episode(task, _roundrobin_baseline, 'roundrobin-baseline', use_rag=use_rag) for task in tasks
            }
        ablations['oracle_or_llm'] = dict(results)
        trained_ckpt = REPO / 'training' / 'checkpoints' / 'cos_final.pt'
        if trained_ckpt.exists():
            if picker is _trained_action:
                ablations['trained_cos'] = dict(results)
            else:
                ablations['trained_cos'] = {
                    task: run_episode(task, _trained_action, 'trained-cos', use_rag=use_rag) for task in tasks
                }
        cache_dir = REPO / 'cache'
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / 'ablation_results.json').write_text(json.dumps(ablations, indent=2), encoding='utf-8')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
