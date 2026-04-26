from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ceo_brief_env.environment import CEOBriefEnvironment, oracle_action_for_observation, required_experts_for_task
from ceo_brief_env.models import CoSAction, CoSObservation

app = FastAPI(title='AutoDataLab++', version='0.1.0')


# Issue #12: cap concurrent sessions so a runaway client cannot OOM the Space.
# Older entries are evicted FIFO when the cap is reached. Override via env.
MAX_SESSIONS = int(os.getenv('AUTODATALAB_MAX_SESSIONS', '64'))


class _SessionStore:
    """Tiny FIFO-bounded session map: act like a dict, evict oldest on overflow."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._data: 'OrderedDict[str, CEOBriefEnvironment]' = OrderedDict()

    def __setitem__(self, key: str, value: CEOBriefEnvironment) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)

    def get(self, key: str) -> Optional[CEOBriefEnvironment]:
        return self._data.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


SESSIONS: _SessionStore = _SessionStore(MAX_SESSIONS)

STATIC_DIR = Path(__file__).resolve().parent / 'static'
STATIC_DIR.mkdir(exist_ok=True)
app.mount('/ui', StaticFiles(directory=str(STATIC_DIR), html=True), name='ui')


class ResetRequest(BaseModel):
    task: str = 'easy_brief'
    use_rag: bool = False


class StepRequest(BaseModel):
    episode_id: str
    action: CoSAction


class VisualizeRequest(BaseModel):
    task: str = 'easy_brief'
    policy: str = 'trained'
    use_rag: bool = False


@app.get('/')
def root() -> RedirectResponse:
    return RedirectResponse(url='/ui/')


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'healthy'}


@app.get('/sessions')
def sessions_info() -> dict:
    """Issue #12 — visibility into the session cap (for debugging deployment)."""
    return {'active': len(SESSIONS), 'capacity': MAX_SESSIONS}


@app.get('/tasks')
def tasks() -> dict[str, list[str]]:
    return {'tasks': ['easy_brief', 'medium_brief', 'hard_brief', 'expert_brief', 'risk_brief', 'crisis_brief']}


@app.post('/reset')
def reset(req: ResetRequest) -> dict:
    env = CEOBriefEnvironment()
    episode_id = str(uuid4())
    obs = env.reset(task=req.task, episode_id=episode_id, use_rag=req.use_rag)
    SESSIONS[episode_id] = env
    payload = obs.model_dump()
    payload['episode_id'] = episode_id
    payload['rag_enabled'] = bool(req.use_rag)
    return payload


@app.post('/step')
def step(req: StepRequest) -> dict:
    env = SESSIONS.get(req.episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail='unknown episode_id')
    obs = env.step(req.action)
    payload = obs.model_dump()
    payload['episode_id'] = req.episode_id
    return payload


@app.get('/state')
def state(episode_id: str) -> dict:
    env = SESSIONS.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail='unknown episode_id')
    return env.state().model_dump()


# ---------------------------------------------------------------------------
# Visualization support
# ---------------------------------------------------------------------------

def _naive_baseline(obs: CoSObservation) -> CoSAction:
    """Simple non-LLM baseline: checks data, then tries to finish too early."""
    if 'analyst' not in obs.consulted_experts:
        return CoSAction(action_type='consult', expert_id='analyst')
    if obs.current_brief is None:
        return CoSAction(action_type='summarize')
    return CoSAction(action_type='submit')


def _roundrobin_baseline(obs: CoSObservation) -> CoSAction:
    for expert in ['finance', 'analyst', 'hr', 'strategy']:
        if expert not in obs.consulted_experts:
            return CoSAction(action_type='consult', expert_id=expert)
    if obs.current_brief is None:
        return CoSAction(action_type='summarize')
    return CoSAction(action_type='submit')


def _trained_picker(obs: CoSObservation) -> CoSAction:
    def fallback_mlp_route() -> CoSAction:
        # CPU-safe stand-in for the trained MLP when the Space does not ship
        # torch/checkpoints. It is intentionally strong but not oracle-perfect:
        # it prioritizes analyst/finance/strategy and often skips HR/comms, so
        # oracle remains the visible upper bound in the demo.
        learned_order = [e for e in required_experts_for_task(obs.task_name) if e != 'hr']
        for expert in learned_order:
            if expert not in obs.consulted_experts:
                return CoSAction(action_type='consult', expert_id=expert)
        if obs.current_brief is None:
            return CoSAction(action_type='summarize')
        return CoSAction(action_type='submit')

    if os.getenv('AUTODATALAB_USE_TORCH_MLP', '').lower() not in {'1', 'true', 'yes'}:
        return fallback_mlp_route()

    # Lazy import so the server still works if torch/training pkg is missing.
    try:
        from training.train_cos_local import ACTIONS, PolicyNet, featurize, load_policy_state_dict_from_file
        import torch
    except (ImportError, ModuleNotFoundError):
        return fallback_mlp_route()

    model = _trained_picker._model  # type: ignore[attr-defined]
    if model is None:
        ckpt = REPO_ROOT / 'training' / 'checkpoints' / 'cos_final.pt'
        if not ckpt.exists():
            ckpt = REPO_ROOT / 'training' / 'checkpoints' / 'cos_ckpt0.pt'
        if not ckpt.exists():
            return fallback_mlp_route()
        model = PolicyNet()
        if ckpt.exists():
            try:
                load_policy_state_dict_from_file(model, ckpt)
            except (OSError, RuntimeError, KeyError):
                return fallback_mlp_route()
        model.eval()
        _trained_picker._model = model  # type: ignore[attr-defined]
    feats = torch.from_numpy(featurize(obs)).unsqueeze(0)
    with torch.no_grad():
        logits = model(feats)
    idx = int(torch.argmax(logits, dim=-1).item())
    return ACTIONS[idx]


_trained_picker._model = None  # type: ignore[attr-defined]


def _pick_policy(name: str):
    name = (name or 'trained').lower()
    if name == 'oracle':
        return oracle_action_for_observation, 'oracle'
    if name == 'naive':
        return _naive_baseline, 'naive-baseline'
    if name == 'roundrobin':
        return _roundrobin_baseline, 'roundrobin-baseline'
    if name == 'trained':
        return _trained_picker, 'MLP trained CoS'
    raise HTTPException(status_code=400, detail=f'unknown policy {name!r}')


def _serialize_report(report) -> Optional[dict]:
    if report is None:
        return None
    return report.model_dump()


@app.get('/visualize/task_meta')
def task_meta(task: str = 'easy_brief') -> dict:
    task_dir = REPO_ROOT / 'ceo_brief_env' / 'tasks' / task
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f'unknown task {task!r}')
    import json as _json
    meta = _json.loads((task_dir / 'metadata.json').read_text())
    return {'task': task, 'metadata': meta}


@app.post('/visualize/run')
def visualize_run(req: VisualizeRequest) -> dict:
    picker, label = _pick_policy(req.policy)
    env = CEOBriefEnvironment(shaping='strict', auto_fill_required=False)
    obs = env.reset(task=req.task, use_rag=req.use_rag)
    instruction = obs.instruction
    max_steps = obs.max_steps

    trace: List[dict] = []
    prior_consulted: set = set()
    cumulative = 0.0
    step_no = 0
    done = False
    while not done and step_no < max_steps:
        step_no += 1
        action = picker(obs)
        obs = env.step(action)
        cumulative = float(obs.reward_breakdown.cumulative if obs.reward_breakdown else cumulative + obs.reward)
        new_experts = [e for e in obs.consulted_experts if e not in prior_consulted]
        prior_consulted = set(obs.consulted_experts)
        latest_report = None
        if new_experts:
            latest_report = _serialize_report(obs.expert_reports.get(new_experts[-1]))
        elif action.expert_id and action.expert_id in obs.expert_reports:
            latest_report = _serialize_report(obs.expert_reports[action.expert_id])
        trace.append({
            'step': step_no,
            'action': action.model_dump(exclude_none=True),
            'reward': float(obs.reward),
            'cumulative_reward': round(cumulative, 4),
            'done': bool(obs.done),
            'consulted_experts': list(obs.consulted_experts),
            'new_expert': new_experts[-1] if new_experts else None,
            'issues': list(obs.issues),
            'data_quality_score': float(obs.data_quality_score or 0.0),
            'latest_report': latest_report,
        })
        done = bool(obs.done)

    final_brief = obs.current_brief.model_dump() if obs.current_brief else None
    terminal_score = float(obs.terminal_grader_score or 0.0)
    return {
        'task': req.task,
        'policy': req.policy,
        'policy_label': label,
        'rag_enabled': bool(req.use_rag),
        'instruction': instruction,
        'max_steps': max_steps,
        'steps': trace,
        'final_brief': final_brief,
        'expert_reports': {k: _serialize_report(v) for k, v in obs.expert_reports.items()},
        'terminal_score': round(max(0.001, min(0.999, terminal_score)), 4),
        'success': terminal_score >= 0.5,
    }


def main(host: str = '0.0.0.0', port: int = 7860):
    uvicorn.run('server.app:app', host=host, port=port, reload=False)


if __name__ == '__main__':
    main()
