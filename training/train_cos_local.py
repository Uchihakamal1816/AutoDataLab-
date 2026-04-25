"""Local REINFORCE trainer for the AutoDataLab++ Chief of Staff.

Runs entirely on CPU in a few minutes and produces:
  - training/reward_curves/reward_curve.png     (real reward curve)
  - training/checkpoints/cos_ckpt0.pt           (random init, before training)
  - training/checkpoints/cos_final.pt           (after training)
  - training/reward_curves/before_after.json    (mean terminal score per task)

This is the primary training path. The Colab notebook (train_cos_colab.ipynb)
is the secondary GRPO-on-LLM path for Kartikay's HF Space run.

Usage:
    python training/train_cos_local.py --episodes 400
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceo_brief_env.environment import CEOBriefEnvironment
from ceo_brief_env.models import CoSAction, CoSObservation

TASKS = ["easy_brief", "medium_brief", "hard_brief", "expert_brief", "risk_brief", "crisis_brief"]

ACTIONS: List[CoSAction] = [
    CoSAction(action_type="consult", expert_id="analyst"),
    CoSAction(action_type="consult", expert_id="finance"),
    CoSAction(action_type="consult", expert_id="strategy"),
    CoSAction(action_type="consult", expert_id="hr"),
    CoSAction(action_type="ask", expert_id="analyst"),
    CoSAction(action_type="ask", expert_id="finance"),
    CoSAction(action_type="ask", expert_id="strategy"),
    CoSAction(action_type="ask", expert_id="hr"),
    CoSAction(action_type="summarize"),
    CoSAction(action_type="submit"),
]
N_ACTIONS = len(ACTIONS)

# Extra features when strategy has run: encodes watchlist stances + Present/Future
# tokens from `ExpertReport.metrics` (see `StrategyExpert` in
# `ceo_brief_env/experts/strategy.py`). Zeros when strategy not consulted yet.
N_STRATEGY_IDEA = 10


def _stance_float(x: str | int | float | None) -> float:
    """Map strategy metric tokens to [0,1] (rough buy→sell / trim spectrum)."""
    t = str(x or "").lower().strip()
    table: dict[str, float] = {
        "buy_more": 0.0,
        "buy": 0.15,
        "add": 0.3,
        "hold": 0.5,
        "reduce": 0.65,
        "trim": 0.72,
        "sell": 0.9,
        "none": 0.12,
    }
    if t in table:
        return table[t]
    for k, v in table.items():
        if k and k in t:
            return v
    return 0.45


def strategy_idea_features(obs: CoSObservation) -> list[float]:
    r = obs.expert_reports.get("strategy")
    if r is None:
        return [0.0] * N_STRATEGY_IDEA
    m = r.metrics
    return [
        1.0,
        _stance_float(m.get("nvda")),
        _stance_float(m.get("aapl")),
        _stance_float(m.get("jpm")),
        _stance_float(m.get("nvda_present")),
        _stance_float(m.get("nvda_future")),
        _stance_float(m.get("aapl_present")),
        _stance_float(m.get("aapl_future")),
        _stance_float(m.get("jpm_present")),
        _stance_float(m.get("jpm_future")),
    ]


def featurize(obs: CoSObservation) -> np.ndarray:
    consulted = set(obs.consulted_experts)
    task_onehot = [1.0 if obs.task_name == t else 0.0 for t in TASKS]
    expert_bits = [
        1.0 if "analyst" in consulted else 0.0,
        1.0 if "finance" in consulted else 0.0,
        1.0 if "strategy" in consulted else 0.0,
        1.0 if "hr" in consulted else 0.0,
    ]
    brief = 1.0 if obs.current_brief is not None else 0.0
    step_frac = float(obs.step_count) / max(1, obs.max_steps)
    dq = float(obs.data_quality_score or 0.0)
    base = task_onehot + expert_bits + [brief, step_frac, dq] + strategy_idea_features(obs)
    return np.array(base, dtype=np.float32)


FEAT_DIM = len(featurize(CEOBriefEnvironment().reset("easy_brief")))


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 64, out_dim: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_policy_state_dict_from_file(model: PolicyNet, path: Path) -> str:
    """
    Load a saved ``PolicyNet`` state dict, padding or truncating the first Linear
    if the checkpoint was trained with a different input size (e.g. older runs
    without the 10-dim strategy-idea block, or 3 task slots vs 4).
    Returns a short status string: ``ok`` | ``padded_input_*`` | ``truncated_*``.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))
    state = torch.load(path, map_location="cpu")
    wkey, bkey = "net.0.weight", "net.0.bias"
    if wkey not in state:
        model.load_state_dict(state)
        return "ok"
    new_in, want_in = state[wkey].shape[1], FEAT_DIM
    if new_in == want_in:
        model.load_state_dict(state, strict=True)
        return "ok"
    w = state[wkey].clone()
    if new_in < want_in:
        pad = torch.zeros(w.shape[0], want_in, dtype=w.dtype, device=w.device)
        pad[:, :new_in] = w
        state[wkey] = pad
        model.load_state_dict(state, strict=True)
        return f"padded_input_{new_in}_to_{want_in}"
    state[wkey] = w[:, :want_in]
    model.load_state_dict(state, strict=True)
    return f"truncated_{new_in}_to_{want_in}"


def run_episode(env: CEOBriefEnvironment, policy: PolicyNet, task: str, greedy: bool = False):
    obs = env.reset(task)
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    steps = 0
    while not obs.done and steps < obs.max_steps:
        feats = torch.from_numpy(featurize(obs)).unsqueeze(0)
        logits = policy(feats)
        dist = Categorical(logits=logits)
        action_idx = int(torch.argmax(logits, dim=-1).item()) if greedy else int(dist.sample().item())
        log_probs.append(dist.log_prob(torch.tensor([action_idx])))
        obs = env.step(ACTIONS[action_idx])
        rewards.append(float(obs.reward))
        steps += 1
    terminal = float(obs.terminal_grader_score or 0.0)
    return log_probs, rewards, terminal, steps


def discount(rewards: list[float], gamma: float = 0.97) -> torch.Tensor:
    g = 0.0
    out: list[float] = []
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    t = torch.tensor(out, dtype=torch.float32)
    if len(t) > 1:
        t = (t - t.mean()) / (t.std() + 1e-6)
    return t


def evaluate(policy: PolicyNet, env: CEOBriefEnvironment, n: int = 10) -> dict:
    out: dict = {}
    for task in TASKS:
        terminals = []
        cumulatives = []
        for _ in range(n):
            _, rewards, terminal, _ = run_episode(env, policy, task, greedy=False)
            terminals.append(terminal)
            cumulatives.append(sum(rewards))
        out[task] = {
            "mean_terminal": round(float(np.mean(terminals)), 4),
            "mean_cumulative": round(float(np.mean(cumulatives)), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CEOBriefEnvironment()
    policy = PolicyNet()
    ckpt_dir = ROOT / "training" / "checkpoints"
    curve_dir = ROOT / "training" / "reward_curves"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    torch.save(policy.state_dict(), ckpt_dir / "cos_ckpt0.pt")
    before = evaluate(policy, env, n=5)
    print(f"[START] local REINFORCE training | episodes={args.episodes} | lr={args.lr}")
    print(f"[EVAL] before: {json.dumps(before)}")

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    reward_history: list[float] = []
    terminal_history: list[float] = []
    window: list[float] = []
    for ep in range(1, args.episodes + 1):
        task = TASKS[ep % len(TASKS)]
        log_probs, rewards, terminal, steps = run_episode(env, policy, task)
        returns = discount(rewards)
        if len(log_probs) == 0:
            continue
        log_prob_t = torch.cat(log_probs)
        loss = -(log_prob_t * returns).sum()
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optim.step()

        ep_reward = sum(rewards)
        reward_history.append(ep_reward)
        terminal_history.append(terminal)
        window.append(ep_reward)
        if len(window) > 20:
            window.pop(0)
        if ep % 20 == 0 or ep == 1:
            print(
                f"[STEP] ep={ep:04d} task={task} steps={steps} "
                f"ep_reward={ep_reward:+.3f} terminal={terminal:.3f} "
                f"ma20={np.mean(window):+.3f}"
            )

    torch.save(policy.state_dict(), ckpt_dir / "cos_final.pt")
    after = evaluate(policy, env, n=10)
    print(f"[EVAL] after:  {json.dumps(after)}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    window_avg = [np.mean(reward_history[max(0, i - 20): i + 1]) for i in range(len(reward_history))]
    ax.plot(reward_history, color="#bbb", alpha=0.5, label="episode reward")
    ax.plot(window_avg, color="#1f77b4", label="20-ep moving avg")
    ax.plot(terminal_history, color="#d62728", alpha=0.7, label="terminal grader")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("AutoDataLab++ Chief of Staff - REINFORCE")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = curve_dir / "reward_curve.png"
    fig.savefig(curve_path, dpi=130)
    plt.close(fig)

    summary = {
        "episodes": args.episodes,
        "lr": args.lr,
        "before": before,
        "after": after,
        "mean_terminal_before": round(float(np.mean([v["mean_terminal"] for v in before.values()])), 4),
        "mean_terminal_after": round(float(np.mean([v["mean_terminal"] for v in after.values()])), 4),
        "curve_path": str(curve_path.relative_to(ROOT)),
    }
    (curve_dir / "before_after.json").write_text(json.dumps(summary, indent=2))
    print(f"[END] saved curve={curve_path.name} final_ckpt=cos_final.pt")
    print(f"[END] before_mean={summary['mean_terminal_before']} after_mean={summary['mean_terminal_after']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
