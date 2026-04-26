# Training Summary

This folder contains the re-runnable training and evidence pipeline for
AutoDataLab++.

## What Was Trained

The trained component is the Chief of Staff policy: given an OpenEnv observation,
it chooses the next action (`consult`, `ask`, `summarize`, `submit`) and routes
work to Data Analyst, Finance, Strategy, and HR experts.

## Re-Runnable Scripts

- `scripts/train_cos_local.py` trains a small CPU MLP CoS with REINFORCE and now saves
  both `reward_curve.png` and `loss_curve.png`. It logs to TensorBoard by
  default via `--report-to tensorboard`; use `--report-to wandb` for Weights &
  Biases.
- `scripts/kaggle_train_1p5b_methods.py` trains Qwen2.5-1.5B policy adapters with SFT,
  DPO, or SFT->DPO.
- `scripts/kaggle_rl_1p5b_methods.py` trains RL variants (`grpo`, `ppo`,
  `grpo_rlvr`) and saves `train_metrics.json`, `train_curve.png`,
  `loss_curve.png`, TensorBoard logs by default, and evaluation evidence.
- `scripts/kaggle_run_all_1p5b_experiments.py` runs the full SFT/DPO/RL comparison.
- `scripts/kaggle_context_results_from_evidence.py` converts saved evidence into full
  textual agent reports without needing adapter files.

## Committed Evidence

Replayable evidence files live in:

- `training/evidence/sft/evidence.json`
- `training/evidence/dpo/evidence.json`
- `training/evidence/sft_dpo/evidence.json`
- `training/evidence/grpo_rlvr/evidence.json`

These are small JSON files with recorded routes, model-controlled rewards,
fallback usage, terminal grader scores, and completion previews.

## Plots

Small plot artifacts committed for judging:

- `training/evidence/plots/expert_brief_reward_curve.png`
- `training/evidence/plots/policy_rewards_by_method.png`
- `training/evidence/plots/terminal_scores_by_method.png`
- `training/evidence/plots/rl_loss_by_method.png`
- `training/evidence/plots/rl_best_reward_by_method.png`
- `training/evidence/plots/rl_chosen_ok_by_method.png`

The real RL loss metrics used for these plots are committed under:

- `training/evidence/rl_training_metrics/grpo_train_metrics.json`
- `training/evidence/rl_training_metrics/grpo_rlvr_train_metrics.json`
- `training/evidence/rl_training_metrics/ppo_train_metrics.json`

For future RL runs, `kaggle_rl_1p5b_methods.py` writes per-run
`loss_curve.png` and `train_curve.png` under the run directory.

## Quick Kaggle Commands

Generate full textual evidence from the committed JSON:

```bash
python3 training/scripts/kaggle_context_results_from_evidence.py --roots .
```

Run RL-only methods:

```bash
python3 training/scripts/kaggle_rl_1p5b_methods.py --method grpo --epochs 1 --max-train-states 80 --report-to tensorboard
python3 training/scripts/kaggle_rl_1p5b_methods.py --method ppo --epochs 1 --max-train-states 80 --report-to tensorboard
python3 training/scripts/kaggle_rl_1p5b_methods.py --method grpo_rlvr --epochs 1 --max-train-states 80 --report-to tensorboard
```

Run all 1.5B experiments:

```bash
python3 training/scripts/kaggle_run_all_1p5b_experiments.py --quick
```
