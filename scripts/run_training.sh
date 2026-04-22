#!/usr/bin/env bash
# One-liner to train the Chief of Staff locally and refresh the ablation table.
# Outputs:
#   training/checkpoints/cos_ckpt0.pt + cos_final.pt
#   training/reward_curves/reward_curve.png
#   training/reward_curves/before_after.json
#   cache/ablation_results.json  (now includes trained_cos arm)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPISODES="${EPISODES:-600}"
LR="${LR:-3e-3}"

echo "[run_training] episodes=$EPISODES lr=$LR"
python3 training/train_cos_local.py --episodes "$EPISODES" --lr "$LR" --seed 0

echo "[run_training] refreshing ablation table (now with trained CoS arm)"
python3 inference.py --ablation

echo "[run_training] done"
echo "  - reward curve: training/reward_curves/reward_curve.png"
echo "  - before/after: training/reward_curves/before_after.json"
echo "  - ablation:     cache/ablation_results.json"
