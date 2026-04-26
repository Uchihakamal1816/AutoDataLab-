# Training Evidence

Small, replayable `evidence.json` files for the four method comparisons used in
the demo:

- `sft/`
- `dpo/`
- `sft_dpo/`
- `grpo_rlvr/`

These files store recorded CoS action routes, rewards, fallback usage, and
terminal scores. They do not require adapter weights, so the GRPO+RLVR evidence
can be used even when the exported run has no `adapter_config.json`.

The `plots/` folder contains small committed PNGs for judges:

- terminal scores by method
- policy rewards by method
- expert-brief reward curve
- RL loss by method
- RL best-reward tracking by method
- RL chosen-action correctness by method

The `rl_training_metrics/` folder contains real `train_metrics.json` exports
from GRPO, PPO, and GRPO+RLVR runs. These are the source for the RL loss plots.

Generate full textual context reports with:

```bash
python3 training/scripts/kaggle_context_results_from_evidence.py --roots .
```
