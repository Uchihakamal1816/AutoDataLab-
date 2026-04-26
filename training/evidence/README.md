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

Generate full textual context reports with:

```bash
python3 training/kaggle_context_results_from_evidence.py --roots .
```
