# Submission checklist

## Done (local)
- [x] Root files present (`openenv.yaml`, `pyproject.toml`, `uv.lock`, `inference.py`, `Dockerfile`, `server/app.py` with `main()` + `__main__` guard)
- [x] `openenv validate --verbose` passes locally
- [x] `python3 validate_submission.py` exits 0 locally
- [x] Oracle episodes score strictly inside `(0.001, 0.999)`
- [x] `/`, `/health`, `/tasks`, `/reset`, `/step`, `/state` return 200 locally
- [x] All 4 specialist agents implemented (Data Analyst, Finance, HR, Strategy)
- [x] Chief of Staff environment with dense reward shaping + terminal grader
- [x] 3 CEO-brief tasks (easy/medium/hard) with auto-generated ground truth

## Training (done)
- [x] Real CoS training script: `training/train_cos_local.py` (REINFORCE, CPU, ~1 min)
- [x] Reward curve PNG: `training/reward_curves/reward_curve.png`
- [x] Checkpoints: `training/checkpoints/cos_ckpt0.pt` + `cos_final.pt`
- [x] Before / after numbers: `training/reward_curves/before_after.json`
      (mean terminal 0.536 -> 0.835, +55.7%)
- [x] Trained policy wired into `inference.py --trained`
- [x] 4-arm ablation refreshed: `cache/ablation_results.json`
      (single=0.493, roundrobin=0.868, oracle=0.868, trained_cos=0.785)
- [x] Colab GRPO notebook (`training/train_cos_colab.ipynb`) as secondary path

## External (blocked, awaiting user action)
- [ ] Docker build (blocked by Docker Hub network reset; retry when network is free or use an already-cached base image)
- [ ] HF Space push (needs `HF_TOKEN` in env + `huggingface-cli login`)
- [ ] Live URL validation (run `python3 validate_submission.py --live-url <hf-space-url>`)

## Submission artifacts (ready)
- [x] `docs/hf_mini_blog.md` (mini-blog with ablation + training tables)
- [x] `docs/pitch_outline.md` (3-min pitch)
- [x] `docs/demo_script.md` (demo run instructions)
- [ ] 2-min demo video (record once live URL is up)
- [ ] Final submission

## Quick commands
```
# Train CoS + refresh ablation
./scripts/run_training.sh

# Full local validation
python3 validate_submission.py

# Run trained CoS on one task
python3 inference.py --trained --task hard_brief
```
