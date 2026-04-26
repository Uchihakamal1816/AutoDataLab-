---
title: AutoDataLab Plus Plus
emoji: 🏢
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AutoDataLab++

**Team: Peaky Blinders** — Sai Kamal Nannuri ([Uchihakamal1816](https://github.com/Uchihakamal1816)) & Ajeet Kumar ([ajeetkartikay](https://github.com/ajeetkartikay))

A multi-agent OpenEnv environment where a **Chief of Staff (CoS)** policy routes work across four AI specialists — **Data Analyst**, **Finance**, **Strategy**, and **HR** — to complete realistic CEO briefing tasks. We trained **Qwen2.5-1.5B-Instruct** with **GRPO** to be that CoS, and added **RAG** over a corpus of company SOPs, policies, and exemplar memos.

## Links

| Resource | URL |
|---|---|
| Live HF Space (primary) | https://huggingface.co/spaces/uchihamadara1816/AutoDataLab2.0 |
| Live demo URL | https://uchihamadara1816-autodatalab2-0.hf.space/ui/ |
| Backup HF Space | https://huggingface.co/spaces/kartik1230/round2-1 |
| GitHub repo | https://github.com/Uchihakamal1816/AutoDataLab- |
| Blog writeup | [Blog.md](./Blog.md) |
| Training notebook (Colab) | [training/cos_grpo_colab.ipynb](./training/cos_grpo_colab.ipynb) |
| Reward curves | [training/reward_curves/reward_curve.png](./training/reward_curves/reward_curve.png) |
| Demo video | _TODO: link YouTube video here_ |

## The Problem

Most LLM benchmarks test single-turn reasoning. Real enterprise work isn't like that. A CEO doesn't ask one question; they ask "look at our data, project the quarter, recommend an action, and draft the memo." That requires routing decisions — which expert to consult, in what order, when you have enough information, when to stop iterating, when to commit.

Next-token training rewards plausible-sounding text. **It does not reward stopping.** Our hypothesis: a small LLM trained with RL on the routing decision will outperform a much larger LLM that's just prompted to orchestrate.

## How the Environment Works

### Action space (small + discrete = trainable)
consult(expert_id)   # expert_id in {analyst, finance, strategy, hr}
ask(expert_id, sub_question_id)
summarize()
submit()
noop()
### Observation space

Typed Pydantic CoSObservation with task_name, instruction, consulted_experts, expert_reports, current_brief, data_quality_score, issues, step_count, max_steps, reward_breakdown.

### Tasks (6 total)

| Task | Difficulty | Strategy required? |
|---|---|---|
| easy_brief | easy | No |
| medium_brief | medium | Yes |
| hard_brief | hard | Yes |
| expert_brief | hard | Yes |
| risk_brief | hard | Yes |
| crisis_brief | hard | Yes |

### Reward shaping

- +0.08 per productive consult of a required expert
- +0.02 for a useful summarize
- -0.05 for redundant consults
- -0.02 per step (efficiency pressure)
- Terminal grader score in (0.001, 0.999) on submit, weighing brief content quality, HR memo structure + tone, and strategy rubric grounding

### The four specialists

- Data Analyst — wraps subenvs/autodatalab/analytics.py: cleaning, KPIs, revenue derivation, data-quality scoring
- Finance — deterministic projection / variance / break-even tools
- Strategy — produces 3-bullet operating plans with citations to upstream numbers
- HR / Comms — wraps subenvs/email/: drafts memos, scored on a blended structure + tone + audience rubric

## The Training

Base model: Qwen2.5-1.5B-Instruct. Method: GRPO (Group Relative Policy Optimization) via Hugging Face TRL. Rollouts collected against the live HF Space env. The policy was warm-started with SFT on demonstration trajectories, then RL'd against the shaped reward.

A 1.5B model is small enough to fine-tune on a single GPU but large enough to follow the JSON action schema. The whole point is to show that a small RL'd policy beats a large prompted-only one on routing.

Reproduce: open [training/cos_grpo_colab.ipynb](./training/cos_grpo_colab.ipynb) in Colab and run end-to-end.

## RAG

The memory/ package contains company SOPs, policies, and exemplar memos. A BM25 retriever (deterministic, no neural embeddings) pulls the top-k passages most relevant to each expert's working query. With RAG ON, experts ground their answers in retrieved snippets and emit memory_citations alongside their reports. Across all 6 evidence rollouts, RAG ON adds ~+0.01 to terminal scores consistently.

## Results

### Terminal scores (3 hard tasks, 4 conditions)

| Task | Base LLM (no RAG) | Base LLM + RAG | Base + GRPO (no RAG) | Base + GRPO + RAG |
|---|---|---|---|---|
| expert_brief | 0.8827 | 0.8925 | 0.8827 | 0.8925 |
| risk_brief | 0.8839 | 0.8948 | 0.8839 | 0.8948 |
| crisis_brief | 0.8805 | 0.8914 | 0.8805 | 0.8914 |

Terminal scores look identical — but only because the env's fallback path completes missing experts before grading.

### What the policy actually did — the real difference

| Dimension | Base LLM (no RL) | Base LLM + GRPO |
|---|---|---|
| model_routed_required | 3 of 4 experts | All 4 experts |
| needed_fallback | True (every run) | False (every run) |
| Action repetition | Spam-consults strategy 4x in a row | Clean linear path |
| Action sequence | analyst -> finance -> strategy -> strategy -> strategy -> strategy | analyst -> finance -> strategy -> hr -> summarize -> submit |

The base LLM never figures out HR is required — it picks the most "interesting" option (strategy) and keeps picking it until the env's fallback force-completes the missing experts.

The GRPO-trained CoS picks all four required experts in the optimal order, summarizes when ready, submits, terminates. Six steps. No fallback. Raw evidence in training/exports/.

If we removed the env's fallback, the base LLM would hit max-steps with an incomplete brief. The RL policy would still complete cleanly. That's the real result.

See [Blog.md](./Blog.md) for the full writeup with rollout citations and methodology.

## Quickstart

### Use the live deployment (no install)
curl https://uchihamadara1816-autodatalab2-0.hf.space/health
{"status":"healthy"}
curl -X POST https://uchihamadara1816-autodatalab2-0.hf.space/reset 
-H 'content-type: application/json' 
-d '{"task":"hard_brief","use_rag":true}'
Open the office UI: https://uchihamadara1816-autodatalab2-0.hf.space/ui/

### Run locally
pip install -e .
python -m server.app
Then open http://127.0.0.1:7860/ui/

### Pre-submission checks
python validate_submission.py
openenv validate --verbose
python tests_oracle_all.py
### Validate a deployed Space
python validate_submission.py --base-url https://your-space.hf.space
## Environment Variables

| Variable | Purpose |
|---|---|
| API_BASE_URL | OpenAI-compatible LLM endpoint (default: HF router) |
| API_KEY or HF_TOKEN | LLM auth; if unset, inference.py falls back to oracle |
| MODEL_NAME | LLM model id |
| AUTODATALAB_MAX_SESSIONS | Server session cap (default 64) |

## Project Layout
ceo_brief_env/        # Pydantic models, environment, graders, 6 task definitions
experts/            # data_analyst, finance, strategy, hr
tasks/              # easy/medium/hard/expert/risk/crisis briefs
inference.py          # oracle / baselines / LLM / trained CoS, [START]/[STEP]/[END] logs
server/app.py         # FastAPI: /reset, /step, /state, /visualize/run, /ui/
training/             # GRPO Colab notebook, checkpoints, reward curves, evidence rollouts
subenvs/              # analyst data tooling + email/HR communication tooling
memory/               # SOPs, policies, history exemplars (RAG corpus)
openenv.yaml          # OpenEnv spec
Dockerfile            # HF Space build
## Docker
docker build -t autodatalab-plus .
docker run --rm -p 7860:7860 autodatalab-plus
curl http://127.0.0.1:7860/health
## License

Hackathon / team use.

## Acknowledgments

Built for the OpenEnv Hackathon. Thanks to the OpenEnv team for the spec, Hugging Face for the Spaces hosting and TRL library, and Sanyam for the clarifications on submission requirements.
