# 3-minute pitch outline

## 1. Problem (20s)
"Enterprise work is never one agent, one tool. A CEO asks a question and four specialists have to
coordinate under partial information. LLMs alone collapse on that. That's the gap we train on."

## 2. Environment (30s)
- Human CEO issues a brief.
- A **trainable Chief of Staff** orchestrates four specialist agents:
  Data Analyst, Finance, HR / Communications, Strategy.
- Each expert has its own tools/graders (wraps AutoDataLab analytics + Email/HR memo graders from Round 1).
- Dense rewards per routing decision + a terminal grader on the final brief.
- Theme fit: Multi-Agent Interactions + Halluminate (multi-actor) + Fleet AI (oversight).

## 3. Demo — lead with ablation (60s)
4-arm comparison on terminal grader score:

| System | easy | medium | hard | avg |
|---|---:|---:|---:|---:|
| Single LLM with all tools | 0.562 | 0.464 | 0.454 | **0.493** |
| Fixed round-robin | 0.907 | 0.853 | 0.844 | 0.868 |
| Oracle (upper bound) | 0.907 | 0.853 | 0.844 | 0.868 |
| **Trained CoS (ours)** | **0.907** | **0.729** | **0.719** | **0.785** |

Punchline: "Even a well-prompted single LLM scores 0.49. Route the work through specialists and we
hit 0.79, and the CoS **learned** that routing from scratch."

## 4. Improvement — show the curve (40s)
- REINFORCE on CoS, 600 episodes, CPU-only, ~1 min wall time.
- Before / after mean terminal score: **0.536 -> 0.835 (+55.7%)**.
- Hardest task jumps the most: hard_brief 0.391 -> 0.781.
- Show `training/reward_curves/reward_curve.png` with episode reward + 20-ep moving average + terminal curve.

## 5. Close (30s)
"AutoDataLab++ is a reusable recipe: swap the four specialists for any enterprise role set (sales,
compliance, legal, ops) and you get a trainable oversight agent on day one. That's the point of
Multi-Agent Interactions on OpenEnv."

## Backup slides
- Action space (10 discrete actions) + observation featurization.
- Reward shaping breakdown (routing + terminal grader, strictly clamped in (0.001, 0.999)).
- Failure-recovery clip: `ask` action re-queries Analyst when data quality flag fires.
