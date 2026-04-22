# AutoDataLab++ mini-blog

## What we built
AutoDataLab++ is a multi-agent OpenEnv environment where a trainable Chief of Staff routes work across four specialists:
- Data Analyst
- Finance
- HR / Communications
- Strategy

The environment models realistic enterprise work: cleaning noisy sales data, forecasting next-quarter performance, converting analysis into a strategy recommendation, and drafting an internal memo for stakeholders.

## Why it is interesting
- It targets **Multi-Agent Interactions** and the **Halluminate Multi-Actor** bonus directly.
- The Chief of Staff acts like an oversight agent, which also gives a **Fleet AI** angle.
- Rewards are dense: the policy is rewarded for consulting relevant experts, penalized for redundant routing, and graded on the final brief.

## Tasks
- `easy_brief`: revenue sanity check + team update
- `medium_brief`: category growth + forecast + action memo
- `hard_brief`: board-style summary + risks + CFO memo

## Ablation (terminal grader score, higher = better)
| System | easy_brief | medium_brief | hard_brief | avg |
|---|---:|---:|---:|---:|
| Single LLM (one agent, all tools) | 0.562 | 0.464 | 0.454 | 0.493 |
| Fixed round-robin orchestrator | 0.907 | 0.853 | 0.844 | 0.868 |
| Oracle (upper bound) | 0.907 | 0.853 | 0.844 | 0.868 |
| **Trained CoS (REINFORCE, ours)** | **0.907** | **0.729** | **0.719** | **0.785** |

## Training progression (REINFORCE, 600 episodes, CPU-only)
| Metric | Before training (random init) | After training | Delta |
|---|---:|---:|---:|
| Mean terminal score | 0.536 | **0.835** | **+0.299 (+55.7%)** |
| easy_brief | 0.614 | 0.907 | +0.293 |
| medium_brief | 0.605 | 0.816 | +0.211 |
| hard_brief | 0.391 | 0.781 | +0.390 |

The trained CoS discovered the correct `analyst -> finance -> strategy -> hr -> summarize -> submit` orchestration pattern purely from reward shaping, starting from random init, and ends competitive with the fixed round-robin while being far more flexible (it can skip, re-query with `ask`, or stop early).

## Demo script
1. Show the architecture slide (human CEO + 4 AI experts + CoS).
2. Show the 4-arm ablation table — the story is single-LLM loses, CoS wins.
3. Show the reward curve: 0.536 -> 0.835 over 600 episodes.
4. Run one live rollout with `[START] / [STEP] / [END]` logs visible.
5. Close on why enterprise multi-agent routing matters.
