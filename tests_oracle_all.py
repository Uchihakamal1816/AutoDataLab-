"""Run oracle on all 3 briefs — validator gate."""
from ceo_brief_env.environment import CEOBriefEnvironment, oracle_action_for_observation

BRIEFS = ["easy_brief", "medium_brief", "hard_brief", "expert_brief"]
results = {}

for brief in BRIEFS:
    env = CEOBriefEnvironment()
    obs = env.reset(brief)
    steps = 0
    cumulative = 0.0
    while not obs.done and steps < 20:
        action = oracle_action_for_observation(obs)
        obs = env.step(action)
        steps += 1
        cumulative += obs.reward

    results[brief] = {
        "terminal": obs.terminal_grader_score,
        "cumulative": round(cumulative, 4),
        "steps": steps,
        "experts": obs.consulted_experts,
    }
    print(f"{brief:14s} | terminal={obs.terminal_grader_score:.4f} | "
          f"cum={cumulative:.4f} | steps={steps} | experts={obs.consulted_experts}")

print()
print("=" * 70)
all_in_band = True
for brief, r in results.items():
    score = r["terminal"]
    in_band = score is not None and 0.001 <= score <= 0.999
    tag = "PASS" if in_band else "FAIL"
    print(f"  [{tag}] {brief:14s} terminal={score}")
    if not in_band:
        all_in_band = False

print()
if all_in_band:
    print("[ALL PASS] Oracle lands in (0.001, 0.999) on every brief.")
else:
    print("[FAIL] At least one brief is out of band — fix before proceeding.")