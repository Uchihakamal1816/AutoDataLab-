"""Oracle smoke test for easy_brief — proves the whole pipeline runs end-to-end."""
from ceo_brief_env.environment import CEOBriefEnvironment, oracle_action_for_observation

env = CEOBriefEnvironment()
obs = env.reset("easy_brief")

steps = 0
cumulative = 0.0
while not obs.done and steps < 15:
    action = oracle_action_for_observation(obs)
    obs = env.step(action)
    steps += 1
    cumulative += obs.reward
    print(
        f"step={steps} "
        f"action={action.action_type}/{action.expert_id} "
        f"reward={obs.reward:.4f} "
        f"cumulative={cumulative:.4f} "
        f"done={obs.done}"
    )

print()
print(f"TERMINAL GRADER SCORE = {obs.terminal_grader_score}")
print(f"CUMULATIVE REWARD     = {cumulative:.4f}")
print(f"CONSULTED EXPERTS     = {obs.consulted_experts}")
print(f"ISSUES                = {obs.issues}")

assert obs.done, "episode must terminate"
assert obs.terminal_grader_score is not None, "terminal score must be set"
assert 0.001 <= obs.terminal_grader_score <= 0.999, \
    f"terminal score out of (0.001, 0.999): {obs.terminal_grader_score}"
print("\n[PASS] Oracle easy_brief pipeline works end-to-end.")