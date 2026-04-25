# POLICY-COMP-01: Company Compliance Policy

## Publication gate
A CEO brief cannot be published when any of the following is true:
1. `data_quality_score` is below 0.70.
2. Strategy recommendations do not cite at least one Analyst metric and one Finance metric.
3. HR memo does not name its intended audience.

## Grounding rule
Every specialist report in the final brief must cite at least one source from `memory/`. This ensures that reasoning is grounded in our SOPs, historical precedents, and compliance policies rather than improvised at rollout time.

## Audit
Citations are programmatically verified against the live `memory/` index. A citation that does not resolve to a real chunk is treated as hallucinated and drops the grounding reward for that specialist.
