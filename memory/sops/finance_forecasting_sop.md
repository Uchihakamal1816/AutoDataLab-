# SOP-FIN-01: Finance Forecasting Standard Operating Procedure

Owner: Finance desk.

## Scope
Next-quarter revenue projection, plan-versus-actual variance, and break-even analysis for every CEO brief.

## Required steps
1. Build monthly revenue from cleaned orders via `monthly_revenue`.
2. Project next quarter using the trailing three months.
3. Apply a confidence band of plus-or-minus 15 percent around the projection.
4. Compare actual revenue to `plan_value` and produce `variance_abs`, `variance_pct`, and `variance_flag`.
   - favorable if variance is above +2 percent
   - on-plan if variance is within plus-or-minus 2 percent
   - unfavorable if variance is below -2 percent
5. Compute break-even units as `fixed_cost / unit_margin`.

## Output contract
The Finance report must include `projection_next_quarter`, `confidence_band`, `variance_abs`, `variance_pct`, `variance_flag`, and `break_even_units`.
