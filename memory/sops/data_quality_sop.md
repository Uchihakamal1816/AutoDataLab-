# SOP-DQ-01: Data Quality Standard Operating Procedure

Owner: Data Analyst desk.

## Scope
All raw order data ingested by the Analyst before KPIs are computed.

## Required steps
1. Remove duplicate rows on (OrderID, SKU).
2. Impute missing prices using the category-level median price.
3. Flag any row whose order date cannot be parsed as ISO-8601.
4. Compute `data_quality_score` as `valid_rows / total_rows`.
5. Report the top revenue category and its share of total revenue.

## Escalation
- Escalate to the Chief of Staff if `data_quality_score < 0.70`.
- The Analyst report must include `duplicates_removed`, `imputed_prices`, and `data_quality_score` metrics.
