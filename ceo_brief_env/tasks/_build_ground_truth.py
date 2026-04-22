from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ceo_brief_env.experts.data_analyst import DataAnalystExpert
from ceo_brief_env.experts.finance import FinanceExpert
from ceo_brief_env.experts.hr import HRExpert
from ceo_brief_env.experts.strategy import StrategyExpert

ROOT = Path(__file__).resolve().parent


def build_task(task_name: str) -> None:
    task_dir = ROOT / task_name
    raw_df = pd.read_csv(task_dir / 'raw.csv')
    meta = json.loads((task_dir / 'metadata.json').read_text(encoding='utf-8'))
    analyst = DataAnalystExpert().run(task_name, meta['instruction'], raw_df)
    finance = FinanceExpert().run(task_name, meta['instruction'], raw_df, analyst.metrics, meta)
    strategy = None
    if 'strategy' in meta.get('required_experts', []):
        strategy = StrategyExpert().run(task_name, meta, analyst, finance)
    hr = HRExpert().run(task_name, meta, analyst, finance, strategy)
    metrics = {
        'data_quality_score': analyst.metrics['data_quality_score'],
        'total_revenue': analyst.metrics['total_revenue'],
        'top_category': analyst.metrics['top_category'],
        'projection_next_quarter': finance.metrics['projection_next_quarter'],
        'variance_pct': finance.metrics['variance_pct'],
        'break_even_units': finance.metrics['break_even_units'],
        'memo_score': hr.metrics['memo_score'],
    }
    rows = [{'metric': k, 'value': v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(task_dir / 'ground_truth.csv', index=False)


def main() -> int:
    for task_dir in ROOT.iterdir():
        if task_dir.is_dir() and task_dir.name.endswith('_brief'):
            build_task(task_dir.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
