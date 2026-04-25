"""
Expand each */raw.csv to 400 columns: 8 business columns (used by the analyst
pipeline) + 392 wide-table \"enterprise context\" fields (unused by KPI math).

Re-run:  python3 ceo_brief_env/tasks/_widen_raw_to_400.py
Then:   python3 ceo_brief_env/tasks/_build_ground_truth.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_COLS = 400
CORE = [
    "OrderID",
    "CustomerID",
    "ExpiryDays",
    "Product",
    "Category",
    "Price",
    "Quantity",
    "OrderDate",
]
N_EXTRA = N_COLS - len(CORE)  # 392


def _wide_frame(df: pd.DataFrame, task_seed: int) -> pd.DataFrame:
    """Append dim_0001..dim_0392 with reproducible pseudo-enterprise noise."""
    n = len(df)
    rng = np.random.default_rng(task_seed)
    cols: dict[str, np.ndarray] = {}
    for j in range(N_EXTRA):
        col = f"dim_{j+1:04d}"
        kind = j % 5
        if kind == 0:
            cols[col] = rng.integers(0, 2, size=n)
        elif kind == 1:
            cols[col] = rng.integers(1, 500, size=n)
        elif kind == 2:
            cols[col] = np.round(rng.random(n) * 100.0, 4)
        elif kind == 3:
            cols[col] = rng.integers(2018, 2025, size=n)
        else:
            cols[col] = (rng.integers(0, 1_000_000, size=n) // 1000).astype(np.int64)
    extra = pd.DataFrame(cols, index=df.index)
    return pd.concat([df.reset_index(drop=True), extra], axis=1)


def main() -> int:
    for task_dir in sorted(ROOT.iterdir()):
        if not task_dir.is_dir() or not str(task_dir.name).endswith("_brief"):
            continue
        p = task_dir / "raw.csv"
        if not p.exists():
            continue
        task_name = task_dir.name
        seed = int.from_bytes(task_name.encode(), "big") % (2**31)
        df = pd.read_csv(p)
        # Keep only core columns if re-run on already-wide file
        if len(df.columns) > len(CORE):
            core_present = [c for c in CORE if c in df.columns]
            df = df[core_present]
        missing = [c for c in CORE if c not in df.columns]
        if missing:
            print(f"skip {task_name}: missing columns {missing}", file=sys.stderr)
            continue
        df = df[[c for c in CORE if c in df.columns]]
        assert len(df.columns) == len(CORE)
        wide = _wide_frame(df, seed)
        if len(wide.columns) != N_COLS:
            print(f"column count {len(wide.columns)} != {N_COLS}", file=sys.stderr)
            return 1
        wide.to_csv(p, index=False)
        print(f"wrote {p}  rows={len(wide)}  cols={len(wide.columns)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
