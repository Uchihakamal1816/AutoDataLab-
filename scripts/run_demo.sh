#!/usr/bin/env bash
set -euo pipefail

python3 inference.py --ablation
python3 inference.py --oracle --task medium_brief
