#!/usr/bin/env bash
set -euo pipefail

docker build -t autodatalab-pp-local .
docker run --rm -p 7860:7860 autodatalab-pp-local
