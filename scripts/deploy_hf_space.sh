#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set" >&2
  exit 1
fi

echo "1. Login to Hugging Face"
hf auth login --token "$HF_TOKEN"

echo "2. Create or link your Docker Space manually, then push this repo"
echo "   hf repo create <space-name> --type space --space_sdk docker"
echo "   git remote add origin https://huggingface.co/spaces/<user>/<space-name>"
echo "   git push origin HEAD"

echo "3. After the Space is live, validate endpoints:"
echo "   curl https://<space-url>/health"
echo "   curl -X POST https://<space-url>/reset -H "content-type: application/json" -d '{"task":"easy_brief"}'"
