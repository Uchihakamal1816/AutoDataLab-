# Deploying AutoDataLab++ to a Hugging Face Space (OpenEnv server)

Goal: expose `/reset` and `/step` on a public URL so the Colab GRPO notebook can train against it.

## 1. Create the Space

1. Go to https://huggingface.co/new-space
2. **Space SDK**: Docker
3. **Hardware**: CPU basic is fine (the env is tiny; the GPU work happens in Colab)
4. **Visibility**: Public (so Colab can reach it without auth) — or Private + pass `HF_TOKEN` to the trainer

## 2. Push the code

From the repo root:

```bash
cd autodatalab-plus
# one-time
huggingface-cli login            # paste your HF token (write scope)
git init && git lfs install
git remote add space https://huggingface.co/spaces/<your-username>/autodatalab-plus
# every push
git add . && git commit -m "deploy"
git push space HEAD:main
```

The provided `Dockerfile` already exposes port `7860` and runs `python -m server.app`, which is exactly what HF Spaces expects.

## 3. Prepend this YAML block to `README.md` on the Space

(Only needed on the Space copy — keep your local `README.md` clean.)

```yaml
---
title: AutoDataLab++
emoji: 🧑‍💼
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
```

## 4. Verify

```bash
curl https://<your-username>-autodatalab-plus.hf.space/health
# {"status":"healthy"}

curl -X POST https://<your-username>-autodatalab-plus.hf.space/reset \
     -H 'content-type: application/json' \
     -d '{"task":"easy_brief","use_rag":false}'
```

Copy that base URL — you will paste it into the Colab notebook
(`training/cos_grpo_colab.ipynb`) as `BASE_URL`.
