# AutoDataLab++

OpenEnv **multi-agent** environment: a **Chief of Staff** routes work across **Data Analyst**, **Finance**, **Strategy**, and **HR** to complete CEO briefing tasks. Includes optional **REINFORCE** training for the CoS and a **FastAPI** server with an **office UI** at `/ui/`.

## Quickstart (local)

```bash
pip install -e .
# or: uv sync && uv run server
python3 -m server.app
```

- API root: <http://127.0.0.1:7860/>  
- Health: <http://127.0.0.1:7860/health>  
- **Demo UI:** <http://127.0.0.1:7860/ui/>

Pre-submission checks:

```bash
python3 validate_submission.py
openenv validate --verbose
```

Oracle rollout (3 tasks):

```bash
python3 inference.py --oracle
```

## Deployment

### Environment variables

| Variable | Purpose |
| -------- | ------- |
| `API_BASE_URL` | OpenAI-compatible endpoint (default: Hugging Face router) |
| `API_KEY` or `HF_TOKEN` | For LLM CoS; if unset, `inference.py` uses **oracle** |
| `MODEL_NAME` | Model id for the CoS LLM path |
| `AUTODATALAB_PLUS_TASKS` | Comma-separated task ids (default: all three briefs) |

Copy `.env.example` to `.env` and adjust. For Docker / Spaces, set secrets in the platform UI.

### Docker

Build and run (port **7860** matches `openenv.yaml`):

```bash
docker build -t autodatalab-plus .
# First build can take several minutes: PyTorch + OpenEnv stack download ~1.5GB+ of wheels.
docker run --rm -p 7860:7860 autodatalab-plus
```

Smoke test:

```bash
curl -s http://127.0.0.1:7860/health
```

The image includes a **HEALTHCHECK** on `/health`. Training checkpoints under `training/checkpoints/` (if present in the build context) are included so the **trained CoS** policy is available in `/visualize/run` when a checkpoint exists.

### Hugging Face Space (OpenEnv)

- Type: **Docker** or use the `openenv.yaml` `app: server.app:app` + `port: 7860` as documented in the [OpenEnv](https://huggingface.co/docs/hub/en/spaces) flow you use for the hackathon.  
- Ensure **build context** is this repo; **do not** commit large secrets.  
- After deploy: hit `/health`, open `/ui/`, run `python3 validate_submission.py` against the **live URL** (adjust `ROOT` or use env if your script supports it).

## Project layout (high level)

- `ceo_brief_env/` — Pydantic models, environment, graders, `tasks/`
- `inference.py` — oracle / baselines / LLM / trained CoS, `[START]`/`[STEP]`/`[END]` logs
- `server/app.py` — FastAPI; `/reset`, `/step`, `/state`, `/visualize/run`
- `training/` — `train_cos_local.py`, checkpoints and curves (optional)
- `subenvs/` — analyst + email/HR tools

## License

Hackathon / team use per repository owner.
