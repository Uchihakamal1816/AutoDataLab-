#!/usr/bin/env python3
"""Submission validator for AutoDataLab++.

Two modes:

  Local mode (default):
      python validate_submission.py
      Runs file-presence checks, py_compile, ground-truth rebuild,
      `inference.py --oracle`, in-process FastAPI smoke, and openenv
      validate. Used pre-deploy.

  Live mode (--base-url):
      python validate_submission.py --base-url https://your-space.hf.space
      Hits the deployed HTTP endpoints (/, /health, /reset, /step, /state)
      to confirm the live deployment responds correctly. Used post-deploy.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED = [
    ROOT / "openenv.yaml",
    ROOT / "pyproject.toml",
    ROOT / "server" / "__init__.py",
    ROOT / "server" / "app.py",
    ROOT / "uv.lock",
    ROOT / "inference.py",
]


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def check_fastapi_routes_local() -> int:
    """Run smoke routes via in-process TestClient. Used in local mode."""
    from fastapi.testclient import TestClient
    from server.app import app

    client = TestClient(app)
    checks = [
        ("GET /", client.get("/")),
        ("GET /health", client.get("/health")),
    ]
    reset = client.post("/reset", json={"task": "easy_brief"})
    checks.append(("POST /reset", reset))
    if reset.status_code == 200:
        episode_id = reset.json()["episode_id"]
        checks.append((
            "POST /step",
            client.post(
                "/step",
                json={
                    "episode_id": episode_id,
                    "action": {"action_type": "consult", "expert_id": "analyst"},
                },
            ),
        ))
        checks.append((
            "GET /state",
            client.get("/state", params={"episode_id": episode_id}),
        ))
    for label, response in checks:
        if response.status_code != 200:
            print(f"FAIL: {label} returned {response.status_code}", file=sys.stderr)
            return 1
    print("ok: FastAPI smoke routes returned 200", flush=True)
    return 0


def check_fastapi_routes_live(base_url: str, timeout: float = 30.0) -> int:
    """Run smoke routes via real HTTP against a deployed base URL."""
    import urllib.error
    import urllib.request
    import json as _json

    base = base_url.rstrip("/")

    def _get(path: str):
        req = urllib.request.Request(base + path, method="GET")
        return urllib.request.urlopen(req, timeout=timeout)

    def _post(path: str, payload: dict):
        body = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=timeout)

    try:
        # GET /
        with _get("/") as r:
            if r.status != 200:
                print(f"FAIL: GET / returned {r.status}", file=sys.stderr)
                return 1
        print("ok: GET / -> 200", flush=True)

        # GET /health
        with _get("/health") as r:
            if r.status != 200:
                print(f"FAIL: GET /health returned {r.status}", file=sys.stderr)
                return 1
        print("ok: GET /health -> 200", flush=True)

        # POST /reset
        with _post("/reset", {"task": "easy_brief"}) as r:
            if r.status != 200:
                print(f"FAIL: POST /reset returned {r.status}", file=sys.stderr)
                return 1
            reset_payload = _json.loads(r.read().decode("utf-8"))
        print("ok: POST /reset -> 200", flush=True)

        episode_id = reset_payload.get("episode_id")
        if not episode_id:
            print("FAIL: /reset response missing episode_id", file=sys.stderr)
            return 1

        # POST /step
        with _post(
            "/step",
            {
                "episode_id": episode_id,
                "action": {"action_type": "consult", "expert_id": "analyst"},
            },
        ) as r:
            if r.status != 200:
                print(f"FAIL: POST /step returned {r.status}", file=sys.stderr)
                return 1
        print("ok: POST /step -> 200", flush=True)

        # GET /state?episode_id=...
        with _get(f"/state?episode_id={episode_id}") as r:
            if r.status != 200:
                print(f"FAIL: GET /state returned {r.status}", file=sys.stderr)
                return 1
        print("ok: GET /state -> 200", flush=True)

        print("ok: live HTTP smoke routes returned 200", flush=True)
        return 0

    except urllib.error.HTTPError as e:
        print(f"FAIL: HTTPError {e.code} for {e.url}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"FAIL: URLError {e.reason} (network or DNS)", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: live HTTP smoke crashed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-url",
        default=None,
        help="If set, run live HTTP smoke checks against this URL instead of local checks.",
    )
    args = parser.parse_args()

    # ----- LIVE MODE: only HTTP smoke against the deployed URL -----
    if args.base_url:
        print(f"validate_submission: live mode against {args.base_url}", flush=True)
        return check_fastapi_routes_live(args.base_url)

    # ----- LOCAL MODE: full suite -----
    for path in REQUIRED:
        if not path.exists():
            print(f"missing required file: {path}", file=sys.stderr)
            return 1

    code = run([sys.executable, "-m", "py_compile", str(ROOT / "inference.py")])
    if code != 0:
        return code

    code = run([sys.executable, str(ROOT / "ceo_brief_env" / "tasks" / "_build_ground_truth.py")])
    if code != 0:
        return code

    code = run([sys.executable, str(ROOT / "inference.py"), "--oracle"])
    if code != 0:
        return code

    code = check_fastapi_routes_local()
    if code != 0:
        return code

    openenv = shutil.which("openenv")
    if openenv:
        code = run([openenv, "validate", "--verbose"])
        if code != 0:
            return code

    print("validate_submission: local checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())