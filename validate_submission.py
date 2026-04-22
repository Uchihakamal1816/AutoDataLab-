#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ROOT / 'openenv.yaml',
    ROOT / 'pyproject.toml',
    ROOT / 'server' / '__init__.py',
    ROOT / 'server' / 'app.py',
    ROOT / 'uv.lock',
    ROOT / 'inference.py',
]


def run(cmd: list[str]) -> int:
    print('$', ' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def check_fastapi_routes() -> int:
    from fastapi.testclient import TestClient
    from server.app import app

    client = TestClient(app)
    checks = [
        ('GET /', client.get('/')),
        ('GET /health', client.get('/health')),
    ]
    reset = client.post('/reset', json={'task': 'easy_brief'})
    checks.append(('POST /reset', reset))
    if reset.status_code == 200:
        episode_id = reset.json()['episode_id']
        checks.append((
            'POST /step',
            client.post('/step', json={'episode_id': episode_id, 'action': {'action_type': 'consult', 'expert_id': 'analyst'}}),
        ))
        checks.append(('GET /state', client.get('/state', params={'episode_id': episode_id})))
    for label, response in checks:
        if response.status_code != 200:
            print(f'FAIL: {label} returned {response.status_code}', file=sys.stderr)
            return 1
    print('ok: FastAPI smoke routes returned 200', flush=True)
    return 0


def main() -> int:
    for path in REQUIRED:
        if not path.exists():
            print(f'missing required file: {path}', file=sys.stderr)
            return 1
    code = run([sys.executable, '-m', 'py_compile', str(ROOT / 'inference.py')])
    if code != 0:
        return code
    code = run([sys.executable, str(ROOT / 'ceo_brief_env' / 'tasks' / '_build_ground_truth.py')])
    if code != 0:
        return code
    code = run([sys.executable, str(ROOT / 'inference.py'), '--oracle'])
    if code != 0:
        return code
    code = check_fastapi_routes()
    if code != 0:
        return code
    openenv = shutil.which('openenv')
    if openenv:
        code = run([openenv, 'validate', '--verbose'])
        if code != 0:
            return code
    print('validate_submission: local checks passed.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
