#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[bootstrap] creating venv (.venv) ..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[bootstrap] upgrading pip (pip3 only) ..."
pip3 install --upgrade pip

echo "[bootstrap] installing project + dev deps ..."
pip3 install -e ".[dev]"

if [ ! -f .env ]; then
  echo "[bootstrap] seeding .env from .env.example ..."
  cp .env.example .env
fi

echo "[bootstrap] running tests ..."
pytest -q

echo "[bootstrap] done."
