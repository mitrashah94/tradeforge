#!/usr/bin/env bash
#
# refresh_data.sh — TradeForge nightly INCREMENTAL data refresh (MASTER_PLAN §5 / P1).
#
# What it does (in order):
#   1. Ingest the last ~5 days of bars from Alpaca   (data.pipelines.alpaca_ingest)
#   2. Rebuild the levels table from those bars       (data.levels)
#   3. Print a quick QQQ validation for sanity        (data.validate; non-fatal)
#
# It is INCREMENTAL and IDEMPOTENT: re-running it only refreshes the recent
# window and rebuilds derived levels, so it is safe to run repeatedly / re-run
# after a failure. It does NOT touch trading or strategy logic.
#
# Requirements:
#   * APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set, normally via a .env
#     file in the repo root (see .env.example). The ingest step reads them.
#   * A Python environment with project deps. If a project .venv exists it is
#     activated automatically.
#
# Invocation:
#   * Run manually:   scripts/refresh_data.sh
#   * Scheduled by the launchd job scripts/cron/com.tradeforge.nightly.plist,
#     which runs it nightly (02:30 local) after market close + overnight settle.
#
set -euo pipefail

# --- Resolve repo root (this script lives in scripts/) and cd there ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "==============================================================="
echo "[refresh_data] START  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[refresh_data] repo:  ${REPO_ROOT}"

# --- Load .env (so APCA_* are available) if present --------------------------
# Export every assignment in .env without clobbering pre-set environment vars
# from the launchd job.
if [ -f .env ]; then
  echo "[refresh_data] loading .env"
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# --- Activate venv if present ------------------------------------------------
if [ -f .venv/bin/activate ]; then
  echo "[refresh_data] activating .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
else
  echo "[refresh_data] no .venv found; using python3 from PATH"
  PY=python3
fi

# --- Sanity: warn (do not hard-fail) if Alpaca creds look unset --------------
if [ -z "${APCA_API_KEY_ID:-}" ] || [ -z "${APCA_API_SECRET_KEY:-}" ]; then
  echo "[refresh_data] WARNING: APCA_API_KEY_ID / APCA_API_SECRET_KEY not set;" \
       "ingest may fail. Populate them in .env (see .env.example)." >&2
fi

# --- 1. Incremental ingest: last ~5 days of bars -----------------------------
echo "[refresh_data] (1/3) ingesting last 5 days of bars ..."
"${PY}" -m data.pipelines.alpaca_ingest --days 5

# --- 2. Rebuild levels from bars --------------------------------------------
echo "[refresh_data] (2/3) rebuilding levels ..."
"${PY}" -m data.levels

# --- 3. Optional quick validation (never fail the run on this) ---------------
echo "[refresh_data] (3/3) QQQ validation (non-fatal) ..."
"${PY}" -m data.validate --symbol QQQ --n 10 || true

echo "[refresh_data] END    $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "==============================================================="
