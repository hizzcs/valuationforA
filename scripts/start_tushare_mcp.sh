#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCP_DIR="${ROOT_DIR}/tools/mcp/tushare_MCP"
PY_BIN="${MCP_DIR}/.venv/bin/python"
SERVER_FILE="${MCP_DIR}/server.py"
PROJECT_ENV="${ROOT_DIR}/.env"

if [[ ! -d "${MCP_DIR}" ]]; then
  echo "MCP directory not found: ${MCP_DIR}" >&2
  echo "Please deploy first: git clone https://github.com/zhewenzhang/tushare_MCP.git ${MCP_DIR}" >&2
  exit 1
fi

if [[ ! -x "${PY_BIN}" ]]; then
  echo "Python runtime not found: ${PY_BIN}" >&2
  echo "Please install dependencies in ${MCP_DIR} first." >&2
  exit 1
fi

if [[ -f "${PROJECT_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${PROJECT_ENV}"
  set +a
fi

if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  echo "TUSHARE_TOKEN is missing. Set it in ${PROJECT_ENV} or environment." >&2
  exit 1
fi

exec "${PY_BIN}" "${SERVER_FILE}"
