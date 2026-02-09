#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
GATEWAY_URL="http://69.63.199.52:38011/api/v1/gtst"
API_KEY=""
DRY_RUN="false"
PRINT_EXPORT="false"
TIMEOUT="8"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/refresh_tushare_token.sh --key <gateway_key> [options]

Options:
  --key <k>            Gateway key (required)
  --url <u>            Token gateway URL
  --env-file <path>    Target .env file path
  --timeout <sec>      HTTP timeout seconds (default: 8)
  --dry-run            Fetch token only, do not write .env
  --print-export       Print: export TUSHARE_TOKEN=...
  -h, --help           Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key)
      API_KEY="${2:-}"
      shift 2
      ;;
    --url)
      GATEWAY_URL="${2:-}"
      shift 2
      ;;
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --print-export)
      PRINT_EXPORT="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${API_KEY}" ]]; then
  echo "--key is required." >&2
  usage
  exit 2
fi

TOKEN="$(curl -fsS --max-time "${TIMEOUT}" "${GATEWAY_URL}?k=${API_KEY}" | tr -d '\r\n\t ')"

if [[ -z "${TOKEN}" ]]; then
  echo "Fetched token is empty." >&2
  exit 1
fi

if [[ ! "${TOKEN}" =~ ^[0-9a-fA-F]{32,}$ ]]; then
  echo "Fetched token format looks suspicious: ${TOKEN}" >&2
  exit 1
fi

if [[ "${PRINT_EXPORT}" == "true" ]]; then
  echo "export TUSHARE_TOKEN=${TOKEN}"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run only. Token fetched successfully."
  exit 0
fi

mkdir -p "$(dirname "${ENV_FILE}")"
if [[ ! -f "${ENV_FILE}" ]]; then
  touch "${ENV_FILE}"
fi

if rg -q '^TUSHARE_TOKEN=' "${ENV_FILE}"; then
  sed -i.bak "s/^TUSHARE_TOKEN=.*/TUSHARE_TOKEN=${TOKEN}/" "${ENV_FILE}"
else
  {
    echo ""
    echo "TUSHARE_TOKEN=${TOKEN}"
  } >> "${ENV_FILE}"
fi

echo "Updated token in ${ENV_FILE}"
