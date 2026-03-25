#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARXIV_ID="2603.15726v1"
DB_PATH=""
MODE="local-artifacts"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_demo.sh [arxiv_id] [--mode crawl|local-artifacts] [--db /path/to/demo.db]

Examples:
  bash scripts/run_demo.sh
  bash scripts/run_demo.sh 2603.15726v1 --mode local-artifacts
  bash scripts/run_demo.sh 2603.15726 --mode crawl --db /tmp/briefgpt-demo.db
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --db)
      DB_PATH="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$ARXIV_ID" == "2603.15726v1" ]]; then
        ARXIV_ID="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$DB_PATH" ]]; then
  DB_PATH="$ROOT/demo_${ARXIV_ID}.db"
fi

case "$MODE" in
  crawl|local-artifacts)
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac

echo "Running demo for ${ARXIV_ID}"
echo "Database: ${DB_PATH}"
echo "Mode:     ${MODE}"
echo

cd "$ROOT"

DATABASE_URL="sqlite:///${DB_PATH}" uv run python scripts/run_pipeline.py \
  "$ARXIV_ID" \
  --mode "$MODE"

echo
echo "Database overview"
DATABASE_URL="sqlite:///${DB_PATH}" uv run python scripts/inspect_db.py overview

echo
echo "Paper details"
DATABASE_URL="sqlite:///${DB_PATH}" uv run python scripts/inspect_db.py paper "$ARXIV_ID"
