#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data data/chroma

if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  source ".env"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
