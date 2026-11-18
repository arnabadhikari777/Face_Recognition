#!/usr/bin/env sh
# Downloads and verifies every model in models/model_manifest.json (SHA-256, size, ONNX header). Fails the build on any problem.
# Logic lives in scripts/download_models.py (Python standard library only).
set -eu
cd "$(dirname "$0")/.."
exec python3 scripts/download_models.py "$@"
