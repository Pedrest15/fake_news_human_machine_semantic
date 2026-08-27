#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${1:-$HOME/bert/dataset}"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset not found: $DATASET_DIR" >&2
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/nilcmetrix" ]; then
    echo "Subfolder nilcmetrix/ not found in $SCRIPT_DIR" >&2
    exit 1
fi

docker run --rm \
    --link pgs_cohmetrix:pgs_cohmetrix \
    -v "$SCRIPT_DIR/nilcmetrix":/opt/text_metrics \
    -v "$SCRIPT_DIR":/opt/wrapper \
    -v "$DATASET_DIR":/data \
    cohmetrix:focal \
    bash -c "cd /opt/text_metrics && PYTHONPATH=/opt/text_metrics python3 /opt/wrapper/run_batch.py"
