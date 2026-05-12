#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${1:-$HOME/bert/dataset}"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset não encontrado: $DATASET_DIR" >&2
    exit 1
fi

docker run --rm \
    --link pgs_cohmetrix:pgs_cohmetrix \
    -v "$SCRIPT_DIR":/opt/text_metrics \
    -v "$DATASET_DIR":/data \
    cohmetrix:focal \
    bash -c "cd /opt/text_metrics && python3 run_batch.py"
