#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$SCRATCH/models" "$SCRATCH/data/collated"
rsync -a "ada:$SHARE/models/$MODEL_NAME/" "$SCRATCH/models/$MODEL_NAME/"
for split in "$@"; do
    rsync -a "ada:$SHARE/data/collated/$split.jsonl" "$SCRATCH/data/collated/"
done
