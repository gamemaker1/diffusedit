#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
revision=0f2787f2d87eac5eed8a087d5ecd24277e6255b2
dest=$SHARE/models/$MODEL_NAME
mkdir -p "$dest"
files=(
    config.json configuration_llada.py generation_config.json modeling_llada.py
    special_tokens_map.json tokenizer.json tokenizer_config.json
    model.safetensors.index.json
    model-0000{1..6}-of-00006.safetensors
)
for f in "${files[@]}"; do
    echo "$(date +%T) $f"
    curl -fL --retry 10 --retry-delay 5 -C - -o "$dest/$f" \
        "https://huggingface.co/$MODEL_ID/resolve/$revision/$f"
done
echo "$(date +%T) done"
du -sh "$dest"
