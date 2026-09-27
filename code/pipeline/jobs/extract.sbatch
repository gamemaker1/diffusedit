#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=30G
#SBATCH -t 4-00:00:00
#SBATCH --signal=B:TERM@900
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source jobs/env.sh
split=$1
shift

bash jobs/stage.sh "$split"
out=$SCRATCH/features/$split
dest=$SHARE/features/$split
mkdir -p "$out"
if [ ! -f "$out/progress.json" ] && ssh ada test -f "$dest/progress.json"; then
    rsync -a "ada:$dest/" "$out/"
fi

push() {
    ssh ada mkdir -p "$dest"
    [ -f "$out/progress.json" ] && cp "$out/progress.json" "$out/.progress.snapshot"
    rsync -a --exclude progress.json --exclude .progress.snapshot "$out/" "ada:$dest/"
    [ -f "$out/.progress.snapshot" ] && rsync -a "$out/.progress.snapshot" "ada:$dest/progress.json"
    return 0
}

( while sleep 3600; do push; done ) &
syncer=$!
HF_HUB_OFFLINE=1 .venv/bin/python -m classifier.extract \
    "$SCRATCH/data/collated/$split.jsonl" "$out" \
    --model "$SCRATCH/models/$MODEL_NAME" "$@" &
worker=$!
trap 'kill -TERM $worker 2>/dev/null; wait $worker; kill $syncer; push; exit 143' TERM
status=0
wait $worker || status=$?
kill $syncer
push
exit $status
