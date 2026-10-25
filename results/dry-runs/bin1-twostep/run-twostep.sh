#!/bin/bash
# Step C of the session-06 plan: two-step S3b (prompt v1.2) with Qwen3-1.7B on the 92-topic copy of the bin-1 dry run.
cd "/Users/alannemirovski/Library/CloudStorage/Dropbox/Research/fifa-reddit-communities" || exit 1
export HF_HOME=$PWD/.hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
OUT=results/dry-runs/bin1-twostep
P=.venv/bin/python; S=code/06-analysis-pipeline.py
echo "=== $(date '+%F %T') start two-step taxonomy ==="
$P $S --stage taxonomy --bin 1 --force --out-root $OUT || { echo "TAXONOMY FAILED"; exit 1; }
echo "=== $(date '+%F %T') collect ==="
$P $S --stage collect --allow-partial --out-root $OUT || { echo "COLLECT FAILED"; exit 1; }
echo "=== $(date '+%F %T') report ==="
$P $S --stage report --bin 1 --out-root $OUT
echo "=== $(date '+%F %T') DONE ==="
