#!/usr/bin/env bash
# Run the Gate-3 structural-loss ablation ladder sequentially, then evaluate
# each variant on the corruption suite.  ~8 hours total on the GB10.
set -euo pipefail
cd "$(dirname "$0")/.."

for variant in task-only plus-recon plus-chain full; do
  echo "=== ablation variant: ${variant} ==="
  .venv/bin/homymoly train --config "configs/gate3/${variant}.yaml" --resume
  echo "=== corruption evaluation: ${variant} ==="
  .venv/bin/python scripts/eval_corruption.py \
    --checkpoint "artifacts/gate3/${variant}/checkpoints/last.pt" \
    --config "configs/gate3/${variant}.yaml" \
    --output "artifacts/gate3/${variant}/corruption_report.json" \
    --max-batches 6
done
echo "=== ablation ladder complete ==="
