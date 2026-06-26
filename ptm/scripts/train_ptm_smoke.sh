#!/usr/bin/env bash
set -euo pipefail

python -m ptm.data.verify_dataset --data_root ptm_minedojo_data/stage0
python -m ptm.train_ptm \
  --data_root ptm_minedojo_data/stage0 \
  --output_dir outputs/ptm_smoke \
  --batch_size "${PTM_SMOKE_BATCH_SIZE:-4}" \
  --epochs "${PTM_SMOKE_EPOCHS:-1}" \
  --max_steps "${PTM_SMOKE_MAX_STEPS:-50}" \
  --memory_dim 256 \
  --num_memory_tokens 16 \
  --num_layers 2
