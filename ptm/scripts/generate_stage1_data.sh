#!/usr/bin/env bash
set -euo pipefail

BACKEND="${PTM_GENERATOR_BACKEND:-minedojo}"
python -m ptm.data.minedojo_generator \
  --out ptm_minedojo_data/stage1 \
  --num_episodes "${PTM_STAGE1_EPISODES:-3000}" \
  --frames_per_episode 128 \
  --families balanced \
  --backend "${BACKEND}" \
  --seed 100
