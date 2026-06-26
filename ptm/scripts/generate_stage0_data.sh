#!/usr/bin/env bash
set -euo pipefail

BACKEND="${PTM_GENERATOR_BACKEND:-minedojo}"
python -m ptm.data.minedojo_generator \
  --out ptm_minedojo_data/stage0 \
  --num_episodes "${PTM_STAGE0_EPISODES:-100}" \
  --frames_per_episode 128 \
  --families balanced \
  --backend "${BACKEND}" \
  --seed 0
