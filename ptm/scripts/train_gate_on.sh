#!/bin/bash
set -e
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /gfs/space/private/zjc/.secrets/wandb_api_key 2>/dev/null || true)}"
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WANDB_API_KEY is not set and /gfs/space/private/zjc/.secrets/wandb_api_key is missing" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /gfs/space/private/zjc/ptm

/gfs/space/private/zjc/envs/worldmem/bin/python -m main \
  +name=ptm_gate_on_targetloss_10k \
  +output_dir=outputs/ptm_gate_on_targetloss_10k \
  +diffusion_model_path=/gfs/space/private/zjc/models/oasis-500m/oasis500m.safetensors \
  +vae_path=/gfs/space/private/zjc/models/oasis-500m/vit-l-20.safetensors \
  +customized_load=true +seperate_load=true +zero_init_gate=true \
  dataset=ptm_minedojo \
  dataset.save_dir=ptm_minedojo_data/long_1500_360x640 \
  dataset.n_frames=8 dataset.context_length=4 dataset.future_length=4 \
  dataset.memory_condition_length=8 dataset.max_history_candidates=16 \
  dataset.ptm_context_length=4 dataset.ptm_future_length=4 \
  +dataset.n_frames_valid=700 +dataset.ptm_context_length_valid=600 +dataset.ptm_future_length_valid=100 \
  +dataset.npz_cache_dir=/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640_npz_cache \
  +dataset.npz_cache_splits=[train,val] \
  +dataset.npz_cache_dir_val=/gfs/space/private/zjc/ptm/ptm_minedojo_data/gen600x100_npz_cache \
  +dataset.video_cache_size=0 \
  algorithm.context_frames=600 algorithm.num_memory_tokens=16 \
  algorithm.x_shape=[3,360,640] ++algorithm.metrics=[lpips,psnr] \
  ++algorithm.memory_condition_length=8 ++algorithm.use_ptm_memory=true \
  ++algorithm.use_ptm_reference_adapter=true ++algorithm.use_memory_attention=false \
  ++algorithm.ptm_loss_weight=0.1 ++algorithm.ptm_bottleneck_weight=0.001 \
  ++algorithm.ptm_eval_only=false ++algorithm.ptm_max_history=16 ++algorithm.ptm_max_history_candidates=16 \
  ++algorithm.generation_target_loss_weight=1.0 ++algorithm.generation_late_loss_weight=0.5 \
  ++algorithm.generation_target_window_radius=1 ++algorithm.generation_late_horizon_start=50 \
  ++algorithm.log_video=true ++algorithm.max_log_videos=1 \
  ++algorithm.use_ptm_cross_attention=true \
  ++algorithm.local_save_dir="outputs/ptm_gate_on_targetloss_10k" \
  ++algorithm.validation_ablation_modes=[normal,zero,hard_shuffle] \
  experiment.tasks=[training] \
  experiment.training.max_steps=10000 \
  experiment.training.checkpointing.every_n_train_steps=2500 \
  wandb.project=ptm wandb.entity=jinczhu12-hkust
