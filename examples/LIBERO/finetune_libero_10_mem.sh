#!/usr/bin/env bash
set -x -e

# Memory/chunk finetune for LIBERO-10.
# Notes:
# - This uses the mem model pipeline (`Gr00tN1d6Mem`) and chunk trainer.
# - `base_model_path` can be either a mem checkpoint or a base N1d6 checkpoint; if it's a base checkpoint,
#   the pipeline will initialize memory weights and load overlapping weights with strict=False.
#
# Key hyperparameters:
# - state_dropout_prob: 0.3 (lowered from 0.8 to preserve state info for memory learning)
# - episode_sampling_rate: 0.3 (increased from default 0.1 for more training data)
# - chunk_len=16, burn_in=4, unroll=12 (25% burn-in ratio)
# - mem_num_layers: 4 (deeper memory transformer)
export NUM_GPUS=4
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune_mem.py \
    --base_model_path "/mnt/hdd/Fangda/data/models/GR00T/" \
    --dataset_path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ \
    --embodiment_tag LIBERO_PANDA \
    --num_gpus $NUM_GPUS \
    --output_dir /tmp/libero_10_mem \
    --use_wandb \
    \
    `# === Training schedule ===` \
    --max_steps 30000 \
    --warmup_ratio 0.05 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --global_batch_size 16 \
    \
    `# === Checkpointing ===` \
    --save_steps 5000 \
    --save_total_limit 1 \
    --output_dir /mnt/hdd/Fangda/data/models/GR00T/runs/libero_10_mem_1.0 \
    \
    `# === Data loading ===` \
    --dataloader_num_workers 4 \
    --episode_sampling_rate 0.3 \
    \
    `# === Chunk training ===` \
    --chunk_len 24 \
    --burn_in_steps 4 \
    --unroll_steps 20 \
    \
    `# === Memory module ===` \
    --num_mem_tokens 8 \
    --mem_out_tokens 8 \
    --mem_max_history_steps 32 \
    --mem_num_layers 4 \
    --mem_num_heads 8 \
    --mem_head_dim 64 \
    --mem_dropout 0.1 \
    --mem_insert_position "after_image" \
    \
    `# === Regularization ===` \
    --state_dropout_prob 0.3 \
    \
    `# === Memory Visualization (wandb) ===` \
    --log_memory_stats_every_n_steps 100 \
    --log_memory_detail_every_n_steps 100 \
    --log_attention_every_n_steps 100

