"""Finetune config for memory (chunk) post-training on a single node.

This mirrors `gr00t/configs/finetune_config.py` but adds:
- chunk training controls (chunk_len/stride, burn_in/unroll)
- memory module controls (num_mem_tokens, etc.)

Used by `gr00t/experiment/launch_finetune_mem.py`.
"""

from dataclasses import dataclass

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneMemConfig:
    # --- Data and Model Paths ---
    base_model_path: str
    dataset_path: str
    embodiment_tag: EmbodimentTag
    modality_config_path: str | None = None

    # --- Model tuning flags ---
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    state_dropout_prob: float = 0.0

    # --- Data augmentation ---
    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None

    # --- Chunk training (must match data.chunk_len) ---
    chunk_len: int = 16
    chunk_stride: int = 1
    burn_in_steps: int = 4
    unroll_steps: int = 12

    # --- Memory module configuration ---
    num_mem_tokens: int = 8
    mem_out_tokens: int = 8
    mem_max_history_steps: int = 32
    mem_num_layers: int = 4
    mem_num_heads: int = 8
    mem_head_dim: int = 64
    mem_dropout: float = 0.1
    mem_activation_fn: str = "gelu"
    mem_token_str: str = "<|mem|>"
    mem_insert_position: str = "after_image"

    # --- Training configuration ---
    global_batch_size: int = 32
    dataloader_num_workers: int = 2
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    output_dir: str = "./outputs"
    save_steps: int = 1000
    save_total_limit: int = 5
    num_gpus: int = 1
    use_wandb: bool = True
    max_steps: int = 10000
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    # --- Sharded dataset controls ---
    shard_size: int = 2**6  # chunk shards are smaller by default
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e3)

    # --- Memory Visualization & Logging ---
    log_memory_stats_every_n_steps: int = 100
    """Log core memory stats (out_norm, loss_trend) every N steps. Set 0 to disable."""
    log_memory_detail_every_n_steps: int = 500
    """Log detailed memory diagnostics (cosine_sim, etc.) every N steps. Set 0 to disable."""
    log_attention_every_n_steps: int = 500
    """Log attention heatmaps to wandb every N steps. Set 0 to disable (saves compute)."""