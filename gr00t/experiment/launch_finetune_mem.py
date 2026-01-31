# Launch finetuning for the *memory (chunk)* model on a single node.
#
# This is the mem/chunk analogue of `gr00t/experiment/launch_finetune.py`.

import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_mem_config import FinetuneMemConfig
from gr00t.configs.model.gr00t_n1d6_mem import Gr00tN1d6MemConfig
from gr00t.experiment.experiment import run


def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(FinetuneMemConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # All ranks should register the modality config module (if user supplies one).
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    if ft_config.burn_in_steps + ft_config.unroll_steps != ft_config.chunk_len:
        raise ValueError(
            f"burn_in_steps({ft_config.burn_in_steps}) + unroll_steps({ft_config.unroll_steps}) "
            f"must equal chunk_len({ft_config.chunk_len})"
        )

    # Start from default config but swap in the mem model config.
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
                "dataset_mode": "chunk",
                "chunk_len": ft_config.chunk_len,
                "chunk_stride": ft_config.chunk_stride,
            }
        }
    )
    config.load_config_path = None

    # Swap the model to the mem config (this is what selects Gr00tN1d6MemPipeline).
    config.model = Gr00tN1d6MemConfig(
        # Keep the standard backbone defaults, but allow tuning overrides.
        tune_llm=ft_config.tune_llm,
        tune_visual=ft_config.tune_visual,
        tune_projector=ft_config.tune_projector,
        tune_diffusion_model=ft_config.tune_diffusion_model,
        tune_vlln=ft_config.tune_vlln,
        state_dropout_prob=ft_config.state_dropout_prob,
        random_rotation_angle=ft_config.random_rotation_angle,
        color_jitter_params=ft_config.color_jitter_params,
        # Memory/chunk parameters
        num_mem_tokens=ft_config.num_mem_tokens,
        mem_out_tokens=ft_config.mem_out_tokens,
        mem_max_history_steps=ft_config.mem_max_history_steps,
        mem_num_layers=ft_config.mem_num_layers,
        mem_num_heads=ft_config.mem_num_heads,
        mem_head_dim=ft_config.mem_head_dim,
        mem_dropout=ft_config.mem_dropout,
        mem_activation_fn=ft_config.mem_activation_fn,
        mem_token_str=ft_config.mem_token_str,
        mem_insert_position=ft_config.mem_insert_position,
        burn_in_steps=ft_config.burn_in_steps,
        unroll_steps=ft_config.unroll_steps,
    )

    # Common model defaults used by existing finetune script.
    config.model.load_bf16 = True
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.use_flash_attention = True

    # Training config
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.trainer_type = "chunk"
    config.training.wandb_project = "finetune-gr00t-n1d6-mem"
    # Sharded datasets currently do not support evaluation splits (see DatasetFactory assertion).
    config.training.eval_strategy = "no"

    # Memory visualization/logging settings
    config.training.log_memory_stats_every_n_steps = ft_config.log_memory_stats_every_n_steps
    config.training.log_memory_detail_every_n_steps = ft_config.log_memory_detail_every_n_steps
    config.training.log_attention_every_n_steps = ft_config.log_attention_every_n_steps

    # Sharded dataset knobs
    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)

