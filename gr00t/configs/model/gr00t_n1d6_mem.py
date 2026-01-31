from dataclasses import MISSING
from typing import Optional

import torch
from transformers import PretrainedConfig

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from . import register_model_config


class Gr00tN1d6MemConfig(Gr00tN1d6Config):
    """Configuration for Gr00tN1d6Mem model with memory module."""

    # Model identification
    model_type: str = "Gr00tN1d6Mem"

    # Memory module configuration
    num_mem_tokens: int = 8  # Number of memory tokens per timestep (N_mem)
    mem_out_tokens: int = 8  # Number of output memory tokens (K)
    mem_max_history_steps: int = 32  # Maximum number of past steps to keep in memory
    mem_num_layers: int = 2  # Number of transformer layers in memory module
    mem_num_heads: int = 8  # Number of attention heads in memory module
    mem_head_dim: int = 64  # Dimension per attention head
    mem_dropout: float = 0.1  # Dropout rate for memory module
    mem_activation_fn: str = "gelu"  # Activation function ("gelu" or "geglu")
    mem_token_str: str = "<|mem|>"  # Special token string for memory tokens
    # Where to insert <|mem|> relative to visual tokens in the prompt.
    # - "after_text": <text ...> <|mem|>... <image...>   (current behavior)
    # - "after_image": <text ...> <image...> <|mem|>...  (mem after visual tokens)
    mem_insert_position: str = "after_image"

    # Training configuration for chunk-based training
    burn_in_steps: int = 4  # Number of burn-in steps (no gradient)
    unroll_steps: int = 12  # Number of unroll steps (with gradient)
    # Note: burn_in_steps + unroll_steps should equal chunk_len from data config

    def __init__(self, **kwargs):
        # Initialize parent config
        super().__init__(**kwargs)
        # Override model_type
        self.model_type = "Gr00tN1d6Mem"

        # Set memory-specific defaults if not provided
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensure memory config fields are set
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())


register_model_config("Gr00tN1d6Mem", Gr00tN1d6MemConfig)
