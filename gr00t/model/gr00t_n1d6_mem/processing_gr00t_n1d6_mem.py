import os
from typing import Any, Dict, Literal

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import (
    Gr00tN1d6DataCollator,
    Gr00tN1d6Processor,
    build_processor,
)


class Gr00tN1d6ChunkDataCollator:
    """
    Data collator for chunk-based training.

    Takes a batch of chunks (each chunk is a dict with 'steps' list)
    and collates them into batched sequences.
    """

    def __init__(
        self,
        model_name: str,
        model_type: Literal["eagle"] = "eagle",
        transformers_loading_kwargs: dict = {},
        num_mem_tokens: int = 8,
        mem_token_str: str = "<|mem|>",
    ):
        # IMPORTANT: this collator is called once per batch by the DataLoader.
        # Do NOT instantiate processors/collators inside __call__, otherwise we end up
        # re-loading the HF processor every batch (slow + noisy logs).
        self._step_collator = Gr00tN1d6DataCollator(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        # Keep a reference for backwards compatibility / debugging.
        self.processor = self._step_collator.processor
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name
        self.transformers_loading_kwargs = transformers_loading_kwargs
        self.num_mem_tokens = num_mem_tokens
        self.mem_token_str = mem_token_str

        # Add mem token to tokenizer if not already present
        if mem_token_str not in self.processor.tokenizer.get_vocab():
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [mem_token_str]})
            # Resize token embeddings (will be handled by model initialization)
            self.mem_token_id = self.processor.tokenizer.convert_tokens_to_ids(mem_token_str)
        else:
            self.mem_token_id = self.processor.tokenizer.convert_tokens_to_ids(mem_token_str)

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        """
        Collate a batch of chunks into batched sequences.

        Args:
            features: List of chunk dicts, each containing:
                - episode_id: int
                - start_t: int
                - steps: List[dict] of L processed step dictionaries
                - chunk_len: int

        Returns:
            BatchFeature containing batched sequence data:
                - input_ids_seq: [B, L, ...] (variable length per step, will be padded)
                - attention_mask_seq: [B, L, ...]
                - pixel_values_seq: [B, L, ...]
                - state_seq: [B, L, state_dim]
                - action_seq: [B, L, action_horizon, action_dim]
                - action_mask_seq: [B, L, action_horizon, action_dim]
                - embodiment_id: [B]
                - mem_token_id: int
            And a separate `meta` dict (not passed to the model) containing:
                - episode_ids: list
                - start_ts: list
        """
        batch = {}
        B = len(features)
        L = features[0]["chunk_len"]  # Assume all chunks have same length

        # Extract all steps from all chunks and collate VLM inputs.
        # We need to process each step's vlm_content separately.
        #
        # IMPORTANT:
        # - Some datasets have a variable number of images per step (e.g. missing wrist cam).
        # - HF processors may return `pixel_values` as a Python list with one entry per *image*,
        #   not per sample.
        # To keep alignment correct, we track the per-step image counts and use that to split.
        all_vlm_contents: list[dict] = []
        per_step_image_counts: list[int] = []
        for chunk in features:
            for step in chunk["steps"]:
                if "vlm_content" in step:
                    vlm = step["vlm_content"]
                    all_vlm_contents.append(vlm)
                    try:
                        per_step_image_counts.append(len(vlm.get("images", [])))
                    except Exception:
                        per_step_image_counts.append(0)

        # Collate VLM inputs using the base collator
        if len(all_vlm_contents) > 0:
            step_batch = self._step_collator([{"vlm_content": vlm} for vlm in all_vlm_contents])[
                "inputs"
            ]

            # Reshape from [B*L, ...] to [B, L, ...]
            # Note: sequences may have variable length, so we keep as list or pad to max length
            input_ids = step_batch["input_ids"]  # [B*L, seq_len]
            attention_mask = step_batch["attention_mask"]  # [B*L, seq_len]

            # Reshape to [B, L, seq_len]
            seq_len = input_ids.shape[1]
            batch["input_ids_seq"] = input_ids.view(B, L, seq_len)
            batch["attention_mask_seq"] = attention_mask.view(B, L, seq_len)

            if "pixel_values" in step_batch:
                pixel_values = step_batch["pixel_values"]
                # Some processors return a Python list with one entry per *image* (not per sample).
                # We must split using the true per-step image counts (can be variable).
                if isinstance(pixel_values, list):
                    steps_count = B * L
                    if len(per_step_image_counts) != steps_count:
                        # Fallback to a uniform split (legacy behavior) if bookkeeping failed.
                        per_step_image_counts = [len(pixel_values) // max(steps_count, 1)] * steps_count
                    expected_total = int(sum(per_step_image_counts))
                    assert expected_total == len(pixel_values), (
                        "pixel_values list length does not match sum(per_step_image_counts). "
                        f"len(pixel_values)={len(pixel_values)} expected_total={expected_total} "
                        f"(B={B}, L={L}). This indicates misaligned VLM image packing."
                    )
                    # Split into per-step lists following the original packing order.
                    per_step: list[list[torch.Tensor]] = []
                    cursor = 0
                    for n_img in per_step_image_counts:
                        per_step.append(pixel_values[cursor : cursor + n_img])
                        cursor += n_img
                    batch["pixel_values_seq"] = [per_step[i * L : (i + 1) * L] for i in range(B)]
                else:
                    pv_shape = pixel_values.shape
                    if len(pv_shape) == 4:  # [B*L, C, H, W]
                        batch["pixel_values_seq"] = pixel_values.view(B, L, *pv_shape[1:])
                    elif len(pv_shape) == 3:  # [B*L, H, W] or similar
                        batch["pixel_values_seq"] = pixel_values.view(B, L, *pv_shape[1:])
                    else:
                        batch["pixel_values_seq"] = pixel_values.view(B, L, -1)
        else:
            # No VLM content - create empty tensors
            batch["input_ids_seq"] = None
            batch["attention_mask_seq"] = None
            batch["pixel_values_seq"] = None

        # Collate state, action, action_mask
        state_list = []
        action_list = []
        action_mask_list = []
        embodiment_id_list = []

        for chunk in features:
            chunk_states = []
            chunk_actions = []
            chunk_action_masks = []
            for step in chunk["steps"]:
                # Convert state to tensor if needed
                state = step["state"]
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state)
                chunk_states.append(state)

                if "action" in step:
                    action = step["action"]
                    if isinstance(action, np.ndarray):
                        action = torch.from_numpy(action)
                    chunk_actions.append(action)
                    action_mask = step.get("action_mask", torch.ones_like(action))
                    if isinstance(action_mask, np.ndarray):
                        action_mask = torch.from_numpy(action_mask)
                    chunk_action_masks.append(action_mask)
                else:
                    # Inference mode: no action
                    chunk_actions.append(None)
                    chunk_action_masks.append(None)
                if "embodiment_id" in step:
                    embodiment_id_list.append(step["embodiment_id"])

            state_list.append(torch.stack(chunk_states))  # [L, state_dim]
            if chunk_actions[0] is not None:
                action_list.append(torch.stack(chunk_actions))  # [L, action_horizon, action_dim]
                action_mask_list.append(torch.stack(chunk_action_masks))
            else:
                action_list.append(None)
                action_mask_list.append(None)

        batch["state_seq"] = torch.stack(state_list)  # [B, L, state_dim]
        if action_list[0] is not None:
            batch["action_seq"] = torch.stack(action_list)  # [B, L, action_horizon, action_dim]
            batch["action_mask_seq"] = torch.stack(action_mask_list)
        else:
            batch["action_seq"] = None
            batch["action_mask_seq"] = None

        if embodiment_id_list:
            # All steps in a chunk should have same embodiment_id
            batch["embodiment_id"] = torch.tensor([embodiment_id_list[i * L] for i in range(B)])
        else:
            batch["embodiment_id"] = None

        # Add mem_token_id for model to use
        batch["mem_token_id"] = self.mem_token_id
        
        # DEBUG: Verify mem tokens are in input_ids
        if "input_ids_seq" in batch and batch["input_ids_seq"] is not None:
            input_ids_flat = batch["input_ids_seq"].flatten()
            num_mem_tokens_found = (input_ids_flat == self.mem_token_id).sum().item()
            expected_mem_tokens = B * L * self.num_mem_tokens
            if num_mem_tokens_found == 0:
                import logging
                _logger = logging.getLogger(__name__)
                _logger.warning(
                    f"[BUG] No <|mem|> tokens found in input_ids after tokenization! "
                    f"mem_token_id={self.mem_token_id}, "
                    f"expected {expected_mem_tokens} tokens. "
                    f"Check if tokenizer has <|mem|> token: "
                    f"'{self.mem_token_str}' in vocab = {self.mem_token_str in self.processor.tokenizer.get_vocab()}"
                )

        # Add metadata (kept out of `inputs` to avoid being treated as model tensors)
        meta = {
            "episode_ids": [chunk.get("episode_id", None) for chunk in features],
            "start_ts": [chunk.get("start_t", None) for chunk in features],
        }

        return BatchFeature(data={"inputs": batch, "meta": meta})

    def __str__(self):
        return f"Gr00tN1d6ChunkDataCollator(model_name={self.model_name}, model_type={self.model_type})"


class Gr00tN1d6MemProcessor(Gr00tN1d6Processor):
    """Processor for Gr00tN1d6Mem that supports memory tokens."""

    data_collator_class = Gr00tN1d6ChunkDataCollator

    def __init__(
        self,
        *args,
        num_mem_tokens: int = 8,
        mem_token_str: str = "<|mem|>",
        mem_insert_position: str = "after_image",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_mem_tokens = num_mem_tokens
        self.mem_token_str = mem_token_str
        self.mem_insert_position = mem_insert_position

        # Add mem token to tokenizer
        if mem_token_str not in self.processor.tokenizer.get_vocab():
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [mem_token_str]})
        self.mem_token_id = self.processor.tokenizer.convert_tokens_to_ids(mem_token_str)

    def save_pretrained(self, save_directory):
        """Save processor with mem-specific config."""
        import json
        from pathlib import Path
        
        # Call parent save_pretrained first
        saved_files = super().save_pretrained(save_directory)
        
        # Load and update config with mem-specific settings
        config_file = Path(save_directory) / "processor_config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Update processor_class and add mem-specific kwargs
        config["processor_class"] = "Gr00tN1d6MemProcessor"
        config["processor_kwargs"]["num_mem_tokens"] = self.num_mem_tokens
        config["processor_kwargs"]["mem_token_str"] = self.mem_token_str
        config["processor_kwargs"]["mem_insert_position"] = self.mem_insert_position
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        return saved_files

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> Dict[str, Any]:
        """
        Apply VLM processing with memory tokens inserted.

        Args:
            images: [T, C, H, W] numpy array
            language: Language instruction string

        Returns:
            Dict with vlm_content including mem tokens
        """
        mem_tokens_str = " ".join([self.mem_token_str] * self.num_mem_tokens)

        if self.mem_insert_position == "after_text":
            # Old behavior: <text ... <|mem|>...> then image tokens
            language_with_mem = f"{language} {mem_tokens_str}"
            return super()._apply_vlm_processing(images, language_with_mem)

        if self.mem_insert_position != "after_image":
            raise ValueError(
                f"Unsupported mem_insert_position={self.mem_insert_position}. "
                "Use 'after_text' or 'after_image'."
            )

        # New behavior: keep instruction text first, but append mem tokens as a second text segment
        # AFTER the image content, so <|mem|> appears after visual tokens in the final token sequence.
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": mem_tokens_str},
                ],
            }
        ]

        # Apply chat template but don't tokenize yet - let collator handle it
        if hasattr(self.processor, "py_apply_chat_template"):
            text = self.processor.py_apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
        else:
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )

        if os.environ.get("GROOT_DEBUG_VLM", "0") == "1":
            has_image_tag = "<image-" in text
            print(f"GROOT_DEBUG_VLM: chat_template has_image_tag={has_image_tag}")

        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    @property
    def collator(self):
        """Return chunk collator."""
        if not hasattr(self, "_collator") or self._collator is None:
            self._collator = self.data_collator_class(
                model_name=self.model_name,
                model_type=self.model_type,
                transformers_loading_kwargs=getattr(self, "transformers_loading_kwargs", {}),
                num_mem_tokens=self.num_mem_tokens,
                mem_token_str=self.mem_token_str,
            )
        return self._collator


AutoProcessor.register("Gr00tN1d6Mem", Gr00tN1d6MemProcessor)
