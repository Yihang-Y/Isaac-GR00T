import json
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6_mem.gr00t_n1d6_mem import Gr00tN1d6Mem
from gr00t.configs.model.gr00t_n1d6_mem import Gr00tN1d6MemConfig
from gr00t.model.gr00t_n1d6_mem.processing_gr00t_n1d6_mem import Gr00tN1d6MemProcessor
from gr00t.model.registry import register_model
import numpy as np
from termcolor import colored
import torch
from transformers import AutoModel, AutoProcessor


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d6MemPipeline(ModelPipeline):
    model_class = Gr00tN1d6Mem
    processor_class = Gr00tN1d6MemProcessor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""

        if self.config.training.start_from_checkpoint is not None:
            loaded_model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            # If the checkpoint is NOT a mem model (e.g., base N1d6), convert it into a mem model
            # by initializing Gr00tN1d6Mem from config and loading overlapping weights.
            if not isinstance(loaded_model, Gr00tN1d6Mem):
                logging.warning(
                    "Checkpoint model is not Gr00tN1d6Mem; initializing Gr00tN1d6Mem and loading weights with strict=False. "
                    "This is expected if starting from a non-mem base model."
                )
                base_state = loaded_model.state_dict()
                model = self.model_class(
                    self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
                )
                missing, unexpected = model.load_state_dict(base_state, strict=False)
                logging.info(
                    f"Loaded base checkpoint into mem model. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
                )
                # For downstream logic that checks missing_keys (e.g. mask_token), emulate HF loading_info.
                loading_info = {"missing_keys": missing, "unexpected_keys": unexpected}
            else:
                model = loaded_model

            # Initialize mask_tokens if they are not present in the base checkpoint
            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)

            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            # Guard against corrupted checkpoints / dtype-cast issues: re-init if non-finite.
            if model.action_head.mask_token is not None:
                mt = model.action_head.mask_token
                try:
                    if torch.is_floating_point(mt) and (torch.isnan(mt).any() or torch.isinf(mt).any()):
                        logging.warning(
                            "[NaN DEBUG][setup] mask_token contains NaN/Inf right after load/init. Reinitializing."
                        )
                        with torch.no_grad():
                            mt.data.copy_(0.02 * torch.randn_like(mt))
                except Exception as e:
                    logging.warning(f"[NaN DEBUG][setup] failed to validate/reinit mask_token: {e}")

            # Resize token embeddings if mem token was added
            logging.info(f"[DEBUG] Checking resize_token_embeddings: hasattr={hasattr(model.backbone.model, 'resize_token_embeddings')}")
            if hasattr(model.backbone.model, "resize_token_embeddings"):
                processor = AutoProcessor.from_pretrained(
                    self.config.training.start_from_checkpoint,
                    **self.transformers_loading_kwargs,
                )
                # Assume our GR00T processor wrapper (no compatibility fallbacks).
                tokenizer = processor.processor.tokenizer
                
                # Check if we need to add the mem token
                need_resize = False
                mem_in_vocab = self.model_config.mem_token_str in tokenizer.get_vocab()
                logging.info(f"[DEBUG] mem_token_str='{self.model_config.mem_token_str}' in vocab: {mem_in_vocab}")
                
                if not mem_in_vocab:
                    # First add the mem token to the tokenizer
                    tokenizer.add_special_tokens(
                        {"additional_special_tokens": [self.model_config.mem_token_str]}
                    )
                    # Now resize embeddings to the NEW length (includes mem token)
                    model.backbone.model.resize_token_embeddings(len(tokenizer))
                    need_resize = True
                    # CRITICAL: Update config vocab_size to match resized embedding
                    # This ensures saved config.json matches the actual model weights
                    if hasattr(model.backbone.model, "config"):
                        model.backbone.model.config.vocab_size = len(tokenizer)
                    if hasattr(model.backbone.model.language_model, "config"):
                        model.backbone.model.language_model.config.vocab_size = len(tokenizer)
                    logging.info(
                        f"Added mem token '{self.model_config.mem_token_str}' and resized "
                        f"embeddings to {len(tokenizer)}. Updated config vocab_size to {len(tokenizer)}"
                    )
                
                # ===== CRITICAL FIX: Initialize <|mem|> with IMAGE TOKEN embedding =====
                # 
                # Problem: VLM backbone has 12 frozen layers (0-11), only layers 12-15 are trainable.
                # If <|mem|> has random/eos embedding, frozen layers don't know how to process it,
                # so <|mem|> only "self-attends" and its hidden state doesn't change across layers.
                # 
                # Solution: Initialize <|mem|> with IMAGE TOKEN embedding!
                # This way, frozen layers will treat <|mem|> similar to image tokens,
                # allowing it to participate in the visual attention patterns.
                # 
                # NOTE: We do this EVERY time to ensure consistent behavior.
                logging.info("[DEBUG] Starting mem token embedding initialization with IMAGE TOKEN...")
                try:
                    logging.info(f"[DEBUG] model.backbone.model type: {type(model.backbone.model)}")
                    logging.info(f"[DEBUG] Trying to access: model.backbone.model.language_model.model.embed_tokens")
                    embed_layer = model.backbone.model.language_model.model.embed_tokens
                    logging.info(f"[DEBUG] embed_layer shape: {embed_layer.weight.shape}")
                    mem_token_id = tokenizer.convert_tokens_to_ids(self.model_config.mem_token_str)
                    logging.info(f"[DEBUG] mem_token_id: {mem_token_id}")
                    
                    # Log BEFORE initialization
                    old_mem_emb = embed_layer.weight[mem_token_id].detach().clone()
                    old_norm = old_mem_emb.float().norm().item()
                    old_mean = old_mem_emb.float().mean().item()
                    logging.info(
                        f"[MEM INIT] BEFORE: <|mem|> token (id={mem_token_id}) "
                        f"embedding norm={old_norm:.4f}, mean={old_mean:.6f}"
                    )
                    
                    # ===== KEY CHANGE: Use IMAGE TOKEN embedding instead of eos_token =====
                    # This allows frozen layers to treat <|mem|> like image tokens
                    source_token_id = model.backbone.model.config.image_token_index
                    source_token_name = "IMAGE_TOKEN"
                    logging.info(f"[MEM INIT] Using IMAGE TOKEN (id={source_token_id}) as source")
                    
                    # Fallback chain if image_token_index is not available
                    if source_token_id is None or source_token_id >= embed_layer.weight.shape[0]:
                        logging.warning(f"[MEM INIT] image_token_index invalid ({source_token_id}), falling back to eos_token")
                        source_token_id = tokenizer.eos_token_id
                        source_token_name = "EOS_TOKEN"
                    if source_token_id is None:
                        source_token_id = tokenizer.pad_token_id
                        source_token_name = "PAD_TOKEN"
                    if source_token_id is None:
                        source_token_id = 1
                        source_token_name = "TOKEN_1"
                    
                    # Log source token embedding
                    source_emb = embed_layer.weight[source_token_id].detach()
                    source_norm = source_emb.float().norm().item()
                    source_mean = source_emb.float().mean().item()
                    logging.info(
                        f"[MEM INIT] SOURCE: token id={source_token_id} "
                        f"embedding norm={source_norm:.4f}, mean={source_mean:.6f}"
                    )
                    
                    with torch.no_grad():
                        source_embedding = embed_layer.weight[source_token_id].clone()
                        # Add small noise to break symmetry if multiple <|mem|> tokens
                        noise = torch.randn_like(source_embedding) * 0.01
                        embed_layer.weight[mem_token_id] = source_embedding + noise
                    
                    # Log AFTER initialization
                    new_mem_emb = embed_layer.weight[mem_token_id].detach()
                    new_norm = new_mem_emb.float().norm().item()
                    new_mean = new_mem_emb.float().mean().item()
                    logging.info(
                        f"[MEM INIT] AFTER: <|mem|> token (id={mem_token_id}) "
                        f"embedding norm={new_norm:.4f}, mean={new_mean:.6f}"
                    )
                    
                    # Verify the change
                    if abs(new_norm - source_norm) < 0.1:
                        logging.info(
                            f"[MEM INIT] ✓ SUCCESS: <|mem|> embedding initialized from {source_token_name} (id={source_token_id})! "
                            f"norm changed: {old_norm:.4f} -> {new_norm:.4f}"
                        )
                    else:
                        logging.warning(
                            f"[MEM INIT] ✗ WARNING: Initialization may have failed! "
                            f"Expected norm ~{source_norm:.4f}, got {new_norm:.4f}. "
                            f"Source: {source_token_name} (id={source_token_id})"
                        )
                    
                except Exception as e:
                    import traceback
                    logging.error(
                        f"[MEM INIT ERROR] Could not initialize <|mem|> embedding from existing token: {e}. "
                        f"Memory module may not attend to image features properly!"
                    )
                    logging.error(f"[MEM INIT ERROR] Traceback:\n{traceback.format_exc()}")
                
                # NOTE: Enabling gradient for entire embedding layer is MEMORY EXPENSIVE!
                # For large VLMs (128K vocab × 2048 dim = 256M params), this adds ~1-2GB VRAM.
                # With the meaningful initialization above, this is less critical.
                # The tune_top_llm_layers can learn to process the <|mem|> tokens appropriately.

        else:
            model = self.model_class(
                self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
            )

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        print("Model: ", model)

        return model

    def _get_statistics(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""

        # Always instantiate the mem processor explicitly so we get the chunk collator.
        # Using AutoProcessor.from_pretrained(checkpoint) can return the non-mem processor,
        # which then uses the single-step collator and crashes on chunk samples (dict/list keys).
        processor = self.processor_class(
            modality_configs=self.config.data.modality_configs,
            statistics=self._get_statistics(),
            embodiment_id_mapping=self._get_embodiment_id_mapping(),
            image_crop_size=self.model_config.image_crop_size,
            image_target_size=self.model_config.image_target_size,
            random_rotation_angle=self.model_config.random_rotation_angle,
            color_jitter_params=self.model_config.color_jitter_params,
            model_name=self.model_config.model_name,
            model_type=self.model_config.backbone_model_type,
            formalize_language=self.model_config.formalize_language,
            max_state_dim=self.model_config.max_state_dim,
            max_action_dim=self.model_config.max_action_dim,
            apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
            max_action_horizon=self.model_config.action_horizon,
            use_albumentations=self.model_config.use_albumentations_transforms,
            shortest_image_edge=self.model_config.shortest_image_edge,
            crop_fraction=self.model_config.crop_fraction,
            use_relative_action=self.config.model.use_relative_action,
            transformers_loading_kwargs=self.transformers_loading_kwargs,
            num_mem_tokens=self.model_config.num_mem_tokens,
            mem_token_str=self.model_config.mem_token_str,
        )

        print(
            colored(
                f"These are all the processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}",
                "yellow",
            )
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d6MemConfig, Gr00tN1d6MemPipeline)
