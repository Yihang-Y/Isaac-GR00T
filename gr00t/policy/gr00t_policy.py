"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy, PolicyWrapper


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        max_mem_history: int | None = None,
        use_noise_mem: bool = False,
        noise_mem_std: float = 0.1,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            max_mem_history: Maximum number of steps to keep in memory before resetting.
                If None, uses the model's mem_max_history_steps config value.
                When exceeded, memory state is reset to None (fresh start).
            use_noise_mem: If True, replace past memory state with noise during inference.
                This can be used for ablation studies or testing robustness.
            noise_mem_std: Standard deviation for noise when use_noise_mem=True.
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        
        # Memory state for mem models (will be initialized after model loading)
        self._mem_state = None
        self._is_mem_model = False
        self._mem_step_count = 0  # Track number of steps with memory
        self._max_mem_history = max_mem_history  # Will be set after model loading if None
        self._use_noise_mem = use_noise_mem
        self._noise_mem_std = noise_mem_std
        
        model_dir = Path(model_path)
        original_model_dir = model_dir  # Keep reference to original path

        # Check if this is a main output directory (contains checkpoint-* subdirectories)
        # or a specific checkpoint directory
        checkpoint_dirs = sorted(model_dir.glob("checkpoint-*"))
        if checkpoint_dirs:
            # Main output directory: use the latest checkpoint
            model_dir = checkpoint_dirs[-1]  # Latest checkpoint (sorted by name)
        
        # Load the processor: try checkpoint/processor/ first, then fall back to main output_dir/processor/
        processor_dir = model_dir / "processor"
        if not processor_dir.exists():
            # Try main output directory's processor
            processor_dir = original_model_dir / "processor"
            if not processor_dir.exists():
                raise FileNotFoundError(
                    f"Processor directory not found in either:\n"
                    f"  - {model_dir / 'processor'}\n"
                    f"  - {original_model_dir / 'processor'}\n"
                    f"Make sure the checkpoint or output directory contains a 'processor' subdirectory."
                )
        
        # Use absolute path as string (os.path.isdir needs string, not Path)
        processor_dir_str = str(processor_dir.resolve())
        
        # Try direct processor class loading first (avoids AutoProcessor's Hub validation)
        try:
            # Import processor classes
            from gr00t.model.gr00t_n1d6_mem.processing_gr00t_n1d6_mem import Gr00tN1d6MemProcessor
            from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
            
            # Try mem processor first, then fall back to regular processor
            try:
                self.processor: BaseProcessor = Gr00tN1d6MemProcessor.from_pretrained(
                    processor_dir_str,
                    trust_remote_code=True
                )
            except (FileNotFoundError, KeyError, OSError):
                self.processor: BaseProcessor = Gr00tN1d6Processor.from_pretrained(
                    processor_dir_str,
                    trust_remote_code=True
                )
        except (ImportError, FileNotFoundError, OSError) as e:
            # Fallback to AutoProcessor if direct loading fails
            self.processor: BaseProcessor = AutoProcessor.from_pretrained(
                processor_dir_str, 
                local_files_only=True,
                trust_remote_code=True
            )
        self.processor.eval()
        
        # Try to load model using standard HuggingFace from_pretrained first
        # This should work if checkpoint was saved correctly with updated config
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            model.eval()
            model.to(device=device)
            print(f"[Gr00tPolicy] Successfully loaded model using from_pretrained")
        except Exception as e:
            # Fallback to manual loading if from_pretrained fails
            # This handles cases where config vocab_size doesn't match weights
            print(f"[Gr00tPolicy] from_pretrained failed ({e}), using manual loading as fallback")
            
            # Get tokenizer vocab size
            tokenizer = self.processor.processor.tokenizer
            tokenizer_vocab_size = len(tokenizer)

            # Load weights first to get checkpoint's embedding size
            checkpoint_files = sorted(model_dir.glob("model*.safetensors"))
            if not checkpoint_files:
                checkpoint_files = sorted(model_dir.glob("pytorch_model*.safetensors"))
            if not checkpoint_files:
                checkpoint_files = sorted(model_dir.glob("pytorch_model*.bin"))
            
            if not checkpoint_files:
                raise FileNotFoundError(f"No model weights found in {model_dir}")
            
            # Load weights
            if checkpoint_files[0].suffix == ".safetensors":
                try:
                    from safetensors import safe_open
                except ImportError:
                    raise ImportError(
                        "safetensors library is required. Install with: pip install safetensors"
                    )
                state_dict = {}
                for shard_file in checkpoint_files:
                    with safe_open(shard_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
            else:
                if len(checkpoint_files) == 1:
                    state_dict = torch.load(checkpoint_files[0], map_location="cpu")
                else:
                    state_dict = {}
                    for shard_file in checkpoint_files:
                        state_dict.update(torch.load(shard_file, map_location="cpu"))
            
            # Get checkpoint vocab size
            embed_key = None
            for key in state_dict.keys():
                if "embed_tokens.weight" in key or "word_embeddings.weight" in key:
                    embed_key = key
                    break
            checkpoint_vocab_size = state_dict[embed_key].shape[0] if embed_key else None
            
            # Create model from config
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModel.from_config(config, trust_remote_code=True)
            
            # Resize if needed
            if checkpoint_vocab_size:
                model_vocab_size = None
                if hasattr(model, "backbone") and hasattr(model.backbone, "model"):
                    try:
                        embed_layer = model.backbone.model.language_model.model.embed_tokens
                        model_vocab_size = embed_layer.weight.shape[0]
                    except Exception:
                        pass
                
                if model_vocab_size and checkpoint_vocab_size != model_vocab_size:
                    print(f"[Gr00tPolicy] Resizing embeddings: {model_vocab_size} -> {checkpoint_vocab_size}")
                    if hasattr(model, "backbone") and hasattr(model.backbone, "model"):
                        try:
                            # Manual resize
                            embed_layer = model.backbone.model.language_model.model.embed_tokens
                            old_weight = embed_layer.weight.data
                            new_embedding = torch.nn.Embedding(checkpoint_vocab_size, old_weight.shape[1])
                            min_size = min(old_weight.shape[0], checkpoint_vocab_size)
                            new_embedding.weight.data[:min_size] = old_weight[:min_size]
                            model.backbone.model.language_model.model.embed_tokens = new_embedding
                            
                            # Resize lm_head
                            if hasattr(model.backbone.model.language_model, "lm_head"):
                                lm_head = model.backbone.model.language_model.lm_head
                                old_lm_weight = lm_head.weight.data
                                new_lm_head = torch.nn.Linear(old_lm_weight.shape[1], checkpoint_vocab_size, bias=lm_head.bias is not None)
                                min_out_size = min(old_lm_weight.shape[0], checkpoint_vocab_size)
                                new_lm_head.weight.data[:min_out_size] = old_lm_weight[:min_out_size]
                                if lm_head.bias is not None:
                                    new_lm_head.bias.data[:min_out_size] = lm_head.bias.data[:min_out_size]
                                model.backbone.model.language_model.lm_head = new_lm_head
                        except Exception as e:
                            print(f"[Gr00tPolicy WARNING] Manual resize failed: {e}")
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(device=device, dtype=torch.bfloat16)
        
        self.model = model

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        
        # For mem models, use single-step collator for inference (not chunk collator)
        # Chunk collator expects chunk_len which is only available during training
        if hasattr(self.processor, "collator") and hasattr(self.processor.collator, "_step_collator"):
            # This is a chunk collator (for mem models), use the underlying step collator for inference
            self.collate_fn = self.processor.collator._step_collator
            self._is_mem_model = True
            
            # Set max_mem_history from model config if not provided
            if self._max_mem_history is None:
                if hasattr(self.model, "config") and hasattr(self.model.config, "mem_max_history_steps"):
                    self._max_mem_history = self.model.config.mem_max_history_steps
                else:
                    # Default fallback
                    self._max_mem_history = 32
            
            # CRITICAL: Ensure _step_collator's tokenizer has mem token!
            # The _step_collator creates its own processor which may not have <|mem|> token.
            # We need to sync the tokenizer so mem tokens are correctly tokenized.
            if hasattr(self.processor, "mem_token_str"):
                mem_token_str = self.processor.mem_token_str
                step_tokenizer = self.collate_fn.processor.tokenizer
                if mem_token_str not in step_tokenizer.get_vocab():
                    step_tokenizer.add_special_tokens({"additional_special_tokens": [mem_token_str]})
                
                # Verify mem token embedding is properly initialized (not random/zeros)
                if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "model"):
                    try:
                        embed_layer = self.model.backbone.model.language_model.model.embed_tokens
                        mem_token_id = self.processor.mem_token_id
                        mem_emb = embed_layer.weight[mem_token_id].detach()
                        mem_norm = mem_emb.float().norm().item()
                        
                        # Check if embedding looks reasonable (not near-zero)
                        if mem_norm < 0.1:
                            print(f"[Gr00tPolicy WARNING] <|mem|> embedding norm is very small ({mem_norm:.4f})! "
                                  f"This may indicate the embedding was not properly loaded from checkpoint.")
                    except Exception:
                        pass  # Silently skip verification if it fails
        else:
            # Regular collator (for non-mem models)
            self.collate_fn = self.processor.collator
            self._is_mem_model = False

        # Extract and validate language configuration
        # Currently only supports single language input per timestep
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        assert len(language_keys) == 1, "Only one language key is supported"
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            # Check that the expected video key exists in the observation
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            # Check that the expected state key exists in the observation
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            # Check that the expected language key exists in the observation
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        
        # Extract inputs from BatchFeature/dict if needed
        # Note: _rec_to_dtype converts BatchFeature to dict, so we check for dict with "inputs" key
        if isinstance(collated_inputs, dict) and "inputs" in collated_inputs:
            collated_inputs = collated_inputs["inputs"]
        elif isinstance(collated_inputs, BatchFeature) and "inputs" in collated_inputs:
            collated_inputs = collated_inputs["inputs"]
        
        # Add mem_token_id for mem models if available
        if hasattr(self.processor, "mem_token_id") and self.processor.mem_token_id is not None:
            collated_inputs["mem_token_id"] = self.processor.mem_token_id

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            if self._is_mem_model:
                # Check if memory history exceeds limit - reset if needed
                if self._max_mem_history is not None and self._mem_step_count >= self._max_mem_history:
                    print(f"[Gr00tPolicy] Memory history limit reached ({self._mem_step_count} >= {self._max_mem_history}). Resetting memory state.")
                    self._mem_state = None
                    self._mem_step_count = 0
                
                # Replace mem_state with noise if use_noise_mem is enabled
                mem_state_to_use = self._mem_state
                if self._use_noise_mem and mem_state_to_use is not None:
                    # Generate noise with same shape as mem_state [B, T, N_mem, D]
                    noise = torch.randn_like(mem_state_to_use) * self._noise_mem_std
                    mem_state_to_use = noise
                
                # For mem models, pass inputs as dict and update memory state for iterative inference
                # Also pass use_noise_mem flag to replace mem_out with noise
                model_pred = self.model.get_action(
                    collated_inputs, 
                    mem_state=mem_state_to_use,
                    use_noise_mem_out=self._use_noise_mem,
                    noise_mem_out_std=self._noise_mem_std,
                )
                # Update memory state for next step (detach to avoid any computation graph issues)
                if "mem_state" in model_pred:
                    new_mem_state = model_pred["mem_state"]
                    # Detach and clone to ensure no shared memory with model internals
                    if torch.is_tensor(new_mem_state):
                        self._mem_state = new_mem_state.detach().clone()
                    else:
                        self._mem_state = new_mem_state
                    
                    # Increment step count if memory state is not None
                    if self._mem_state is not None:
                        self._mem_step_count += 1
            else:
                # For non-mem models, pass inputs as kwargs (original behavior)
                model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        For mem models, this clears the memory state so the next episode
        starts fresh without any historical context.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        # Reset memory state for mem models
        if self._is_mem_model:
            self._mem_state = None
            self._mem_step_count = 0
        return {}


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language: tuple[str] | list[str] = observation[language_key]  # (B,)

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple or list. Got {type(batched_language)}"
            )

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
