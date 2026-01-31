"""Custom Trainer for chunk-based memory training with burn-in and unroll.

This trainer implements the training protocol:
1. Burn-in phase: Fill memory cache without computing loss (no gradients)
2. Unroll phase: Predict actions and compute loss with gradients
3. Memory state is passed between steps within a chunk
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

import torch
from transformers.trainer import Trainer

from gr00t.experiment.trainer import Gr00tTrainer

logger = logging.getLogger(__name__)


class Gr00tChunkTrainer(Gr00tTrainer):
    """Trainer for chunk-based memory training with burn-in and unroll phases."""

    def __init__(
        self,
        *args: Any,
        log_memory_stats_every_n_steps: int = 100,
        log_memory_detail_every_n_steps: int = 500,
        log_attention_every_n_steps: int = 500,
        **kwargs: Any,
    ) -> None:
        """Initialize the chunk trainer.
        
        Args:
            log_memory_stats_every_n_steps: Log core memory stats (out_norm, loss_trend) every N steps.
            log_memory_detail_every_n_steps: Log detailed memory diagnostics (cosine_sim, etc.) every N steps.
            log_attention_every_n_steps: Log attention heatmaps every N steps. Set 0 to disable.
        """
        super().__init__(*args, **kwargs)
        self.log_memory_stats_every_n_steps = log_memory_stats_every_n_steps
        self.log_memory_detail_every_n_steps = log_memory_detail_every_n_steps
        self.log_attention_every_n_steps = log_attention_every_n_steps
        self._step_losses: list[float] = []  # Track per-step losses for logging

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Compute loss for chunk-based training with burn-in and unroll.

        Args:
            model: The model instance (should be Gr00tN1d6Mem)
            inputs: Dictionary containing batched chunk data:
                - input_ids_seq: [B, L, seq_len] or list of variable-length sequences
                - attention_mask_seq: [B, L, seq_len]
                - pixel_values_seq: [B, L, C, H, W] or similar
                - state_seq: [B, L, state_dim]
                - action_seq: [B, L, action_horizon, action_dim]
                - action_mask_seq: [B, L, action_horizon, action_dim]
                - embodiment_id: [B]
                - mem_token_id: int
                - burn_in_steps: int (optional, from config if not provided)
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for loss normalization)

        Returns:
            loss tensor (and optionally outputs)
        """
        def _nan_debug_tensor(name: str, x: Any, max_elems: int = 2048) -> None:
            """Log quick finite/NaN/Inf stats for a tensor-like object."""
            if x is None:
                logger.warning(f"[NaN DEBUG][trainer] {name}: None")
                return
            if isinstance(x, list):
                logger.warning(f"[NaN DEBUG][trainer] {name}: list(len)={len(x)}")
                # Optionally inspect first element if it's a tensor
                if len(x) > 0 and torch.is_tensor(x[0]):
                    _nan_debug_tensor(f"{name}[0]", x[0], max_elems=max_elems)
                return
            if not torch.is_tensor(x):
                logger.warning(f"[NaN DEBUG][trainer] {name}: type={type(x)} (non-tensor)")
                return
            try:
                xd = x.detach()
                flat = xd.reshape(-1)
                if flat.numel() > max_elems:
                    flat = flat[:max_elems]
                n = int(flat.numel())
                if flat.is_floating_point():
                    is_nan = torch.isnan(flat)
                    is_inf = torch.isinf(flat)
                    is_fin = torch.isfinite(flat)
                    n_nan = int(is_nan.sum().item())
                    n_inf = int(is_inf.sum().item())
                    n_fin = int(is_fin.sum().item())
                    msg = (
                        f"[NaN DEBUG][trainer] {name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
                        f"numel(sampled)={n}, finite={n_fin}, nan={n_nan}, inf={n_inf}"
                    )
                    if n_fin > 0:
                        fin = flat[is_fin].float()
                        msg += (
                            f", min={fin.min().item():.6g}, max={fin.max().item():.6g}, "
                            f"mean={fin.mean().item():.6g}, std={fin.std(unbiased=False).item():.6g}"
                        )
                    logger.warning(msg)
                else:
                    logger.warning(
                        f"[NaN DEBUG][trainer] {name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
                        f"numel(sampled)={n} (non-float tensor)"
                    )
            except Exception as e:
                logger.warning(f"[NaN DEBUG][trainer] {name}: failed to compute stats: {e}")
        # Strip `meta` field if present (not needed by the model, added by collator for debugging)
        if isinstance(inputs, Mapping) and "meta" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "meta"}

        # Extract chunk data
        # Inputs may come as {"inputs": {...}} or directly as dict
        if isinstance(inputs, Mapping) and "inputs" in inputs:
            batch_inputs = inputs["inputs"]
        else:
            batch_inputs = inputs

        # Ensure we have the required keys
        if "state_seq" not in batch_inputs:
            # Fallback: try to use standard forward (for compatibility)
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        B = batch_inputs["state_seq"].shape[0]
        L = batch_inputs["state_seq"].shape[1]

        # DEBUG: Check if different timesteps have different data
        if self.state.global_step < 5:
            self._debug_check_chunk_data(batch_inputs, B, L)
            
            # Also check for NaN in input data
            if "action_seq" in batch_inputs and batch_inputs["action_seq"] is not None:
                action_seq = batch_inputs["action_seq"]
                if torch.isnan(action_seq).any():
                    logger.warning(f"[NaN DEBUG] action_seq has NaN: {torch.isnan(action_seq).sum().item()} / {action_seq.numel()}")
            if "state_seq" in batch_inputs and batch_inputs["state_seq"] is not None:
                state_seq = batch_inputs["state_seq"]
                if torch.isnan(state_seq).any():
                    logger.warning(f"[NaN DEBUG] state_seq has NaN: {torch.isnan(state_seq).sum().item()} / {state_seq.numel()}")

        # Get burn-in and unroll steps from config or inputs
        burn_in_steps = batch_inputs.get("burn_in_steps", None)
        if burn_in_steps is None:
            # Try to get from model config (handle DDP/DeepSpeed wrapper)
            base_model = model.module if hasattr(model, "module") else model
            if hasattr(base_model, "config") and hasattr(base_model.config, "burn_in_steps"):
                burn_in_steps = base_model.config.burn_in_steps
            else:
                # Default: use half of chunk length
                burn_in_steps = L // 2

        unroll_steps = L - burn_in_steps

        assert burn_in_steps > 0, "burn_in_steps must be > 0"
        assert unroll_steps > 0, "unroll_steps must be > 0"
        assert burn_in_steps + unroll_steps == L, (
            f"burn_in_steps ({burn_in_steps}) + unroll_steps ({unroll_steps}) != chunk_len ({L})"
        )

        # Initialize memory state (None for first step)
        mem_state = None

        # ===== BURN-IN PHASE =====
        # Fill memory cache without computing loss (no gradients)
        for t in range(burn_in_steps):
            # Prepare step inputs
            step_inputs = self._prepare_step_inputs(batch_inputs, t, model.device)

            # Forward through model (no loss computation)
            with torch.no_grad():
                step_output = model.forward_step(
                    step_inputs,
                    mem_state=mem_state,
                    compute_loss=False,
                )

            # Update memory state (detached, no gradients)
            mem_state = step_output["mem_state"].detach()
            
            # DEBUG: Check for NaN in burn-in
            if self.state.global_step < 5 and torch.isnan(mem_state).any():
                logger.warning(f"[NaN DEBUG] mem_state has NaN after burn-in t={t}!")

        # ===== UNROLL PHASE =====
        # Predict actions and compute loss with gradients
        loss_sum = None
        loss_count = 0
        all_outputs = [] if return_outputs else None
        step_losses: list[float] = []  # Track per-step losses
        last_attention_weights = None  # For attention visualization

        # Check if we should capture attention this step
        should_log_attention = (
            self.log_attention_every_n_steps > 0
            and self.state.global_step % self.log_attention_every_n_steps == 0
        )

        for t in range(burn_in_steps, L):
            # Prepare step inputs
            step_inputs = self._prepare_step_inputs(batch_inputs, t, model.device)

            # Request attention on the last unroll step if visualization is enabled
            is_last_step = (t == L - 1)
            request_attention = should_log_attention and is_last_step

            # Forward through model (with loss computation)
            step_output = model.forward_step(
                step_inputs,
                mem_state=mem_state,
                compute_loss=True,
                return_attention=request_attention,
            )

            # Capture attention weights for visualization
            if request_attention and "mem_attention" in step_output:
                last_attention_weights = step_output["mem_attention"]

            # Accumulate loss
            if "loss" in step_output:
                step_loss = step_output["loss"]
                
                # DEBUG: Check for NaN in loss
                if torch.isnan(step_loss).any():
                    if self.state.global_step < 10:
                        logger.warning(
                            f"[NaN DEBUG] step_loss is NaN at t={t}! "
                            f"Checking inputs..."
                        )
                        # Inspect step inputs (data pipeline)
                        _nan_debug_tensor("step_inputs.state", step_inputs.get("state"))
                        _nan_debug_tensor("step_inputs.action", step_inputs.get("action"))
                        _nan_debug_tensor("step_inputs.action_mask", step_inputs.get("action_mask"))
                        _nan_debug_tensor("step_inputs.input_ids", step_inputs.get("input_ids"))
                        _nan_debug_tensor("step_inputs.attention_mask", step_inputs.get("attention_mask"))
                        _nan_debug_tensor("step_inputs.pixel_values", step_inputs.get("pixel_values"))
                        _nan_debug_tensor("step_inputs.embodiment_id", step_inputs.get("embodiment_id"))
                        # Check what's NaN
                        if "mem_out" in step_output and step_output["mem_out"] is not None:
                            mo = step_output["mem_out"]
                            if torch.isnan(mo).any():
                                logger.warning(f"  - mem_out has NaN: {torch.isnan(mo).sum().item()} / {mo.numel()}")
                        if "mem_state" in step_output and step_output["mem_state"] is not None:
                            ms = step_output["mem_state"]
                            if torch.isnan(ms).any():
                                logger.warning(f"  - mem_state has NaN: {torch.isnan(ms).sum().item()} / {ms.numel()}")
                
                loss_sum = step_loss if loss_sum is None else (loss_sum + step_loss)
                loss_count += 1
                step_losses.append(step_loss.detach().item())

            # Update memory state (with gradients for unroll phase)
            mem_state = step_output["mem_state"]

            if return_outputs:
                all_outputs.append(step_output)

        # Store step losses for logging
        self._step_losses = step_losses

        # Aggregate losses
        if loss_count > 0 and loss_sum is not None:
            loss = loss_sum / loss_count
        else:
            # Fallback: create a zero loss with proper computation graph
            # Using a model parameter ensures the loss is connected to the computation graph
            # and won't cause issues with gradient computation.
            base_model = model.module if hasattr(model, "module") else model
            dummy_param = next(iter(base_model.parameters()))
            loss = (dummy_param * 0.0).sum()
            logger.warning(
                f"No valid losses computed in chunk (loss_count=0). "
                f"burn_in={burn_in_steps}, unroll={unroll_steps}, L={L}. "
                f"Returning zero loss with gradient connection."
            )

        # Log memory and training statistics to wandb
        should_log_detailed = (
            self.state.global_step % self.log_memory_stats_every_n_steps == 0
            or self.state.global_step < 10
        )
        
        # Get last step output for mem_out statistics
        last_step_output = all_outputs[-1] if all_outputs else step_output
        mem_out = last_step_output.get("mem_out", None) if last_step_output else None
        mem_tokens = last_step_output.get("mem_tokens", None) if last_step_output else None
        
        if should_log_detailed and mem_state is not None:
            self._log_memory_stats(
                mem_state, mem_out, mem_tokens, step_losses, burn_in_steps, unroll_steps
            )
        
        # Log attention visualization (less frequently due to overhead)
        if should_log_attention and last_attention_weights is not None and mem_state is not None:
            # Get model config for num_mem_tokens
            base_model = model.module if hasattr(model, "module") else model
            num_mem_tokens = getattr(base_model.config, "num_mem_tokens", 8)
            history_length = mem_state.shape[1] if mem_state is not None else 1
            
            self._log_attention_visualization(
                last_attention_weights,
                num_mem_tokens=num_mem_tokens,
                history_length=history_length,
            )

        # Prepare outputs
        outputs = {
            "loss": loss,
            # Avoid storing per-step tensors unless explicitly requested (can be memory heavy).
            "losses": None,
            "mem_state": mem_state,
            "step_outputs": all_outputs,
        }

        if return_outputs:
            return loss, outputs
        else:
            return loss

    def _log_memory_stats(
        self,
        mem_state: torch.Tensor,
        mem_out: Optional[torch.Tensor],
        mem_tokens: Optional[torch.Tensor],
        step_losses: list[float],
        burn_in_steps: int,
        unroll_steps: int,
    ) -> None:
        """
        Log memory-related statistics to wandb/tensorboard.
        
        Split into CORE (always) and DETAIL (every 500 steps) to reduce log spam.
        """
        import torch.distributed as dist
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        logs = {}
        log_detailed = (
            self.log_memory_detail_every_n_steps > 0
            and (
                (self.state.global_step % self.log_memory_detail_every_n_steps == 0)
                or (self.state.global_step < 10)
            )
        )

        # ==========================================
        # CORE METRICS (always logged, compact)
        # ==========================================
        
        # Memory output norm - what affects action prediction
        if mem_out is not None and torch.is_tensor(mem_out):
            with torch.no_grad():
                logs["mem/out_norm"] = mem_out.float().norm(dim=-1).mean().item()

        # Loss trend - key indicator (negative = good)
        if step_losses and len(step_losses) > 0:
            logs["mem/loss_trend"] = step_losses[-1] - step_losses[0]

        # ==========================================
        # DETAILED METRICS (every 500 steps)
        # ==========================================
        if log_detailed:
            if mem_state is not None and torch.is_tensor(mem_state):
                with torch.no_grad():
                    mem_flat = mem_state.float()
                    logs["mem_d/state_std"] = mem_flat.std().item()
                    logs["mem_d/state_norm"] = mem_flat.norm(dim=-1).mean().item()
                    logs["mem_d/history_len"] = mem_state.shape[1]
                    
                    # Cosine sim between oldest/newest (1.0 = bug, <1.0 = ok)
                    if mem_flat.shape[1] > 1:
                        oldest = mem_flat[:, 0].reshape(-1, mem_flat.shape[-1])
                        newest = mem_flat[:, -1].reshape(-1, mem_flat.shape[-1])
                        cos_sim = torch.nn.functional.cosine_similarity(oldest, newest, dim=-1).mean()
                        logs["mem_d/cosine_oldest_newest"] = cos_sim.item()

            if mem_out is not None and torch.is_tensor(mem_out):
                with torch.no_grad():
                    logs["mem_d/out_std"] = mem_out.float().std().item()

            if mem_tokens is not None and torch.is_tensor(mem_tokens):
                with torch.no_grad():
                    logs["mem_d/tokens_norm"] = mem_tokens.float().norm(dim=-1).mean().item()

            if step_losses and len(step_losses) > 0:
                logs["mem_d/loss_first"] = step_losses[0]
                logs["mem_d/loss_last"] = step_losses[-1]

        self.log(logs)

    def _debug_check_chunk_data(self, batch_inputs: dict, B: int, L: int) -> None:
        """
        Debug: Check if different timesteps in a chunk have different data.
        If all timesteps have identical data, that's a bug!
        """
        import torch.distributed as dist
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        with torch.no_grad():
            # Check state_seq
            if "state_seq" in batch_inputs:
                state_seq = batch_inputs["state_seq"]  # [B, L, state_dim]
                if state_seq.shape[1] > 1:
                    state_t0 = state_seq[:, 0]  # [B, state_dim]
                    state_t1 = state_seq[:, 1]  # [B, state_dim]
                    state_same = torch.allclose(state_t0, state_t1, atol=1e-6)
                    logger.info(f"[DEBUG] state_seq t0 vs t1 identical: {state_same}")
                    if state_same:
                        logger.warning("[BUG?] All timesteps have IDENTICAL state data!")

            # Check input_ids_seq
            if "input_ids_seq" in batch_inputs and batch_inputs["input_ids_seq"] is not None:
                ids_seq = batch_inputs["input_ids_seq"]  # [B, L, seq_len]
                if ids_seq.shape[1] > 1:
                    ids_t0 = ids_seq[:, 0]  # [B, seq_len]
                    ids_t1 = ids_seq[:, 1]  # [B, seq_len]
                    ids_same = torch.equal(ids_t0, ids_t1)
                    logger.info(f"[DEBUG] input_ids_seq t0 vs t1 identical: {ids_same}")
                    if ids_same:
                        logger.warning("[BUG?] All timesteps have IDENTICAL input_ids!")

            # Check pixel_values_seq
            if "pixel_values_seq" in batch_inputs and batch_inputs["pixel_values_seq"] is not None:
                pv_seq = batch_inputs["pixel_values_seq"]
                logger.info(f"[DEBUG] pixel_values_seq type: {type(pv_seq)}")
                
                if isinstance(pv_seq, torch.Tensor) and pv_seq.ndim >= 2 and pv_seq.shape[1] > 1:
                    pv_t0 = pv_seq[:, 0]  # [B, ...]
                    pv_t1 = pv_seq[:, 1]  # [B, ...]
                    pv_same = torch.allclose(pv_t0.float(), pv_t1.float(), atol=1e-6)
                    logger.info(f"[DEBUG] pixel_values_seq (tensor) t0 vs t1 identical: {pv_same}")
                    if pv_same:
                        logger.warning("[BUG?] All timesteps have IDENTICAL pixel_values!")
                        
                elif isinstance(pv_seq, list) and len(pv_seq) > 0:
                    # List format: [B][L][n_images] - each element is a list of image tensors
                    first_batch = pv_seq[0]  # [L] timesteps for batch 0
                    logger.info(f"[DEBUG] pixel_values_seq[0] length: {len(first_batch)}")
                    
                    if len(first_batch) > 1:
                        pv_t0 = first_batch[0]  # First timestep's images (list of tensors)
                        pv_t1 = first_batch[1]  # Second timestep's images
                        
                        # Each timestep is a list of image tensors
                        if isinstance(pv_t0, list) and isinstance(pv_t1, list):
                            logger.info(f"[DEBUG] pv_t0 has {len(pv_t0)} images, pv_t1 has {len(pv_t1)} images")
                            if len(pv_t0) > 0 and len(pv_t1) > 0:
                                # Compare first image of each timestep
                                img0 = pv_t0[0] if torch.is_tensor(pv_t0[0]) else torch.tensor(pv_t0[0])
                                img1 = pv_t1[0] if torch.is_tensor(pv_t1[0]) else torch.tensor(pv_t1[0])
                                pv_same = torch.allclose(img0.float(), img1.float(), atol=1e-6)
                                logger.info(f"[DEBUG] pixel_values_seq (list) t0 vs t1 first image identical: {pv_same}")
                                if pv_same:
                                    logger.warning("[BUG?] All timesteps have IDENTICAL pixel_values!")
                        elif torch.is_tensor(pv_t0) and torch.is_tensor(pv_t1):
                            # Direct tensor format
                            pv_same = torch.allclose(pv_t0.float(), pv_t1.float(), atol=1e-6)
                            logger.info(f"[DEBUG] pixel_values_seq (tensor items) t0 vs t1 identical: {pv_same}")
                            if pv_same:
                                logger.warning("[BUG?] All timesteps have IDENTICAL pixel_values!")
            else:
                logger.info("[DEBUG] pixel_values_seq NOT in batch_inputs or is None")

    def _log_attention_visualization(
        self,
        attention_weights: list[torch.Tensor],
        num_mem_tokens: int,
        history_length: int,
    ) -> None:
        """
        Log memory attention visualization to wandb.

        This creates heatmap images showing how memory tokens attend to each other
        across time steps. Useful for understanding:
        1. Does the model attend to recent vs. distant history?
        2. Are there specific memory tokens that are more important?
        3. Is attention sparse or dense?

        Args:
            attention_weights: List of attention tensors [B, H, T*N, T*N] from each layer
                              where T = history timesteps, N = num_mem_tokens
            num_mem_tokens: Number of memory tokens per timestep
            history_length: Number of timesteps in memory history
        """
        import torch.distributed as dist
        
        # Only log from rank 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        try:
            import wandb
            import numpy as np
            
            if wandb.run is None:
                return  # wandb not initialized
                
        except ImportError:
            logger.debug("wandb not available, skipping attention visualization")
            return

        if not attention_weights or len(attention_weights) == 0:
            return

        # Take the last layer's attention (usually most meaningful)
        # Shape: [B, H, T*N, T*N]
        last_layer_attn = attention_weights[-1]
        
        # Average over batch and heads to get [T*N, T*N]
        # Convert to float32 first since numpy doesn't support bfloat16
        attn_avg = last_layer_attn.float().mean(dim=(0, 1)).detach().cpu().numpy()
        
        # Create temporal attention summary:
        # Reshape to [T, N, T, N] and average over token dimension -> [T, T]
        T = history_length
        N = num_mem_tokens
        seq_len = attn_avg.shape[0]
        
        if seq_len == T * N:
            try:
                attn_reshaped = attn_avg.reshape(T, N, T, N)
                # Average over token dimensions: [T, T] (query timestep -> key timestep)
                temporal_attn = attn_reshaped.mean(axis=(1, 3))
                
                # Log temporal attention heatmap
                wandb.log({
                    "attention/temporal_heatmap": wandb.Image(
                        self._create_attention_heatmap(
                            temporal_attn,
                            title="Memory Temporal Attention",
                            xlabel="Key Timestep (history)",
                            ylabel="Query Timestep",
                        )
                    )
                })
                
                # Log attention statistics
                # In causal attention, only the last query timestep can see the most recent key timestep.
                # So we measure: (1) how much the last query attends to the most recent key,
                # and (2) diagonal attention (each query attending to itself).
                recent_focus = float(temporal_attn[-1, -1])  # Last query -> last key (only valid in causal)
                diagonal_attention = np.diag(temporal_attn)  # Each query -> corresponding key
                self.log({
                    "attention/recent_focus": recent_focus,  # Last query's attention to most recent key
                    "attention/diagonal_mean": float(diagonal_attention.mean()),  # Self-attention strength
                    "attention/diagonal_std": float(diagonal_attention.std()),
                })
                
            except Exception as e:
                logger.debug(f"Could not create temporal attention visualization: {e}")
        
        # Also log the full attention pattern (downsampled if too large)
        try:
            if attn_avg.shape[0] > 64:
                # Downsample for visualization
                step = attn_avg.shape[0] // 64
                attn_downsampled = attn_avg[::step, ::step]
            else:
                attn_downsampled = attn_avg
                
            wandb.log({
                "attention/full_heatmap": wandb.Image(
                    self._create_attention_heatmap(
                        attn_downsampled,
                        title="Memory Self-Attention (Last Layer)",
                        xlabel="Key",
                        ylabel="Query",
                    )
                )
            })
        except Exception as e:
            logger.debug(f"Could not create full attention visualization: {e}")

    def _create_attention_heatmap(
        self,
        attn_matrix,  # np.ndarray
        title: str = "Attention",
        xlabel: str = "Key",
        ylabel: str = "Query",
    ):
        """
        Create a matplotlib figure for attention heatmap.
        
        Args:
            attn_matrix: 2D numpy array of attention weights
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use a perceptually uniform colormap
        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=10)
        
        # Add diagonal line to show causal mask boundary
        min_dim = min(attn_matrix.shape)
        ax.plot([0, min_dim-1], [0, min_dim-1], 'r--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        return fig

    def _prepare_step_inputs(
        self, batch_inputs: dict, step_idx: int, device: torch.device
    ) -> dict:
        """
        Prepare inputs for a single step within a chunk.

        Args:
            batch_inputs: Batched chunk inputs
            step_idx: Index of the step within the chunk (0 to L-1)
            device: Device to move tensors to

        Returns:
            Dictionary of step inputs ready for model.forward_step
        """
        step_inputs = {}

        # Extract step t from sequences
        if "input_ids_seq" in batch_inputs:
            # Handle variable-length sequences
            input_ids_seq = batch_inputs["input_ids_seq"]
            if isinstance(input_ids_seq, list):
                # List of variable-length sequences per batch
                step_inputs["input_ids"] = torch.stack([seq[step_idx] for seq in input_ids_seq]).to(device)
            else:
                # Tensor: [B, L, seq_len]
                step_inputs["input_ids"] = input_ids_seq[:, step_idx].to(device)

        if "attention_mask_seq" in batch_inputs:
            attention_mask_seq = batch_inputs["attention_mask_seq"]
            if isinstance(attention_mask_seq, list):
                step_inputs["attention_mask"] = torch.stack([mask[step_idx] for mask in attention_mask_seq]).to(device)
            else:
                step_inputs["attention_mask"] = attention_mask_seq[:, step_idx].to(device)

        if "pixel_values_seq" in batch_inputs:
            pixel_values_seq = batch_inputs["pixel_values_seq"]
            if isinstance(pixel_values_seq, list):
                # `pixel_values_seq` may be:
                # - list[B] of list[L] of Tensor (single image per step), OR
                # - list[B] of list[L] of list[V] of Tensor (multiple images/views per step).
                pv_step = [pv[step_idx] for pv in pixel_values_seq]  # len B
                if len(pv_step) == 0:
                    step_inputs["pixel_values"] = None
                elif torch.is_tensor(pv_step[0]):
                    step_inputs["pixel_values"] = torch.stack(pv_step).to(device)
                else:
                    # Assume list[V] per sample; flatten to list[B*V] preserving sample order.
                    flat: list[torch.Tensor] = []
                    for item in pv_step:
                        flat.extend(item)
                    step_inputs["pixel_values"] = [t.to(device) for t in flat]
            else:
                # Handle different pixel_values shapes
                if pixel_values_seq.ndim == 5:  # [B, L, C, H, W]
                    step_inputs["pixel_values"] = pixel_values_seq[:, step_idx].to(device)
                else:
                    step_inputs["pixel_values"] = pixel_values_seq[:, step_idx].to(device)

        # Extract state, action, action_mask
        step_inputs["state"] = batch_inputs["state_seq"][:, step_idx].to(device)

        if batch_inputs.get("action_seq") is not None:
            step_inputs["action"] = batch_inputs["action_seq"][:, step_idx].to(device)
            step_inputs["action_mask"] = batch_inputs["action_mask_seq"][:, step_idx].to(device)
        else:
            # This should not happen during training - log a warning once per chunk
            if step_idx == 0:
                logger.warning(
                    f"action_seq is None in batch_inputs! "
                    f"Keys: {list(batch_inputs.keys())}. "
                    f"This may indicate a data pipeline issue."
                )
            step_inputs["action"] = None
            step_inputs["action_mask"] = None

        # Extract embodiment_id and mem_token_id
        if "embodiment_id" in batch_inputs and batch_inputs["embodiment_id"] is not None:
            step_inputs["embodiment_id"] = batch_inputs["embodiment_id"].to(device)
        else:
            step_inputs["embodiment_id"] = None

        if "mem_token_id" in batch_inputs:
            step_inputs["mem_token_id"] = batch_inputs["mem_token_id"]

        return step_inputs
