from typing import Optional, Tuple

import logging
import os

from gr00t.configs.model.gr00t_n1d6_mem import Gr00tN1d6MemConfig

logger = logging.getLogger(__name__)
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.memory_module import MemoryModule
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class Gr00tN1d6MemActionHead(nn.Module):
    """Action head component for flow matching diffusion policy with memory support."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6MemConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            # Keep `mask_token` in FP32 for stability (BF16 optimizer/ZeRO can corrupt tiny params).
            # We'll cast to the runtime dtype in `forward()` when applying it.
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim, dtype=torch.float32))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """Set frozen modules to eval mode during training."""
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    @staticmethod
    def _nan_debug_tensor(name: str, x: Optional[torch.Tensor], max_elems: int = 2048) -> None:
        """Log quick finite/NaN/Inf stats for a tensor. Designed for rare NaN/Inf debugging."""
        if x is None:
            logger.warning(f"[NaN DEBUG][action_head] {name}: None")
            return
        if not torch.is_tensor(x):
            logger.warning(f"[NaN DEBUG][action_head] {name}: type={type(x)} (non-tensor)")
            return
        try:
            xd = x.detach()
            # Sample to keep stats cheap
            flat = xd.reshape(-1)
            if flat.numel() > max_elems:
                flat = flat[:max_elems]

            # Compute finite / NaN / Inf counts
            is_nan = torch.isnan(flat) if flat.is_floating_point() else torch.zeros_like(flat, dtype=torch.bool)
            is_inf = torch.isinf(flat) if flat.is_floating_point() else torch.zeros_like(flat, dtype=torch.bool)
            is_finite = torch.isfinite(flat) if flat.is_floating_point() else torch.ones_like(flat, dtype=torch.bool)
            n = int(flat.numel())
            n_nan = int(is_nan.sum().item())
            n_inf = int(is_inf.sum().item())
            n_fin = int(is_finite.sum().item())

            msg = (
                f"[NaN DEBUG][action_head] {name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
                f"numel(sampled)={n}, finite={n_fin}, nan={n_nan}, inf={n_inf}"
            )

            # Add basic stats for finite floats
            if flat.is_floating_point() and n_fin > 0:
                fin = flat[is_finite].float()
                msg += (
                    f", min={fin.min().item():.6g}, max={fin.max().item():.6g}, "
                    f"mean={fin.mean().item():.6g}, std={fin.std(unbiased=False).item():.6g}"
                )
            logger.warning(msg)
        except Exception as e:
            logger.warning(f"[NaN DEBUG][action_head] {name}: failed to compute stats: {e}")

    @staticmethod
    def _nan_debug_param(name: str, p: Optional[torch.Tensor], max_elems: int = 2048) -> None:
        """Alias for parameters (same as tensor stats but different prefix for readability)."""
        Gr00tN1d6MemActionHead._nan_debug_tensor(f"PARAM {name}", p, max_elems=max_elems)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        memory_features: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        """
        Forward pass through the action head with optional memory features.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
                - mem_mask: [B, seq_len] (optional, indicates mem token positions)
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]
            memory_features: Optional memory features [B, K, D] to concatenate with backbone_features

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # NOTE: mem token extraction is handled in `Gr00tN1d6Mem.extract_mem_tokens()`.
        # The action head only consumes `memory_features` (read features) if provided.
        mem_tokens = None

        # Concatenate memory features with backbone features if provided
        if memory_features is not None:
            # memory_features: [B, K, D]
            # vl_embeds: [B, S, D]
            # Concatenate along sequence dimension
            vl_embeds = torch.cat([memory_features, vl_embeds], dim=1)  # [B, K+S, D]
            # Update attention mask
            K = memory_features.shape[1]
            B, S = backbone_output.backbone_attention_mask.shape
            mem_attn_mask = torch.ones(B, K, device=device, dtype=torch.bool)
            backbone_output.backbone_attention_mask = torch.cat(
                [mem_attn_mask, backbone_output.backbone_attention_mask], dim=1
            )
            # Update image_mask (memory tokens are not image tokens)
            if "image_mask" in backbone_output:
                mem_image_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
                backbone_output.image_mask = torch.cat(
                    [mem_image_mask, backbone_output.image_mask], dim=1
                )

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # ---- Embed state (stage-by-stage NaN/Inf diagnostics) ----
        state_in = action_input.state
        state_features_raw = self.state_encoder(state_in, embodiment_id)
        state_features = state_features_raw

        # Dropout state features.
        do_dropout = None
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            mask_tok = self.mask_token.to(dtype=state_features.dtype, device=state_features.device)
            state_features = state_features * (1 - do_dropout) + mask_tok * do_dropout

        # Add Gaussian noise to state features.
        noise_sf = None
        if self.training and self.state_additive_noise_scale > 0:
            noise_sf = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise_sf

        # If state branch becomes non-finite, dump encoder params and ids once (very high-signal).
        if state_features.is_floating_point() and (not torch.isfinite(state_features).all()):
            counter = getattr(self, "_nan_state_debug_counter", 0)
            if counter < 5:
                setattr(self, "_nan_state_debug_counter", counter + 1)
                logger.warning(
                    f"[NaN DEBUG][action_head] Non-finite state_features detected "
                    f"(count={counter+1}/5). Investigating state_encoder..."
                )
                logger.warning(
                    f"[NaN DEBUG][action_head] state_dropout_prob={self.state_dropout_prob}, "
                    f"state_additive_noise_scale={self.state_additive_noise_scale}, training={self.training}"
                )
                # Input + stage tensors
                self._nan_debug_tensor("state_in(action_input.state)", state_in)
                self._nan_debug_tensor("state_features_raw(state_encoder out)", state_features_raw)
                self._nan_debug_tensor("state_features(after dropout/noise)", state_features)
                self._nan_debug_tensor("embodiment_id", embodiment_id)
                if do_dropout is not None:
                    self._nan_debug_tensor("do_dropout(mask)", do_dropout)
                    try:
                        # mean is fraction dropped (since mask is 0/1)
                        logger.warning(
                            f"[NaN DEBUG][action_head] do_dropout.mean={do_dropout.detach().float().mean().item():.6g}"
                        )
                    except Exception as e:
                        logger.warning(f"[NaN DEBUG][action_head] failed to compute do_dropout.mean: {e}")
                if self.mask_token is not None:
                    self._nan_debug_param("mask_token", self.mask_token)
                if noise_sf is not None:
                    self._nan_debug_tensor("noise_sf(state additive noise)", noise_sf)

                # Show which categories are present in this batch
                try:
                    uniq = embodiment_id.detach().to("cpu").unique().tolist() if embodiment_id is not None else []
                    logger.warning(f"[NaN DEBUG][action_head] embodiment_id unique: {uniq}")
                except Exception as e:
                    logger.warning(f"[NaN DEBUG][action_head] failed to compute embodiment_id unique: {e}")

                # State encoder parameter stats (helps confirm weight explosion / NaN weights)
                try:
                    # CategorySpecificMLP -> layer1/layer2 are CategorySpecificLinear with params W,b
                    if hasattr(self.state_encoder, "layer1") and hasattr(self.state_encoder.layer1, "W"):
                        self._nan_debug_param("state_encoder.layer1.W", self.state_encoder.layer1.W)
                        self._nan_debug_param("state_encoder.layer1.b", self.state_encoder.layer1.b)
                    if hasattr(self.state_encoder, "layer2") and hasattr(self.state_encoder.layer2, "W"):
                        self._nan_debug_param("state_encoder.layer2.W", self.state_encoder.layer2.W)
                        self._nan_debug_param("state_encoder.layer2.b", self.state_encoder.layer2.b)

                    # Also dump the selected weights for the first element's category (often enough)
                    if embodiment_id is not None and torch.is_tensor(embodiment_id) and embodiment_id.numel() > 0:
                        cid0 = int(embodiment_id.detach().flatten()[0].to("cpu").item())
                        if hasattr(self.state_encoder, "layer1") and hasattr(self.state_encoder.layer1, "W"):
                            self._nan_debug_param(
                                f"state_encoder.layer1.W[cid={cid0}]",
                                self.state_encoder.layer1.W[cid0],
                            )
                        if hasattr(self.state_encoder, "layer2") and hasattr(self.state_encoder.layer2, "W"):
                            self._nan_debug_param(
                                f"state_encoder.layer2.W[cid={cid0}]",
                                self.state_encoder.layer2.W[cid0],
                            )
                except Exception as e:
                    logger.warning(f"[NaN DEBUG][action_head] failed to dump state_encoder params: {e}")
        # ---- end state branch diagnostics ----

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        denom = action_mask.sum()
        # If the mask is all zeros, the loss will be exactly 0 and training will silently do nothing.
        # Fail fast with a clear error so users can fix their dataset / modality config.
        if self.training and float(denom.detach().float().cpu().item()) == 0.0:
            raise ValueError(
                "action_mask.sum() == 0 during training, so the action loss is forced to 0.\n"
                "This almost always means your processed actions have zero valid dimensions/horizon "
                "(e.g., action_dim==0 after processing, or everything is padded/masked).\n"
                f"- action shape: {tuple(actions.shape) if actions is not None else None}\n"
                f"- action_mask shape: {tuple(action_mask.shape) if action_mask is not None else None}\n"
                "Debug tips:\n"
                "- Inspect one batch right after the processor/collator: print `action.shape`, `action_mask.sum()`, "
                "`action_mask.nonzero().shape[0]`.\n"
                "- Check your dataset action fields are non-empty arrays and that modality_configs action keys/delta_indices match."
            )
        loss = action_loss.sum() / (denom + 1e-6)

        # ===== NaN/Inf DEBUG (only triggers when loss is non-finite) =====
        if loss.is_floating_point() and (not torch.isfinite(loss).all()):
            # Limit spam: only log the first few occurrences per module instance.
            counter = getattr(self, "_nan_debug_counter", 0)
            if counter < 5:
                setattr(self, "_nan_debug_counter", counter + 1)
                logger.warning(
                    f"[NaN DEBUG][action_head] Non-finite loss detected. "
                    f"count={counter+1}/5, loss={loss.detach().float().cpu().item() if loss.numel()==1 else 'tensor'}"
                )
                self._nan_debug_tensor("memory_features", memory_features)
                self._nan_debug_tensor("vl_embeds(backbone_features)", vl_embeds)
                self._nan_debug_tensor("state", action_input.state)
                self._nan_debug_tensor("state_features", state_features)
                self._nan_debug_tensor("actions(action_input.action)", actions)
                self._nan_debug_tensor("action_mask", action_mask)
                self._nan_debug_tensor("t(sampled)", t)
                self._nan_debug_tensor("noise(randn)", noise)
                self._nan_debug_tensor("noisy_trajectory", noisy_trajectory)
                self._nan_debug_tensor("velocity(actions-noise)", velocity)
                self._nan_debug_tensor("action_features", action_features)
                self._nan_debug_tensor("sa_embs(cat state+action)", sa_embs)
                self._nan_debug_tensor("model_output(DiT)", model_output)
                self._nan_debug_tensor("pred(action_decoder)", pred)
                self._nan_debug_tensor("pred_actions", pred_actions)
                self._nan_debug_tensor("action_loss(mse*mask)", action_loss)
                self._nan_debug_tensor("denom(action_mask.sum)", denom)
        # ===== END NaN/Inf DEBUG =====

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            "mem_tokens": mem_tokens,
        }

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        memory_features: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        """Generate actions using the flow matching diffusion process with memory."""
        vl_embeds = backbone_features

        # Concatenate memory features if provided
        if memory_features is not None:
            vl_embeds = torch.cat([memory_features, vl_embeds], dim=1)
            K = memory_features.shape[1]
            B, S = backbone_output.backbone_attention_mask.shape
            device = vl_embeds.device
            mem_attn_mask = torch.ones(B, K, device=device, dtype=torch.bool)
            backbone_output.backbone_attention_mask = torch.cat(
                [mem_attn_mask, backbone_output.backbone_attention_mask], dim=1
            )
            if "image_mask" in backbone_output:
                mem_image_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
                backbone_output.image_mask = torch.cat(
                    [mem_image_mask, backbone_output.image_mask], dim=1
                )

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """Encode features for the action head."""
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)
        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6MemConfig):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6Mem(PreTrainedModel):
    """Gr00tN1d6Mem: Vision-Language-Action model with backbone and memory module."""

    config_class = Gr00tN1d6MemConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6MemConfig,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6Mem model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize memory module
        self.memory_module = MemoryModule(
            mem_token_dim=config.backbone_embedding_dim,
            num_mem_tokens=config.num_mem_tokens,
            mem_out_tokens=config.mem_out_tokens,
            max_history_steps=config.mem_max_history_steps,
            num_layers=config.mem_num_layers,
            num_heads=config.mem_num_heads,
            head_dim=config.mem_head_dim,
            dropout=config.mem_dropout,
            activation_fn=config.mem_activation_fn,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6MemActionHead(config)
        from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""
        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        if "vlm_content" in inputs:
            vlm_content_list = inputs["vlm_content"]
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Debug: print exactly which fields are None before any device/dtype moves.
        # Enable with: `export GROOT_DEBUG_NONE=1`
        if os.environ.get("GROOT_DEBUG_NONE", "0") == "1":
            def _collect_none_paths(obj, prefix=""):
                out = []
                if obj is None:
                    return [prefix or "<root>"]
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        out.extend(_collect_none_paths(v, f"{prefix}.{k}" if prefix else str(k)))
                elif isinstance(obj, (list, tuple)):
                    for i, v in enumerate(obj):
                        out.extend(_collect_none_paths(v, f"{prefix}[{i}]"))
                return out

            try:
                bi = dict(backbone_inputs) if hasattr(backbone_inputs, "items") else backbone_inputs
                ai = dict(action_inputs) if hasattr(action_inputs, "items") else action_inputs
                none_bi = _collect_none_paths(bi, "backbone_inputs")
                none_ai = _collect_none_paths(ai, "action_inputs")
                if none_bi or none_ai:
                    print("GROOT_DEBUG_NONE: None paths:", (none_bi + none_ai)[:200])
            except Exception as e:
                print("GROOT_DEBUG_NONE: failed to inspect None paths:", e)

        # Move to device and dtype
        def to_device_with_dtype(x):
            # `tree.map_structure` will traverse *all* leaves in the input structures.
            # Some leaves may be non-tensor metadata / scalars (e.g. `mem_token_id`) or None.
            # Only tensors should be moved to device / cast to dtype.
            if x is None:
                return None
            if not torch.is_tensor(x):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def _filter_mem_tokens_from_backbone(self, backbone_output: BatchFeature) -> BatchFeature:
        """
        Remove memory-token positions from the backbone sequence outputs.

        Motivation:
        - `mem_tokens` are extracted (per-step) from the backbone output and then processed
          by `MemoryModule` to produce `mem_out` (aka `memory_features`).
        - For conditioning the action head, we typically want *only* `mem_out` (read features),
          not the raw mem tokens duplicated inside `backbone_features`.

        This function filters out token positions where `mem_mask == True` while keeping
        tensor shapes batchable (pads to the max kept length in the batch and updates masks).
        """
        if "mem_mask" not in backbone_output:
            return backbone_output

        mem_mask = backbone_output["mem_mask"]  # [B, S] bool
        backbone_features = backbone_output["backbone_features"]  # [B, S, D]
        attn_mask = backbone_output["backbone_attention_mask"]  # [B, S] bool

        if mem_mask.dtype != torch.bool:
            mem_mask = mem_mask.to(dtype=torch.bool)
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(dtype=torch.bool)

        # Keep only valid (attendable) non-mem tokens.
        keep_mask = attn_mask & (~mem_mask)  # [B, S]

        B, S, D = backbone_features.shape
        keep_lens = keep_mask.sum(dim=1).tolist()
        max_keep = int(max(keep_lens)) if keep_lens else 0

        # If nothing to keep, fall back to original (should be rare / misconfigured).
        if max_keep == 0:
            return backbone_output

        device = backbone_features.device
        dtype = backbone_features.dtype

        new_features = torch.zeros((B, max_keep, D), device=device, dtype=dtype)
        new_attn_mask = torch.zeros((B, max_keep), device=device, dtype=torch.bool)

        has_image_mask = "image_mask" in backbone_output
        if has_image_mask:
            image_mask = backbone_output["image_mask"]
            if image_mask.dtype != torch.bool:
                image_mask = image_mask.to(dtype=torch.bool)
            new_image_mask = torch.zeros((B, max_keep), device=device, dtype=torch.bool)
        else:
            new_image_mask = None

        # Pack per-batch then pad.
        for b in range(B):
            idx = keep_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [L_b]
            if idx.numel() == 0:
                continue
            Lb = int(idx.numel())
            new_features[b, :Lb] = backbone_features[b].index_select(0, idx)
            new_attn_mask[b, :Lb] = True
            if has_image_mask:
                new_image_mask[b, :Lb] = image_mask[b].index_select(0, idx)

        backbone_output["backbone_features"] = new_features
        backbone_output["backbone_attention_mask"] = new_attn_mask
        if has_image_mask and new_image_mask is not None:
            backbone_output["image_mask"] = new_image_mask

        # After filtering, mem tokens are no longer present in the backbone sequence.
        backbone_output["mem_mask"] = torch.zeros_like(new_attn_mask, dtype=torch.bool)
        return backbone_output

    def extract_mem_tokens(
        self, backbone_output: BatchFeature, mem_token_id: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], BatchFeature]:
        """
        Extract memory tokens from backbone output.

        Args:
            backbone_output: Backbone output containing backbone_features
            mem_token_id: Token ID for memory tokens (optional, can be in inputs)

        Returns:
            mem_tokens: [B, N_mem, D] memory tokens or None if not found
            updated_backbone_output: Backbone output with mem_mask added
        """
        backbone_features = backbone_output["backbone_features"]  # [B, seq_len, D]
        B, seq_len, D = backbone_features.shape

        # Check if mem_mask is already in backbone_output
        if "mem_mask" in backbone_output:
            mem_mask = backbone_output["mem_mask"]  # [B, seq_len]
        elif "input_ids" in backbone_output and mem_token_id is not None:
            # Extract mem_mask from input_ids
            input_ids = backbone_output["input_ids"]  # [B, seq_len]
            mem_mask = input_ids == mem_token_id  # [B, seq_len]
            # Add to backbone_output for future use
            backbone_output["mem_mask"] = mem_mask
        else:
            # Cannot extract mem tokens - return None
            return None, backbone_output

        # Extract mem tokens: find positions where mem_mask is True
        mem_tokens_list = []
        num_found_total = int(mem_mask.sum().item())
        if num_found_total == 0:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                f"[BUG?] No <|mem|> tokens found in input_ids! "
                f"mem_token_id={mem_token_id}, input_ids unique values sample: "
                f"{backbone_output['input_ids'][0, :20].tolist() if 'input_ids' in backbone_output else 'N/A'}... "
                f"Expected {self.config.num_mem_tokens * B} mem tokens, found 0."
            )
        for b in range(B):
            mem_positions = mem_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [N_mem]
            if len(mem_positions) > 0:
                if len(mem_positions) == self.config.num_mem_tokens:
                    mem_tokens_b = backbone_features[b, mem_positions]  # [N_mem, D]
                elif len(mem_positions) < self.config.num_mem_tokens:
                    # Pad with zeros if fewer tokens found
                    mem_tokens_b = backbone_features[b, mem_positions]  # [found, D]
                    padding = torch.zeros(
                        self.config.num_mem_tokens - len(mem_positions),
                        D,
                        device=backbone_features.device,
                        dtype=backbone_features.dtype,
                    )
                    mem_tokens_b = torch.cat([mem_tokens_b, padding], dim=0)
                else:
                    # Take first N_mem tokens if more found
                    mem_tokens_b = backbone_features[b, mem_positions[: self.config.num_mem_tokens]]
            else:
                # No mem tokens found, create zeros
                mem_tokens_b = torch.zeros(
                    self.config.num_mem_tokens,
                    D,
                    device=backbone_features.device,
                    dtype=backbone_features.dtype,
                )
            mem_tokens_list.append(mem_tokens_b)

        mem_tokens = torch.stack(mem_tokens_list, dim=0)  # [B, N_mem, D]

        return mem_tokens, backbone_output

    def forward_step(
        self,
        inputs: dict,
        mem_state: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        return_attention: bool = False,
    ) -> BatchFeature:
        """
        Forward pass for a single step (used in chunk training).

        Args:
            inputs: Dictionary containing step inputs (same as forward)
            mem_state: Previous memory state [B, T_past, N_mem, D] or None
            compute_loss: Whether to compute loss (True for unroll, False for burn-in)
            return_attention: If True, also return attention weights from memory module

        Returns:
            BatchFeature containing:
                - loss: action prediction loss (if compute_loss=True)
                - mem_state: Updated memory state [B, T_new, N_mem, D]
                - mem_tokens: Current step memory tokens [B, N_mem, D]
                - mem_attention: (if return_attention=True) List of attention weights
        """
        # Prepare inputs
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Get mem_token_id from inputs if available
        mem_token_id = inputs.get("mem_token_id", None)

        # Forward through backbone (pass mem_token_id for debug)
        backbone_outputs = self.backbone(backbone_inputs, mem_token_id=mem_token_id)

        # Store input_ids in backbone_outputs for mem token extraction
        if "input_ids" in backbone_inputs:
            backbone_outputs["input_ids"] = backbone_inputs["input_ids"]

        # Extract memory tokens
        mem_tokens, backbone_outputs = self.extract_mem_tokens(backbone_outputs, mem_token_id)
        
        # ===== DEBUG: Check if mem_tokens vary across different inputs =====
        _debug_counter = getattr(self, '_debug_mem_counter', 0)
        if _debug_counter < 32:  # Only log first 32 forward calls
            import logging
            _logger = logging.getLogger(__name__)
            
            if mem_tokens is not None:
                # Compare with previous mem_tokens (if exists)
                prev_mt = getattr(self, '_prev_mem_tokens', None)
                
                mt_norm = mem_tokens.float().norm().item()
                mt_mean = mem_tokens.float().mean().item()
                
                if prev_mt is not None:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        prev_mt.flatten().float(),
                        mem_tokens.flatten().float(),
                        dim=0
                    ).item()
                    _logger.info(
                        f"[BACKBONE->MEM DEBUG] step={_debug_counter}: "
                        f"mem_tokens norm={mt_norm:.4f}, mean={mt_mean:.6f}, "
                        f"cos_sim(prev)={cos_sim:.6f}"
                    )
                    if cos_sim > 0.9999:
                        _logger.warning(
                            f"[BUG!] mem_tokens IDENTICAL across timesteps! "
                            f"<|mem|> token is NOT attending to image features!"
                        )
                else:
                    _logger.info(
                        f"[BACKBONE->MEM DEBUG] step={_debug_counter}: "
                        f"mem_tokens norm={mt_norm:.4f}, mean={mt_mean:.6f} (first step)"
                    )
                
                # Also check backbone_features variance
                bf = backbone_outputs["backbone_features"]
                _logger.info(
                    f"[BACKBONE->MEM DEBUG] step={_debug_counter}: "
                    f"backbone_features norm={bf.float().norm().item():.4f}, "
                    f"mean={bf.float().mean().item():.6f}, std={bf.float().std().item():.6f}"
                )
                
                self._prev_mem_tokens = mem_tokens.detach().clone()
            
            self._debug_mem_counter = _debug_counter + 1
        # ===== END DEBUG =====

        if mem_tokens is None:
            # Fallback: create zero mem tokens
            # WARNING: This means <|mem|> tokens were NOT found in input_ids!
            # This is likely a bug - memory tokens should be in the prompt.
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                f"[BUG?] mem_tokens is None! mem_token_id={mem_token_id}, "
                f"'input_ids' in backbone_outputs={('input_ids' in backbone_outputs)}, "
                f"'mem_mask' in backbone_outputs={('mem_mask' in backbone_outputs)}. "
                f"Creating zero mem_tokens - memory will NOT work correctly!"
            )
            B = backbone_outputs["backbone_features"].shape[0]
            mem_tokens = torch.zeros(
                B,
                self.config.num_mem_tokens,
                self.config.backbone_embedding_dim,
                device=backbone_outputs["backbone_features"].device,
                dtype=backbone_outputs["backbone_features"].dtype,
            )

        # Process through memory module
        mem_attention = None
        if compute_loss:
            # Unroll step: compute gradients
            if return_attention:
                mem_out, new_mem_state, mem_attention = self.memory_module(
                    mem_tokens, past_m=mem_state, return_attention=True
                )
            else:
                mem_out, new_mem_state = self.memory_module(mem_tokens, past_m=mem_state)
        else:
            # Burn-in step: no gradients
            with torch.no_grad():
                mem_out, new_mem_state = self.memory_module(mem_tokens, past_m=mem_state)
                # Detach to prevent gradients from flowing back
                new_mem_state = new_mem_state.detach()

        # Forward through action head with memory features
        if compute_loss:
            # Filter raw mem tokens out of the backbone conditioning sequence (avoid duplication).
            backbone_outputs = self._filter_mem_tokens_from_backbone(backbone_outputs)
            action_outputs = self.action_head(
                backbone_outputs, action_inputs, memory_features=mem_out
            )
        else:
            # Burn-in: don't compute loss, just return mem_state
            action_outputs = BatchFeature(
                data={
                    "loss": torch.tensor(0.0, device=mem_tokens.device),
                    "mem_state": new_mem_state,
                    "mem_tokens": mem_tokens,
                }
            )

        # Add memory-related outputs for monitoring/visualization
        action_outputs["mem_state"] = new_mem_state
        action_outputs["mem_tokens"] = mem_tokens
        action_outputs["mem_out"] = mem_out  # Memory output used for action conditioning
        
        # Add attention weights if requested
        if return_attention and mem_attention is not None:
            action_outputs["mem_attention"] = mem_attention

        return action_outputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model (for single-step training, not chunk).

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # For single-step, we don't use memory (fallback to standard behavior)
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        mem_token_id = inputs.get("mem_token_id", None)
        backbone_outputs = self.backbone(backbone_inputs, mem_token_id=mem_token_id)

        # Store input_ids for mem token extraction
        if "input_ids" in backbone_inputs:
            backbone_outputs["input_ids"] = backbone_inputs["input_ids"]

        # Try to extract mem tokens, but if not available, use empty memory
        mem_tokens, backbone_outputs = self.extract_mem_tokens(backbone_outputs, mem_token_id)
        if mem_tokens is None:
            B = backbone_outputs["backbone_features"].shape[0]
            mem_tokens = torch.zeros(
                B,
                self.config.num_mem_tokens,
                self.config.backbone_embedding_dim,
                device=backbone_outputs["backbone_features"].device,
                dtype=backbone_outputs["backbone_features"].dtype,
            )

        # Process through memory module (no past for single-step)
        mem_out, _ = self.memory_module(mem_tokens, past_m=None)

        # Forward through action head
        backbone_outputs = self._filter_mem_tokens_from_backbone(backbone_outputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs, memory_features=mem_out)

        # Expose mem tokens for debugging/analysis (optional)
        action_outputs["mem_tokens"] = mem_tokens
        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """Generate actions using the complete model."""
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        mem_token_id = inputs.get("mem_token_id", None)
        backbone_outputs = self.backbone(backbone_inputs, mem_token_id=mem_token_id)

        # Store input_ids for mem token extraction
        if "input_ids" in backbone_inputs:
            backbone_outputs["input_ids"] = backbone_inputs["input_ids"]

        # Extract mem tokens
        mem_tokens, backbone_outputs = self.extract_mem_tokens(backbone_outputs, mem_token_id)
        if mem_tokens is None:
            B = backbone_outputs["backbone_features"].shape[0]
            mem_tokens = torch.zeros(
                B,
                self.config.num_mem_tokens,
                self.config.backbone_embedding_dim,
                device=backbone_outputs["backbone_features"].device,
                dtype=backbone_outputs["backbone_features"].dtype,
            )

        # Process through memory module
        mem_out, _ = self.memory_module(mem_tokens, past_m=None)

        # Get action
        backbone_outputs = self._filter_mem_tokens_from_backbone(backbone_outputs)
        features = self.action_head._encode_features(backbone_outputs, action_inputs)
        action_outputs = self.action_head.get_action_with_features(
            backbone_features=features["backbone_features"],
            state_features=features["state_features"],
            embodiment_id=action_inputs.embodiment_id,
            backbone_output=backbone_outputs,
            memory_features=mem_out,
        )

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6Mem", Gr00tN1d6MemConfig)
AutoModel.register(Gr00tN1d6MemConfig, Gr00tN1d6Mem)
