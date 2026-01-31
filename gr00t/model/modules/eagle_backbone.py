import os

import torch
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature


class EagleBackbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        """
        EagleBackbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Eagle-Block2A-2B-v2
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            extra_kwargs["attn_implementation"] = "flash_attention_2"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        if model_name == "nvidia/Eagle-Block2A-2B-v2":
            assert use_flash_attention, (
                "nvidia/Eagle-Block2A-2B-v2 requires flash attention by default"
            )
            assert load_bf16, "nvidia/Eagle-Block2A-2B-v2 requires bfloat16 by default"
            eagle_path = os.path.join(os.path.dirname(__file__), "nvidia", "Eagle-Block2A-2B-v2")
            config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
            self.model = AutoModel.from_config(config, trust_remote_code=True)
        else:
            raise ValueError(f"Model {model_name} not supported")

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    print(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature, mem_token_id: int = None) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input_filtered = {k: vl_input[k] for k in keys_to_use}
        
        # ===== DEBUG: Hook to capture Q, K before FlashAttention =====
        _debug_counter = getattr(self, '_debug_attn_counter', 0)
        captured_qk = {}
        hook_handle = None
        
        if _debug_counter < 5 and mem_token_id is not None:
            # Hook the last LLM layer's self-attention
            try:
                # Get the last layer's attention module
                last_layer = self.model.language_model.model.layers[-1]
                attn_module = last_layer.self_attn
                
                def qk_hook(module, args, kwargs):
                    # Capture hidden_states before attention
                    # args[0] is usually hidden_states
                    if len(args) > 0:
                        hidden_states = args[0]
                        captured_qk['hidden_states'] = hidden_states.detach()
                    return None
                
                hook_handle = attn_module.register_forward_pre_hook(qk_hook, with_kwargs=True)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"[ATTN DEBUG] Could not register hook: {e}")
        # ===== END HOOK SETUP =====
        
        outputs = self.model(**vl_input_filtered, output_hidden_states=True)
        final_hidden_states = outputs["hidden_states"][-1]
        
        # Remove hook
        if hook_handle is not None:
            hook_handle.remove()
        
        # ===== DEBUG: Basic token stats + Q·K scores =====
        if _debug_counter < 5 and mem_token_id is not None:
            import logging
            _logger = logging.getLogger(__name__)

            try:
                input_ids = vl_input_filtered["input_ids"]  # [B, seq_len]
                mem_mask = input_ids == mem_token_id  # [B, seq_len]
                image_mask_debug = input_ids == self.model.config.image_token_index  # [B, seq_len]
                num_mem = int(mem_mask.sum().item())
                num_img = int(image_mask_debug.sum().item())
                _logger.info(
                    f"[QK DEBUG] step={_debug_counter}: "
                    f"num_mem={num_mem}, num_img={num_img}, seq_len={input_ids.shape[1]}"
                )
                if num_img == 0:
                    _logger.warning(
                        "[QK DEBUG] No image tokens found in input_ids. "
                        "pixel_values may be ignored by the backbone!"
                    )

                if mem_mask.any() and image_mask_debug.any() and "hidden_states" in captured_qk:
                    mem_positions = mem_mask[0].nonzero(as_tuple=False).squeeze(-1)
                    img_positions = image_mask_debug[0].nonzero(as_tuple=False).squeeze(-1)

                    # Get hidden states captured before attention
                    hs = captured_qk["hidden_states"]  # [B, seq_len, D]

                    # Get Q, K projections from attention module
                    last_layer = self.model.language_model.model.layers[-1]
                    attn = last_layer.self_attn

                    # Compute Q and K manually
                    if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
                        Q = attn.q_proj(hs)  # [B, seq_len, num_heads * head_dim]
                        K = attn.k_proj(hs)  # [B, seq_len, num_heads * head_dim]
                    elif hasattr(attn, "qkv_proj"):
                        qkv = attn.qkv_proj(hs)
                        Q, K, _ = qkv.chunk(3, dim=-1)
                    else:
                        _logger.warning(f"[QK DEBUG] Unknown attention structure: {type(attn)}")
                        Q = K = None

                    if Q is not None and K is not None:
                        # Get first mem token's Q
                        mem_pos = mem_positions[0].item()
                        Q_mem = Q[0, mem_pos]  # [D]

                        # Get K for mem tokens and img tokens
                        K_mem = K[0, mem_positions]  # [num_mem, D]
                        K_img = K[0, img_positions]  # [num_img, D]

                        # Compute dot products (attention scores before softmax)
                        qk_mem = torch.matmul(Q_mem.unsqueeze(0).float(), K_mem.float().T)  # [1, num_mem]
                        qk_mem_max = qk_mem.max().item()
                        qk_mem_mean = qk_mem.mean().item()

                        qk_img = torch.matmul(Q_mem.unsqueeze(0).float(), K_img.float().T)  # [1, num_img]
                        qk_img_max = qk_img.max().item()
                        qk_img_mean = qk_img.mean().item()

                        # Normalize by sqrt(d) to get actual attention logits
                        head_dim = Q.shape[-1] // getattr(attn, "num_heads", 32)
                        scale = head_dim**0.5

                        _logger.info(
                            f"[QK DEBUG] Q_mem · K_mem: max={qk_mem_max/scale:.4f}, mean={qk_mem_mean/scale:.4f}"
                        )
                        _logger.info(
                            f"[QK DEBUG] Q_mem · K_img: max={qk_img_max/scale:.4f}, mean={qk_img_mean/scale:.4f}"
                        )
                        _logger.info(
                            f"[QK DEBUG] Ratio (mem/img): max={qk_mem_max/max(qk_img_max, 1e-6):.4f}, "
                            f"mean={qk_mem_mean/max(qk_img_mean, 1e-6):.4f}"
                        )

                        if qk_mem_max > qk_img_max * 2:
                            _logger.warning(
                                "[QK DEBUG] ⚠️ <|mem|> strongly prefers SELF over IMAGE!"
                            )
                elif "hidden_states" not in captured_qk:
                    _logger.warning(
                        "[QK DEBUG] Hook did not capture hidden_states. "
                        "FlashAttention path may bypass the pre-hook."
                    )

                # Track if image token hidden states actually change across steps
                if image_mask_debug.any():
                    img_positions = image_mask_debug[0].nonzero(as_tuple=False).squeeze(-1)
                    img_hs = final_hidden_states[0, img_positions].mean(dim=0)
                    prev_img_hs = getattr(self, "_prev_img_hs", None)
                    img_norm = img_hs.float().norm().item()
                    if prev_img_hs is not None:
                        img_cos = torch.nn.functional.cosine_similarity(
                            prev_img_hs.flatten().float(),
                            img_hs.flatten().float(),
                            dim=0,
                        ).item()
                        _logger.info(
                            f"[QK DEBUG] img_hs norm={img_norm:.4f}, cos_sim(prev)={img_cos:.6f}"
                        )
                    else:
                        _logger.info(f"[QK DEBUG] img_hs norm={img_norm:.4f} (first)")
                    self._prev_img_hs = img_hs.detach()

            except Exception as e:
                import traceback
                _logger.error(f"[QK DEBUG] Error: {e}\n{traceback.format_exc()}")

        self._debug_attn_counter = _debug_counter + 1
        # ===== END DEBUG =====
        
        image_mask = vl_input_filtered["input_ids"] == self.model.config.image_token_index
        attention_mask = vl_input_filtered["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": final_hidden_states,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
