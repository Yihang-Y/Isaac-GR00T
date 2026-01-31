from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal self-attention block for memory module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** (-0.5)
        self.inner_dim = num_heads * head_dim

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] input tensor
            mask: [B, T] optional attention mask (True = attend, False = mask out)
            return_attention: If True, also return attention weights [B, H, T, T]

        Returns:
            [B, T, D] output tensor, or (output, attention_weights) if return_attention=True
        """
        B, T, D = x.shape
        qkv = self.to_qkv(x)  # [B, T, 3 * inner_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, T, inner_dim]

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Apply causal mask (lower triangular)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply optional attention mask
        if mask is not None:
            # mask: [B, T] -> [B, 1, 1, T] for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn_for_output = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn_for_output, v)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)  # [B, T, inner_dim]

        if return_attention:
            return self.to_out(out), attn  # Return pre-dropout attention for visualization
        return self.to_out(out)


class MemoryTransformerBlock(nn.Module):
    """Single transformer block for memory module with causal self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        ff_bias: bool = True,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, head_dim, dropout, attn_bias)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward network
        if activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "geglu":
            # GEGLU: Gated GLU
            self.activation = None  # Will use custom GEGLU
            ff_inner_dim = dim * 2
        else:
            raise ValueError(f"Unsupported activation: {activation_fn}")

        if activation_fn == "geglu":
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_inner_dim, bias=ff_bias),
                nn.GELU(),
                nn.Linear(ff_inner_dim, dim, bias=ff_bias),
            )
        else:
            ff_inner_dim = dim * 4
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_inner_dim, bias=ff_bias),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(ff_inner_dim, dim, bias=ff_bias),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, T, D]
            mask: Optional attention mask [B, T]
            return_attention: If True, return attention weights from self-attention
            
        Returns:
            Output tensor, or (output, attention_weights) if return_attention=True
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(x_norm, mask, return_attention=True)
        else:
            attn_out = self.attn(x_norm, mask)
            attn_weights = None
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        if return_attention:
            return x, attn_weights
        return x


class MemoryModule(nn.Module):
    """
    Memory module that processes past and current memory tokens using a causal transformer.

    This module maintains a history of memory tokens and processes them causally to generate
    memory features that can be used for conditioning the action prediction.

    Key design choices:
    1. **Temporal positional encoding**: Each timestep gets a learnable positional embedding
       so the model can distinguish between different time steps in history.
    2. **Original input for memory state**: The memory state stores the original input tokens
       (not transformer outputs) to prevent feature drift over long sequences.

    Args:
        mem_token_dim: Dimension of each memory token (should match VLM output dim)
        num_mem_tokens: Number of memory tokens per timestep (N_mem)
        mem_out_tokens: Number of output memory tokens (K)
        max_history_steps: Maximum number of past steps to keep in memory
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        dropout: Dropout rate
        activation_fn: Activation function ("gelu" or "geglu")
    """

    def __init__(
        self,
        mem_token_dim: int,
        num_mem_tokens: int = 8,
        mem_out_tokens: int = 8,
        max_history_steps: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        self.mem_token_dim = mem_token_dim
        self.num_mem_tokens = num_mem_tokens
        self.mem_out_tokens = mem_out_tokens
        self.max_history_steps = max_history_steps

        # Temporal positional encoding: learned embeddings for each timestep
        # Shape: [max_history_steps, D] - will be broadcast across N_mem tokens per step
        self.temporal_pos_embedding = nn.Embedding(max_history_steps, mem_token_dim)
        nn.init.normal_(self.temporal_pos_embedding.weight, mean=0.0, std=0.02)

        # Token-level positional encoding within each timestep
        # Shape: [N_mem, D] - distinguishes tokens within the same timestep
        self.token_pos_embedding = nn.Embedding(num_mem_tokens, mem_token_dim)
        nn.init.normal_(self.token_pos_embedding.weight, mean=0.0, std=0.02)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                MemoryTransformerBlock(
                    dim=mem_token_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection to generate mem_out_tokens from num_mem_tokens
        # This projection operates on the token dimension: [B, N_mem, D] -> [B, K, D]
        # Implemented as transpose -> linear -> transpose
        if mem_out_tokens != num_mem_tokens:
            self.output_proj = nn.Linear(num_mem_tokens, mem_out_tokens)
            self.use_output_proj = True
        else:
            self.output_proj = None
            self.use_output_proj = False

        # Final layer norm
        self.norm_out = nn.LayerNorm(mem_token_dim)

    def _add_positional_encoding(
        self,
        seq: torch.Tensor,
        T_total: int,
    ) -> torch.Tensor:
        """
        Add temporal and token-level positional encodings to the sequence.

        Args:
            seq: Input sequence [B, T_total * N_mem, D]
            T_total: Number of timesteps in the sequence

        Returns:
            Sequence with positional encodings added [B, T_total * N_mem, D]
        """
        B = seq.shape[0]
        device = seq.device

        # Create temporal position indices [T_total]
        # Use the last T_total positions to handle variable history lengths
        # This ensures the most recent step always gets the same position
        if T_total <= self.max_history_steps:
            temporal_pos = torch.arange(T_total, device=device)
        else:
            # If somehow T_total > max_history_steps, clamp to valid range
            temporal_pos = torch.arange(T_total, device=device) % self.max_history_steps

        # Create token position indices [N_mem]
        token_pos = torch.arange(self.num_mem_tokens, device=device)

        # Get embeddings
        temporal_emb = self.temporal_pos_embedding(temporal_pos)  # [T_total, D]
        token_emb = self.token_pos_embedding(token_pos)  # [N_mem, D]

        # Expand and combine positional embeddings
        # temporal_emb: [T_total, D] -> [T_total, 1, D] -> [T_total, N_mem, D]
        # token_emb: [N_mem, D] -> [1, N_mem, D] -> [T_total, N_mem, D]
        temporal_emb_expanded = temporal_emb.unsqueeze(1).expand(-1, self.num_mem_tokens, -1)
        token_emb_expanded = token_emb.unsqueeze(0).expand(T_total, -1, -1)

        # Combined positional encoding: [T_total, N_mem, D] -> [T_total * N_mem, D]
        pos_encoding = (temporal_emb_expanded + token_emb_expanded).view(-1, self.mem_token_dim)

        # Add to sequence: [B, T_total * N_mem, D] + [1, T_total * N_mem, D]
        seq = seq + pos_encoding.unsqueeze(0)

        return seq

    def forward(
        self,
        cur_m: torch.Tensor,
        past_m: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Process current and past memory tokens to generate memory output and updated state.

        Args:
            cur_m: Current memory tokens [B, N_mem, D]
            past_m: Past memory tokens [B, T_past, N_mem, D] or None (for first step)
            mask: Optional mask [B, T_total] indicating valid tokens
            return_attention: If True, also return attention weights from each layer

        Returns:
            mem_out: Memory output tokens [B, K, D]
            new_mem_state: Updated memory state [B, T_new, N_mem, D] (T_new <= max_history_steps)
                           Note: Memory state stores ORIGINAL input tokens (not transformer outputs)
                           to prevent feature drift over long sequences.
            attention_weights: (only if return_attention=True) List of attention weights [B, H, T, T]
                              from each transformer layer
        """
        B = cur_m.shape[0]
        device = cur_m.device
        dtype = cur_m.dtype

        # Prepare input sequence
        if past_m is None:
            # First step: only current tokens
            seq = cur_m  # [B, N_mem, D]
            T_total = 1
            T_past = 0
        else:
            # Concatenate past and current
            T_past = past_m.shape[1]
            # Flatten past: [B, T_past, N_mem, D] -> [B, T_past * N_mem, D]
            past_flat = past_m.view(B, T_past * self.num_mem_tokens, self.mem_token_dim)
            # Current: [B, N_mem, D]
            seq = torch.cat([past_flat, cur_m], dim=1)  # [B, (T_past * N_mem + N_mem), D]
            T_total = T_past + 1

        # Add temporal and token-level positional encodings
        seq = self._add_positional_encoding(seq, T_total)

        # Apply transformer layers
        x = seq
        attention_weights_list = [] if return_attention else None
        for layer in self.layers:
            if return_attention:
                x, attn_weights = layer(x, mask=mask, return_attention=True)
                attention_weights_list.append(attn_weights)
            else:
                x = layer(x, mask=mask)

        x = self.norm_out(x)

        # Extract output from the last N_mem tokens (current step)
        if past_m is None:
            mem_out_raw = x  # [B, N_mem, D]
        else:
            mem_out_raw = x[:, -self.num_mem_tokens:]  # [B, N_mem, D]

        # Project to desired output size if needed using learned projection
        if self.use_output_proj:
            # [B, N_mem, D] -> [B, D, N_mem] -> Linear -> [B, D, K] -> [B, K, D]
            mem_out = self.output_proj(mem_out_raw.transpose(1, 2)).transpose(1, 2)
        else:
            mem_out = mem_out_raw

        # ===== Update memory state using ORIGINAL input tokens =====
        # IMPORTANT: We store the original input tokens (cur_m), NOT the transformer outputs.
        # This prevents feature drift where memory tokens accumulate transformer transformations
        # over long sequences, causing them to drift away from the original feature distribution.
        if past_m is None:
            new_mem_state = cur_m.unsqueeze(1)  # [B, 1, N_mem, D]
        else:
            # Append current ORIGINAL tokens to past memory state
            cur_reshaped = cur_m.unsqueeze(1)  # [B, 1, N_mem, D]
            combined = torch.cat([past_m, cur_reshaped], dim=1)  # [B, T_past + 1, N_mem, D]

            # Truncate to max_history_steps (keep most recent)
            if combined.shape[1] > self.max_history_steps:
                new_mem_state = combined[:, -self.max_history_steps:]
            else:
                new_mem_state = combined

        if return_attention:
            return mem_out, new_mem_state, attention_weights_list
        return mem_out, new_mem_state
