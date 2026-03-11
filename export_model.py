"""
Export-compatible Qwen3.5 MoE model.

Transforms the eager model.py into a version that torch.export can trace:
  - forward(tokens, input_pos) signature for autoregressive decode
  - KV cache as registered buffers for full-attention layers
  - conv_state + recurrent_state as registered buffers for GDN layers
  - Export-friendly MoE dispatch (all-experts + gather, no data-dependent branching)

Reference: executorch/examples/models/llama/attention.py (KVCache, AttentionGatedDeltaNet)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from model import Qwen35MoEConfig, RMSNorm, RotaryEmbedding, MLP


# ---------------------------------------------------------------------------
# KV Cache — registered buffers for full-attention layers

class KVCache(nn.Module):

    def __init__(self, n_heads, head_dim, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.register_buffer("k_cache", torch.zeros(1, n_heads, max_seq_len, head_dim))
        self.register_buffer("v_cache", torch.zeros(1, n_heads, max_seq_len, head_dim))

    def update(self, input_pos, k_val, v_val):
        # input_pos: (S,), k_val/v_val: (1, H, S, D)
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


# ---------------------------------------------------------------------------
# Full Attention with KV cache

class ExportCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_kv_groups = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

        self.kv_cache = KVCache(self.n_kv_head, self.head_dim, config.block_size)

        # pre-registered causal mask
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x, input_pos):
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q_norm(q.view(B, T, self.n_head, self.head_dim)).view(B, T, -1)
        k = self.k_norm(k.view(B, T, self.n_kv_head, self.head_dim)).view(B, T, -1)
        q, k = self.rotary_emb(input_pos, q, k)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # update KV cache
        k, v = self.kv_cache.update(input_pos, k, v)

        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        attn_mask = self.mask[input_pos]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# GatedDeltaNet with state buffers

class ExportGatedDeltaNet(nn.Module):

    def __init__(self, config, use_scan=True):
        super().__init__()
        self.use_scan = use_scan
        self.n_k_heads = config.linear_num_key_heads
        self.n_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.n_k_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkvz = nn.Linear(config.n_embd, self.conv_dim + self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.n_embd, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.n_embd, self.n_v_heads, bias=False)

        # conv1d with no padding — we manage state manually
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, config.linear_conv_kernel_dim,
            groups=self.conv_dim, padding=0, bias=False,
        )

        self.norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.n_embd, bias=False)

        # state buffers
        self.register_buffer(
            "conv_state", torch.zeros(1, self.conv_dim, config.linear_conv_kernel_dim)
        )
        self.register_buffer(
            "recurrent_state", torch.zeros(1, self.n_k_heads, self.head_k_dim, self.head_v_dim)
        )

    def forward(self, x, input_pos):
        B, T, C = x.size()

        # reset state at position 0 (exportable — no Python if)
        reset = (input_pos[0] == 0).to(self.conv_state.dtype)
        keep = 1.0 - reset
        self.conv_state[:B].mul_(keep)
        self.recurrent_state[:B].mul_(keep)

        # projections
        qkvz = self.in_proj_qkvz(x)
        qkv, z = qkvz.split([self.conv_dim, self.value_dim], dim=-1)
        z = z.view(B, T, self.n_v_heads, self.head_v_dim)

        beta = torch.sigmoid(self.in_proj_b(x))
        alpha = torch.sigmoid(self.in_proj_a(x))

        # causal conv1d with state
        qkv_t = qkv.transpose(1, 2)  # (B, conv_dim, T)
        conv_input = torch.cat([self.conv_state[:B], qkv_t], dim=-1)  # (B, conv_dim, K+T)
        with torch.no_grad():
            self.conv_state[:B].copy_(conv_input[:, :, -self.conv_kernel_size:])
        qkv_conv = F.conv1d(conv_input, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)
        qkv_conv = F.silu(qkv_conv[:, :, -T:])  # (B, conv_dim, T)
        qkv_conv = qkv_conv.transpose(1, 2)  # (B, T, conv_dim)

        # split and normalize
        q, k, v = qkv_conv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(q.view(B, T, self.n_k_heads, self.head_k_dim), p=2, dim=-1)
        k = F.normalize(k.view(B, T, self.n_k_heads, self.head_k_dim), p=2, dim=-1)
        v = v.view(B, T, self.n_v_heads, self.head_v_dim)

        # delta rule recurrence
        # use_scan=True: torch.scan (supports dynamic T, but not all backends)
        # use_scan=False: for loop (unrolled at trace time, requires static T)
        state = self.recurrent_state[:B].clone()

        if self.use_scan:
            from torch._higher_order_ops.scan import scan

            q_t = q.transpose(0, 1).contiguous()
            k_t = k.transpose(0, 1).contiguous()
            v_t = v.transpose(0, 1).contiguous()
            alpha_t = alpha.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).contiguous()
            beta_t = beta.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).contiguous()

            def step_fn(state, xs):
                q_i, k_i, v_i, a_i, b_i = xs
                new_state = a_i * state + b_i * (k_i.unsqueeze(-1) * v_i.unsqueeze(-2))
                out_i = torch.einsum('bhk,bhkv->bhv', q_i, new_state)
                return new_state.clone(), out_i.clone()

            state, outputs_stacked = scan(step_fn, state, (q_t, k_t, v_t, alpha_t, beta_t))
            output = outputs_stacked.transpose(0, 1)
        else:
            outputs = []
            for t in range(T):
                beta_t = beta[:, t, :, None, None]
                alpha_t = alpha[:, t, :, None, None]
                state = alpha_t * state + beta_t * (k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2))
                outputs.append(torch.einsum('bhk,bhkv->bhv', q[:, t], state))
            output = torch.stack(outputs, dim=1)

        with torch.no_grad():
            self.recurrent_state[:B].copy_(state)

        output = self.norm(output) * F.sigmoid(z)
        return self.out_proj(output.reshape(B, T, -1))


# ---------------------------------------------------------------------------
# Export-friendly MoE: stacked expert weights + index by top-k indices.
# Only the selected experts' weights participate in computation.
# Pattern from ET's ConditionalFeedForward (llama_transformer.py).

class ConditionalFeedForward(nn.Module):
    """All expert weights stacked as (E, H, D) tensors. Indexed by expert_indices."""

    def __init__(self, n_embd, intermediate_size, n_experts):
        super().__init__()
        self.w_gate = nn.Parameter(torch.randn(n_experts, intermediate_size, n_embd))  # (E, H, D)
        self.w_up = nn.Parameter(torch.randn(n_experts, intermediate_size, n_embd))    # (E, H, D)
        self.w_down = nn.Parameter(torch.randn(n_experts, n_embd, intermediate_size))  # (E, D, H)

    def forward(self, x, expert_indices):
        # x: (T, D), expert_indices: (T, top_k)
        # w_gate, w_up: (E, H, D) — same layout as nn.Linear.weight
        # w_down: (E, D, H) — transposed from nn.Linear(H, D).weight which is (D, H)
        w_gate = self.w_gate[expert_indices]   # (T, A, H, D)
        w_up = self.w_up[expert_indices]       # (T, A, H, D)
        w_down = self.w_down[expert_indices]   # (T, A, D, H)
        x1 = F.silu(torch.einsum("td,tahd -> tah", x, w_gate))
        x3 = torch.einsum("td,tahd -> tah", x, w_up)
        return torch.einsum("tah,tadh -> tad", x1 * x3, w_down)


class ExportSparseMoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.top_k = config.n_experts_per_tok
        self.n_experts = config.n_routed_experts
        self.dim = config.n_embd

        self.gate = nn.Linear(config.n_embd, config.n_routed_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(
            config.n_embd, config.expert_intermediate_size, config.n_routed_experts
        )
        self.shared_expert = MLP(config.n_embd, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)

        # Route: softmax → top-k
        scores = self.gate(x_flat)                                     # (T, E)
        expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)                # (T, top_k)

        # Only selected experts run (indexed into stacked weights)
        expert_outs = self.cond_ffn(x_flat, expert_indices)            # (T, top_k, D)
        routed_out = torch.einsum("tai,ta->ti", expert_outs, expert_weights)

        # Shared expert
        shared_out = self.shared_expert(x_flat)
        shared_gate = torch.sigmoid(self.shared_expert_gate(x_flat))
        return (routed_out + shared_gate * shared_out).view(B, T, C)


# ---------------------------------------------------------------------------
# Block and full model

class ExportBlock(nn.Module):

    def __init__(self, config, layer_idx, use_scan=True):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)

        if self.layer_type == "full_attention":
            self.attn = ExportCausalSelfAttention(config)
        else:
            self.attn = ExportGatedDeltaNet(config, use_scan=use_scan)

        self.mlp = ExportSparseMoE(config)

    def forward(self, x, input_pos):
        x = x + self.attn(self.ln_1(x), input_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class ExportQwen35MoE(nn.Module):

    def __init__(self, config, use_scan=True):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            ExportBlock(config, layer_idx=i, use_scan=use_scan)
            for i in range(config.n_layer)
        ])
        self.ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, tokens: torch.LongTensor, input_pos: torch.LongTensor) -> torch.Tensor:
        x = self.wte(tokens)
        for layer in self.layers:
            x = layer(x, input_pos)
        x = self.ln_f(x)
        return self.lm_head(x)

    @staticmethod
    def from_checkpoint(ckpt_path, device='cpu', use_scan=True):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt['config']

        export_model = ExportQwen35MoE(config, use_scan=use_scan)
        eager_sd = ckpt['model']

        # Remap eager keys → export keys, and stack per-expert weights
        new_sd = {}
        # Collect per-expert weights for stacking
        expert_weights = {}  # (layer_idx, proj_name, expert_idx) → tensor

        for k, v in eager_sd.items():
            # Remap prefixes
            ek = k.replace('transformer.wte.', 'wte.')
            ek = ek.replace('transformer.ln_f.', 'ln_f.')
            ek = ek.replace('transformer.h.', 'layers.')

            # Check if this is a per-expert weight: layers.N.mlp.experts.E.{gate,up,down}_proj.weight
            import re
            m = re.match(r'layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight', ek)
            if m:
                layer_idx, expert_idx, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                expert_weights[(layer_idx, proj, expert_idx)] = v
            else:
                new_sd[ek] = v

        # Stack per-expert weights into (E, H, D) tensors for ConditionalFeedForward
        proj_map = {'gate': 'w_gate', 'up': 'w_up', 'down': 'w_down'}
        for layer_idx in range(config.n_layer):
            for proj, param_name in proj_map.items():
                stacked = torch.stack([
                    expert_weights[(layer_idx, proj, e)]
                    for e in range(config.n_routed_experts)
                ], dim=0)  # (E, H, D) or (E, D, H) depending on Linear layout
                new_sd[f'layers.{layer_idx}.mlp.cond_ffn.{param_name}'] = stacked

        export_model.load_state_dict(new_sd, strict=False)
        return export_model, config
