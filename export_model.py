"""
Export-compatible Qwen3.5 MoE model.

Transforms the eager model.py into a version that torch.export can trace:
  - forward(tokens, input_pos) signature for autoregressive decode
  - KV cache as registered buffers for full-attention layers
  - conv_state + recurrent_state as registered buffers for GDN layers
  - Export-friendly MoE: grouped nn.Linear experts (quantization-compatible)

Reference: executorch/examples/models/qwen3_5_moe/ (full-size model)
"""

import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from model import Qwen35MoEConfig, RMSNorm, RMSNormGated, RotaryEmbedding, MLP


# ---------------------------------------------------------------------------
# KV Cache — registered buffers for full-attention layers

class KVCache(nn.Module):

    def __init__(self, n_heads, head_dim, max_seq_len):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(1, n_heads, max_seq_len, head_dim))
        self.register_buffer("v_cache", torch.zeros(1, n_heads, max_seq_len, head_dim))

    def update(self, input_pos, k_val, v_val):
        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val
        return self.k_cache, self.v_cache


# ---------------------------------------------------------------------------
# Full Attention with KV cache + output gate

class ExportCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_kv_groups = self.n_head // self.n_kv_head

        # q_proj includes output gate: 2x heads
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, config.partial_rotary_factor, config.rope_theta,
        )

        self.kv_cache = KVCache(self.n_kv_head, self.head_dim, config.block_size)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x, input_pos):
        B, T, _ = x.size()
        dtype = x.dtype

        q_and_gate = self.q_proj(x).view(B, T, self.n_head, self.head_dim * 2)
        q = q_and_gate[..., :self.head_dim]
        gate = q_and_gate[..., self.head_dim:]

        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(input_pos, q, k)

        q = q.to(dtype).transpose(1, 2)
        k = k.to(dtype).transpose(1, 2)
        v = v.transpose(1, 2)

        k, v = self.kv_cache.update(input_pos, k, v)

        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        attn_mask = self.mask[input_pos].unsqueeze(0).unsqueeze(0)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        gate = gate.reshape(B, T, -1)
        y = y * torch.sigmoid(gate)

        return self.o_proj(y)


# ---------------------------------------------------------------------------
# GatedDeltaNet with state buffers

class ExportGatedDeltaNet(nn.Module):

    def __init__(self, config, use_fla=False):
        super().__init__()
        self.use_fla = use_fla
        self.n_k_heads = config.linear_num_key_heads
        self.n_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.n_k_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        assert self.n_v_heads % self.n_k_heads == 0
        self.head_repeat = self.n_v_heads // self.n_k_heads

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(config.n_embd, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.n_embd, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.n_embd, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.n_embd, self.n_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, config.linear_conv_kernel_dim,
            groups=self.conv_dim, padding=0, bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        A = torch.empty(self.n_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.n_embd, bias=False)

        # State buffers
        self.register_buffer(
            "conv_state", torch.zeros(1, self.conv_dim, config.linear_conv_kernel_dim)
        )
        self.register_buffer(
            "recurrent_state", torch.zeros(1, self.n_v_heads, self.head_k_dim, self.head_v_dim)
        )

    def forward(self, x, input_pos):
        B, T, C = x.size()

        # Reset state at position 0 (exportable — no Python if)
        reset = (input_pos[0] == 0).to(self.conv_state.dtype)
        keep = 1.0 - reset
        self.conv_state[:B].mul_(keep)
        self.recurrent_state[:B].mul_(keep)

        # Projections
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, T, self.n_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        # Causal conv1d with state
        qkv_t = qkv.transpose(1, 2)
        conv_input = torch.cat([self.conv_state[:B], qkv_t], dim=-1)
        with torch.no_grad():
            self.conv_state[:B].copy_(conv_input[:, :, -self.conv_kernel_size:])
        qkv_conv = F.conv1d(
            conv_input, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim
        )
        qkv_conv = F.silu(qkv_conv[:, :, -T:]).transpose(1, 2)

        # Split and L2-normalize Q and K
        kd = self.key_dim
        q = qkv_conv[..., :kd].reshape(B, T, self.n_k_heads, self.head_k_dim)
        k = qkv_conv[..., kd:2*kd].reshape(B, T, self.n_k_heads, self.head_k_dim)
        v = qkv_conv[..., 2*kd:].reshape(B, T, self.n_v_heads, self.head_v_dim)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        if self.head_repeat > 1:
            q = q.repeat_interleave(self.head_repeat, dim=2)
            k = k.repeat_interleave(self.head_repeat, dim=2)

        # Mamba-style gating
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Delta rule: FLA Triton kernel (CUDA) or torch.scan (portable)
        state = self.recurrent_state[:B].clone()

        if self.use_fla:
            output, state = torch.ops.triton.chunk_gated_delta_rule(
                q, k, v, g, beta, state
            )
        else:
            from torch._higher_order_ops.scan import scan

            q_t = q.transpose(0, 1).contiguous()
            k_t = k.transpose(0, 1).contiguous()
            v_t = v.transpose(0, 1).contiguous()
            g_t = g.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).contiguous()
            beta_t = beta.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).contiguous()

            def step_fn(carry, xs):
                q_i, k_i, v_i, g_i, b_i = xs
                new_carry = torch.exp(g_i) * carry + b_i * (
                    k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
                )
                out_i = torch.einsum('bhk,bhkv->bhv', q_i, new_carry)
                return new_carry.clone(), out_i.clone()

            state, outputs_stacked = scan(step_fn, state, (q_t, k_t, v_t, g_t, beta_t))
            output = outputs_stacked.transpose(0, 1)

        with torch.no_grad():
            self.recurrent_state[:B].copy_(state)

        # RMSNorm(output) * silu(z)
        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(B, T, -1)

        return self.out_proj(output)


# ---------------------------------------------------------------------------
# Export-friendly MoE: grouped nn.Linear experts for quantization.
# Experts are split into groups so each nn.Linear stays small enough for
# tinygemm int4 packing. quantize_model_() handles them automatically.

_EXPERTS_PER_GROUP = 4


class ConditionalFeedForward(nn.Module):

    def __init__(self, n_embd, intermediate_size, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = n_embd
        G = _EXPERTS_PER_GROUP
        assert n_experts % G == 0
        num_groups = n_experts // G

        self.gate_up_projs = nn.ModuleList([
            nn.Linear(n_embd, G * intermediate_size * 2, bias=False)
            for _ in range(num_groups)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(intermediate_size, G * n_embd, bias=False)
            for _ in range(num_groups)
        ])

    def forward(self, x, expert_indices):
        T = x.size(0)
        top_k = expert_indices.size(1)
        G = _EXPERTS_PER_GROUP
        H = self.intermediate_size
        D = self.hidden_size

        # Gate + Up: compute per-group, cat, gather top-k
        gate_up_parts = [proj(x).view(T, G, 2, H) for proj in self.gate_up_projs]
        gate_up = torch.cat(gate_up_parts, dim=1)  # (T, E, 2, H)

        idx = expert_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, H)
        gate_up_sel = gate_up.gather(1, idx)  # (T, top_k, 2, H)
        intermediate = F.silu(gate_up_sel[:, :, 0, :]) * gate_up_sel[:, :, 1, :]

        # Down: compute per-group, cat, gather correct expert per slot
        intermediate_flat = intermediate.reshape(T * top_k, H)
        down_parts = [
            proj(intermediate_flat).view(T, top_k, G, D) for proj in self.down_projs
        ]
        all_down = torch.cat(down_parts, dim=2)  # (T, top_k, E, D)

        eidx = expert_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)
        return all_down.gather(2, eidx).squeeze(2)  # (T, top_k, D)


class ExportSparseMoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.top_k = config.n_experts_per_tok

        self.gate = nn.Linear(config.n_embd, config.n_routed_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(
            config.n_embd, config.expert_intermediate_size, config.n_routed_experts,
        )
        self.shared_expert = MLP(config.n_embd, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)

        scores = self.gate(x_flat)
        expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)

        expert_outs = self.cond_ffn(x_flat, expert_indices)
        routed_out = torch.einsum("tai,ta->ti", expert_outs, expert_weights)

        shared_out = self.shared_expert(x_flat)
        shared_gate = torch.sigmoid(self.shared_expert_gate(x_flat))
        return (routed_out + shared_gate * shared_out).view(B, T, C)


# ---------------------------------------------------------------------------
# Block and full model

class ExportBlock(nn.Module):

    def __init__(self, config, layer_idx, use_fla=False):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)

        if self.layer_type == "full_attention":
            self.attn = ExportCausalSelfAttention(config)
        else:
            self.attn = ExportGatedDeltaNet(config, use_fla=use_fla)

        self.mlp = ExportSparseMoE(config)

    def forward(self, x, input_pos):
        x = x + self.attn(self.ln_1(x), input_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class ExportQwen35MoE(nn.Module):

    def __init__(self, config, use_fla=False):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            ExportBlock(config, layer_idx=i, use_fla=use_fla)
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
    def from_checkpoint(ckpt_path, device='cpu', use_fla=False):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt['config']

        export_model = ExportQwen35MoE(config, use_fla=use_fla)
        eager_sd = ckpt['model']

        new_sd = {}
        expert_weights = {}  # (layer_idx, proj, expert_idx) → tensor

        for k, v in eager_sd.items():
            ek = k.replace('transformer.wte.', 'wte.')
            ek = ek.replace('transformer.ln_f.', 'ln_f.')
            ek = ek.replace('transformer.h.', 'layers.')

            m = re.match(r'layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight', ek)
            if m:
                layer_idx, expert_idx, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                expert_weights[(layer_idx, proj, expert_idx)] = v
            else:
                new_sd[ek] = v

        # Stack per-expert weights into grouped nn.Linear format
        G = _EXPERTS_PER_GROUP
        for layer_idx in range(config.n_layer):
            gate_list = [expert_weights.get((layer_idx, "gate", e))
                         for e in range(config.n_routed_experts)]
            up_list = [expert_weights.get((layer_idx, "up", e))
                       for e in range(config.n_routed_experts)]
            down_list = [expert_weights.get((layer_idx, "down", e))
                         for e in range(config.n_routed_experts)]

            if gate_list[0] is not None:
                w_gate = torch.stack(gate_list, dim=0)  # (E, H, D)
                w_up = torch.stack(up_list, dim=0)
                fused = torch.cat([w_gate, w_up], dim=1)  # (E, 2*H, D)
                num_groups = config.n_routed_experts // G
                for g in range(num_groups):
                    chunk = fused[g * G:(g + 1) * G]
                    new_sd[f'layers.{layer_idx}.mlp.cond_ffn.gate_up_projs.{g}.weight'] = (
                        chunk.reshape(-1, chunk.size(-1))
                    )
            if down_list[0] is not None:
                w_down = torch.stack(down_list, dim=0)  # (E, D, H)
                num_groups = config.n_routed_experts // G
                for g in range(num_groups):
                    chunk = w_down[g * G:(g + 1) * G]
                    new_sd[f'layers.{layer_idx}.mlp.cond_ffn.down_projs.{g}.weight'] = (
                        chunk.reshape(-1, chunk.size(-1))
                    )

        export_model.load_state_dict(new_sd, strict=False)
        return export_model, config
