"""
Qwen3.5 MoE — a minimal, self-contained implementation in pure PyTorch.

Architecture overview:
  - Hybrid attention: 75% GatedDeltaNet (linear O(n)), 25% full softmax attention
  - Mixture of Experts (MoE) on every layer: router selects top-k of N experts
  - Shared expert: always-on expert with sigmoid gate
  - Gemma-style RMSNorm (1 + weight), partial RoPE, GQA with QK-Norm + output gate

Reference:
  - HF transformers: models/qwen3_5_moe/modeling_qwen3_5_moe.py
  - vLLM: vllm/model_executor/models/qwen3_5.py
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Qwen35MoEConfig:
    block_size: int = 512
    vocab_size: int = 256
    n_layer: int = 8
    n_embd: int = 64

    # Full attention (used on every full_attention_interval-th layer)
    n_head: int = 4
    n_kv_head: int = 2
    head_dim: int = 16
    partial_rotary_factor: float = 0.25

    # GatedDeltaNet / linear attention (used on remaining layers)
    linear_num_key_heads: int = 4
    linear_num_value_heads: int = 4
    linear_key_head_dim: int = 16
    linear_value_head_dim: int = 16
    linear_conv_kernel_dim: int = 4

    # Hybrid pattern: every Nth layer is full attention, rest are GDN
    full_attention_interval: int = 4

    # MoE: ALL layers use sparse MoE for FFN
    n_routed_experts: int = 8
    n_experts_per_tok: int = 2
    expert_intermediate_size: int = 16
    shared_expert_intermediate_size: int = 16

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(self.n_layer)
            ]


# ---------------------------------------------------------------------------
# Gemma-style RMSNorm: y = x / sqrt(mean(x²) + eps) * (1 + weight)
# Weight initialized to zeros so effective scale starts at 1.

class RMSNorm(nn.Module):

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim))
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class RMSNormGated(nn.Module):
    """RMSNorm(x) * silu(z) — used in GatedDeltaNet output."""

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x, z):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = (self.weight * normed.type_as(x))
        return (normed * F.silu(z.float())).type_as(x)


# ---------------------------------------------------------------------------
# Partial Rotary Position Embeddings — only rotates first rotary_dim
# dimensions of each head, passes through the rest.

class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim, partial_rotary_factor=1.0, rope_theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        ))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions, q, k):
        # q: (B, T, n_heads, head_dim), k: (B, T, n_kv_heads, head_dim)
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = freqs.cos().unsqueeze(1)  # (T, 1, rotary_dim/2)
        sin = freqs.sin().unsqueeze(1)

        q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
        k_rot, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]

        q_rot = self._apply_rotary(q_rot, cos, sin)
        k_rot = self._apply_rotary(k_rot, cos, sin)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    @staticmethod
    def _apply_rotary(x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# Full Attention with GQA + QK-Norm + partial RoPE + output gate.
# q_proj produces Q and gate (2x heads). Output = attn_output * sigmoid(gate).

class CausalSelfAttention(nn.Module):

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

    def forward(self, x, positions):
        B, T, _ = x.size()

        q_and_gate = self.q_proj(x).view(B, T, self.n_head, self.head_dim * 2)
        q = q_and_gate[..., :self.head_dim]
        gate = q_and_gate[..., self.head_dim:]

        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        q = q.type_as(x).transpose(1, 2)
        k = k.type_as(x).transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        # Output gate
        gate = gate.reshape(B, T, -1)
        y = y * torch.sigmoid(gate)

        return self.o_proj(y)


# ---------------------------------------------------------------------------
# GatedDeltaNet — linear attention used on 75% of layers.
# O(n) per token via recurrent state matrix instead of O(n²) softmax.
#
# Recurrence (per head, per token t):
#   q, k    = L2norm(Q[t]), L2norm(K[t])
#   beta_t  = sigmoid(beta[t])          — how strongly to write new info
#   g_t     = -exp(A_log) * softplus(a + dt_bias)  — Mamba-style decay
#   state   = exp(g_t) * state + beta_t * outer(k, v)
#   out[t]  = q @ state
#
# Position info comes from causal conv1d on QKV (not RoPE).

class GatedDeltaNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_k_heads = config.linear_num_key_heads
        self.n_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.n_k_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim

        assert self.n_v_heads % self.n_k_heads == 0
        self.head_repeat = self.n_v_heads // self.n_k_heads

        qkv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(config.n_embd, qkv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.n_embd, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.n_embd, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.n_embd, self.n_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            qkv_dim, qkv_dim, config.linear_conv_kernel_dim,
            groups=qkv_dim, padding=config.linear_conv_kernel_dim - 1, bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        A = torch.empty(self.n_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.n_embd, bias=False)

    def forward(self, x, positions):
        B, T, C = x.size()

        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).view(B, T, self.n_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        # Causal conv1d + SiLU activation
        qkv = qkv.transpose(1, 2)
        qkv = self.conv1d(qkv)[..., :T]
        qkv = F.silu(qkv)
        qkv = qkv.transpose(1, 2)

        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(q.view(B, T, self.n_k_heads, self.head_k_dim), p=2, dim=-1)
        k = F.normalize(k.view(B, T, self.n_k_heads, self.head_k_dim), p=2, dim=-1)
        v = v.view(B, T, self.n_v_heads, self.head_v_dim)

        if self.head_repeat > 1:
            q = q.repeat_interleave(self.head_repeat, dim=2)
            k = k.repeat_interleave(self.head_repeat, dim=2)

        # Mamba-style gating
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Delta rule recurrence (per-token loop for training)
        state = torch.zeros(B, self.n_v_heads, self.head_k_dim, self.head_v_dim,
                            device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            beta_t = beta[:, t, :, None, None]
            g_t = g[:, t, :, None, None]
            state = torch.exp(g_t) * state + beta_t * (
                k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
            )
            outputs.append(torch.einsum('bhk,bhkv->bhv', q[:, t], state))

        output = torch.stack(outputs, dim=1)

        # RMSNorm(output) * silu(z)
        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(B, T, -1)

        return self.out_proj(output)


# ---------------------------------------------------------------------------
# SwiGLU MLP — gated activation with 3 projections:
#   output = down(SiLU(gate(x)) * up(x))

class MLP(nn.Module):

    def __init__(self, n_embd, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Sparse MoE — replaces dense MLP with router + N experts + shared expert.
#   routed:  sum_k(weight_k * expert_k(x))      — only top-k experts run
#   shared:  sigmoid(gate(x)) * shared_expert(x) — always runs

class SparseMoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.top_k = config.n_experts_per_tok
        self.n_experts = config.n_routed_experts

        self.gate = nn.Linear(config.n_embd, config.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([
            MLP(config.n_embd, config.expert_intermediate_size)
            for _ in range(config.n_routed_experts)
        ])
        self.shared_expert = MLP(config.n_embd, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)

        router_logits = self.gate(x_flat)
        weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(weights, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Sparse expert dispatch: only run selected experts on their assigned tokens.
        num_tokens = x_flat.shape[0]
        flat_expert_ids = indices.view(-1)
        flat_token_ids = torch.arange(num_tokens, device=x.device) \
            .unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        order = flat_expert_ids.argsort()
        sorted_inputs = x_flat[flat_token_ids[order]]

        counts = torch.bincount(flat_expert_ids[order], minlength=self.n_experts).tolist()
        sorted_outputs = torch.empty_like(sorted_inputs)
        start = 0
        for i, count in enumerate(counts):
            if count > 0:
                sorted_outputs[start:start + count] = self.experts[i](sorted_inputs[start:start + count])
            start += count

        unsorted = torch.empty_like(sorted_outputs)
        unsorted[order] = sorted_outputs
        routed_out = (unsorted.view(num_tokens, self.top_k, C) * weights.unsqueeze(-1)).sum(dim=1)

        shared_out = self.shared_expert(x_flat)
        shared_gate = torch.sigmoid(self.shared_expert_gate(x_flat))
        return (routed_out + shared_gate * shared_out).view(B, T, C)


# ---------------------------------------------------------------------------
# Hybrid Block — selects full or linear attention per layer_type.
# All blocks use SparseMoE for FFN regardless of attention type.

class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)

        if self.layer_type == "full_attention":
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = GatedDeltaNet(config)

        self.mlp = SparseMoE(config)

    def forward(self, x, positions):
        x = x + self.attn(self.ln_1(x), positions)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full model

class Qwen35MoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight') or pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, pos)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
