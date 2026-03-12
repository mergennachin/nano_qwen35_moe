"""
Qwen3.5 MoE — a minimal, self-contained implementation in pure PyTorch.

Architecture overview:
  - Hybrid attention: 75% GatedDeltaNet (linear O(n)), 25% full softmax attention
  - Mixture of Experts (MoE) on every layer: router selects top-k of N experts
  - Shared expert: always-on expert with sigmoid gate
  - RMSNorm, RoPE, GQA with QK-Norm, SwiGLU activation

Reference: vLLM's Qwen3.5 MoE implementation
  vllm/model_executor/models/qwen3_5.py
  vllm/model_executor/models/qwen3_next.py
  vllm/transformers_utils/configs/qwen3_5_moe.py
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
# RMSNorm — simpler than LayerNorm: no mean subtraction, no bias.
# Formula: y = x / sqrt(mean(x²) + eps) * weight

class RMSNorm(nn.Module):

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        rms = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(rms + self.eps)
        return (x * self.weight).to(dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings — parameter-free, encodes position by rotating
# Q and K vectors. Only used in full-attention layers; GDN layers use conv1d.

class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim, rope_theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions, q, k):
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x, cos, sin):
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.head_dim)
        x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        c1, c2 = cos[..., :self.head_dim//2], cos[..., self.head_dim//2:]
        s1, s2 = sin[..., :self.head_dim//2], sin[..., self.head_dim//2:]
        o1 = x1 * c1 - x2 * s1
        o2 = x2 * c2 + x1 * s2
        return torch.cat([o1, o2], dim=-1).view(shape)


# ---------------------------------------------------------------------------
# Full Attention with GQA + QK-Norm + RoPE
# Used on 25% of layers (every full_attention_interval-th).
# O(n²) per token — precise token-to-token matching.

class CausalSelfAttention(nn.Module):

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

        # QK-Norm: RMSNorm per head on Q and K before RoPE
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, x, positions):
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q_norm(q.view(B, T, self.n_head, self.head_dim)).view(B, T, -1)
        k = self.k_norm(k.view(B, T, self.n_kv_head, self.head_dim)).view(B, T, -1)
        q, k = self.rotary_emb(positions, q, k)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# GatedDeltaNet — linear attention used on 75% of layers.
# O(n) per token via recurrent state matrix instead of O(n²) softmax.
#
# Recurrence (per head, per token t):
#   q, k    = L2norm(Q[t]), L2norm(K[t])
#   gate    = sigmoid(alpha[t])       — how much to retain old state
#   beta_t  = sigmoid(beta[t])        — how strongly to write new info
#   state   = gate * state + beta_t * outer(k, v)
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

        qkv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkvz = nn.Linear(config.n_embd, qkv_dim + self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.n_embd, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.n_embd, self.n_v_heads, bias=False)

        # Depthwise causal conv1d gives local positional context
        self.conv1d = nn.Conv1d(
            qkv_dim, qkv_dim, config.linear_conv_kernel_dim,
            groups=qkv_dim, padding=config.linear_conv_kernel_dim - 1, bias=False,
        )

        self.norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.n_embd, bias=False)

    def forward(self, x, positions):
        B, T, C = x.size()

        qkvz = self.in_proj_qkvz(x)
        qkv, z = qkvz.split([self.key_dim * 2 + self.value_dim, self.value_dim], dim=-1)
        z = z.view(B, T, self.n_v_heads, self.head_v_dim)

        beta = torch.sigmoid(self.in_proj_b(x))
        alpha = torch.sigmoid(self.in_proj_a(x))

        # Causal conv1d + SiLU activation
        qkv = qkv.transpose(1, 2)
        qkv = self.conv1d(qkv)[..., :T]
        qkv = F.silu(qkv)
        qkv = qkv.transpose(1, 2)

        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(q.view(B, T, self.n_k_heads, self.head_k_dim), p=2, dim=-1)
        k = F.normalize(k.view(B, T, self.n_k_heads, self.head_k_dim), p=2, dim=-1)
        v = v.view(B, T, self.n_v_heads, self.head_v_dim)

        # Delta rule recurrence
        n_heads = self.n_k_heads
        state = torch.zeros(B, n_heads, self.head_k_dim, self.head_v_dim,
                            device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            beta_t = beta[:, t, :, None, None]
            alpha_t = alpha[:, t, :, None, None]
            state = alpha_t * state + beta_t * (k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2))
            outputs.append(torch.einsum('bhk,bhkv->bhv', q[:, t], state))

        output = torch.stack(outputs, dim=1)
        output = self.norm(output) * F.sigmoid(z)
        return self.out_proj(output.reshape(B, T, -1))


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
        # 1. Flatten (T, top_k) into a dispatch list, sort by expert ID
        # 2. Run each expert on its contiguous batch
        # 3. Unsort outputs back to original order, reshape, weighted sum
        # In production, this is a single fused CUDA kernel (vLLM's FusedMoE).
        num_tokens = x_flat.shape[0]
        flat_expert_ids = indices.view(-1)                               # (T*top_k,)
        flat_token_ids = torch.arange(num_tokens, device=x.device) \
            .unsqueeze(1).expand(-1, self.top_k).reshape(-1)            # (T*top_k,)

        order = flat_expert_ids.argsort()
        sorted_inputs = x_flat[flat_token_ids[order]]                   # grouped by expert

        counts = torch.bincount(flat_expert_ids[order], minlength=self.n_experts).tolist()
        sorted_outputs = torch.empty_like(sorted_inputs)
        start = 0
        for i, count in enumerate(counts):
            if count > 0:
                sorted_outputs[start:start + count] = self.experts[i](sorted_inputs[start:start + count])
            start += count

        # Unsort back to (T*top_k,) original order, reshape to (T, top_k, C), weighted sum
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
