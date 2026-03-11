# nanoQwen3.5MoE

A minimal, educational implementation of the Qwen3.5 MoE (Mixture of Experts) architecture in pure PyTorch. Single-file model definition (~300 lines), trainable on CPU.

## Files

| File | Description |
|------|-------------|
| `nano_qwen35_moe_model.py` | Model definition — all components in one file |
| `nano_qwen35_moe_train.py` | Training script with embedded Shakespeare data |
| `nano_qwen35_moe_inference.py` | Load checkpoint and generate text |

## Architecture

Qwen3.5 MoE is a hybrid-attention transformer with sparse Mixture of Experts:

```
Input tokens
    |
    v
Token Embedding (no learned position embedding -- RoPE is inside attention)
    |
    v
+--- Decoder Layer xN ------------------------------------------+
|                                                                |
|  RMSNorm -> Attention (hybrid) -> residual add                 |
|    +- 75% of layers: GatedDeltaNet (linear, O(n))              |
|    +- 25% of layers: Full Attention (softmax, O(n^2))          |
|                                                                |
|  RMSNorm -> Sparse MoE -> residual add                        |
|    +- Router: softmax -> top-k expert selection                |
|    +- Routed experts: k of N independent SwiGLU MLPs           |
|    +- Shared expert: always-on SwiGLU with sigmoid gate        |
|                                                                |
+----------------------------------------------------------------+
    |
    v
RMSNorm -> LM Head -> logits
```

Layer pattern (default `full_attention_interval=4`):

```
L L L F L L L F    (L = GatedDeltaNet, F = Full Attention)
```

## Key Components

| Component | Description |
|-----------|-------------|
| **RMSNorm** | Normalization without mean subtraction: `x / sqrt(mean(x^2) + eps) * weight` |
| **RoPE** | Rotary Position Embeddings -- parameter-free position encoding via rotation of Q,K vectors. Only used in full-attention layers. |
| **Full Attention (GQA + QK-Norm)** | Grouped Query Attention with fewer KV heads than Q heads. RMSNorm applied per-head to Q and K before RoPE. Used on 25% of layers for precise token-to-token matching. |
| **GatedDeltaNet** | Linear attention via recurrent state matrix. Causal conv1d provides local position context. Delta rule: `state = gate * state + beta * outer(k, v)`, read: `out = q @ state`. O(n) per token. Used on 75% of layers. |
| **SwiGLU MLP** | Gated FFN: `down(SiLU(gate(x)) * up(x))` -- 3 weight matrices with SiLU gating. |
| **Sparse MoE** | Router selects top-k of N experts per token. Each expert is an independent SwiGLU MLP. A shared expert with sigmoid gate always runs on all tokens. |

## Tiny Config vs Real Qwen3.5 MoE

| Parameter | Tiny | Real |
|-----------|------|------|
| `n_embd` | 64 | 2048 |
| `n_layer` | 4-8 | 40 |
| `n_head` / `n_kv_head` | 4 / 2 | 16 / 2 |
| `n_routed_experts` | 4-8 | 256 |
| `n_experts_per_tok` | 2 | 8 |
| `full_attention_interval` | 4 | 4 |
| `expert_intermediate / n_embd` | 0.25 | 0.25 |
| `shared / expert intermediate` | 1:1 | 1:1 |
| `linear_conv_kernel_dim` | 4 | 4 |
| Total parameters | ~200K | ~billions |

## Simplifications vs Reference (vLLM)

This implementation captures the high-level architecture but simplifies several components for clarity:

| Feature | This Implementation | Real Qwen3.5 (vLLM) |
|---------|-------------------|---------------------|
| GDN decay gate | `sigmoid(linear(x))` | Mamba-style: `exp(-exp(A_log) * softplus(a + dt_bias))` with 2 extra learned params |
| GDN output gate | `RMSNorm(out) * sigmoid(z)` | `GemmaRMSNorm(out) * SiLU(z)` |
| RMSNorm variant | `x * weight` | `x * (1 + weight)` (GemmaRMSNorm) |
| Attention output gate | None | Sigmoid gate from doubled Q projection |
| Partial rotary factor | 1.0 (full) | 0.25 (RoPE on 25% of head dims) |
| GDN recurrence | Naive Python loop | Chunked parallel scan (FLA/FlashInfer) |
| MoE dispatch | Python loop + `index_add_` | Fused CUDA kernel |
| Parallelism | None (single device) | TP, EP, PP, sequence parallel |

These simplifications mean **real Qwen3.5 checkpoints cannot be loaded** -- the GDN gating function has a different mathematical form and extra parameters (`A_log`, `dt_bias`). The model is intended for understanding the architecture, not for production inference.

## Usage

```bash
# Train on Shakespeare (CPU, ~30 min)
python nano_qwen35_moe_train.py

# Resume training from checkpoint
# (set init_from = 'resume' in nano_qwen35_moe_train.py)
python nano_qwen35_moe_train.py

# Generate text
python nano_qwen35_moe_inference.py
python nano_qwen35_moe_inference.py --prompt "MENENIUS:" --num_tokens 200
```

## References

- [vLLM Qwen3.5 implementation](https://github.com/vllm-project/vllm) -- `vllm/model_executor/models/qwen3_5.py`
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) -- the linear attention mechanism
- [nanoGPT](https://github.com/karpathy/nanoGPT) -- style inspiration for single-file model
