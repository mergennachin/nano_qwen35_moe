# nanoQwen3.5MoE

A minimal, educational implementation of the Qwen3.5 MoE (Mixture of Experts) architecture in pure PyTorch. Single-file model definition (~300 lines), trainable on CPU or GPU, exportable to ExecuTorch.

## Files

| File | Description |
|------|-------------|
| `model.py` | Model definition — all components in one file |
| `export_model.py` | Export-compatible model (KV cache, GDN state buffers, stacked MoE) |
| `export.py` | Export to ExecuTorch .pte format |
| `inference.py` | Run inference in three modes: eager, export_eager, exported |
| `verify_export.py` | Verify all three modes produce identical output |
| `train.py` | Training script (Shakespeare char-level, CPU/GPU) |
| `data_shakespeare_char/prepare.py` | Prepare dataset from `input.txt` |

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
| **Sparse MoE** | Router selects top-k of N experts per token. Tokens are sorted by expert ID, batched per expert, then scattered back. A shared expert with sigmoid gate always runs on all tokens. |

## Export to ExecuTorch

`export_model.py` transforms the eager model for `torch.export`:

| Change | Why |
|--------|-----|
| KV cache as registered buffers | Full-attention layers need persistent state for autoregressive decode |
| conv_state + recurrent_state buffers | GDN layers need persistent state across tokens |
| `torch.scan` for GDN recurrence | Replaces `for t in range(T)` loop, enabling dynamic sequence lengths |
| Stacked expert weights `(E, H, D)` | MoE uses tensor indexing instead of data-dependent dispatch (no Python branching) |
| `forward(tokens, input_pos)` signature | Matches ExecuTorch's LLM convention |

```bash
# Export to .pte (portable)
python export.py

# Export with XNNPACK backend
python export.py --backend xnnpack

# Verify all modes produce identical output
python verify_export.py
```

## Tiny Config vs Real Qwen3.5 MoE

| Parameter | Tiny | Real |
|-----------|------|------|
| `n_embd` | 64 | 2048 |
| `n_layer` | 4 | 40 |
| `n_head` / `n_kv_head` | 4 / 2 | 16 / 2 |
| `n_routed_experts` | 4 | 256 |
| `n_experts_per_tok` | 2 | 8 |
| `full_attention_interval` | 4 | 4 |
| `expert_intermediate / n_embd` | 0.5 | 0.25 |
| `shared / expert intermediate` | 1:1 | 1:1 |
| `linear_conv_kernel_dim` | 4 | 4 |
| Total parameters | ~210K | ~billions |

## Simplifications vs Reference (vLLM)

This implementation captures the high-level architecture but simplifies several components for clarity:

| Feature | This Implementation | Real Qwen3.5 (vLLM) |
|---------|-------------------|---------------------|
| GDN decay gate | `sigmoid(linear(x))` | Mamba-style: `exp(-exp(A_log) * softplus(a + dt_bias))` with 2 extra learned params |
| GDN output gate | `RMSNorm(out) * sigmoid(z)` | `GemmaRMSNorm(out) * SiLU(z)` |
| RMSNorm variant | `x * weight` | `x * (1 + weight)` (GemmaRMSNorm) |
| Attention output gate | None | Sigmoid gate from doubled Q projection |
| Partial rotary factor | 1.0 (full) | 0.25 (RoPE on 25% of head dims) |
| GDN recurrence | `torch.scan` | Chunked parallel scan (FLA/FlashInfer) |
| MoE dispatch | Stacked weights + index | Fused CUDA kernel |
| Parallelism | None (single device) | TP, EP, PP, sequence parallel |

These simplifications mean **real Qwen3.5 checkpoints cannot be loaded** -- the GDN gating function has a different mathematical form and extra parameters (`A_log`, `dt_bias`). The model is intended for understanding the architecture, not for production inference.

## Usage

```bash
# Prepare data (required -- generates train.bin, val.bin, and meta.pkl from input.txt)
python data_shakespeare_char/prepare.py

# Generate text (a pre-trained checkpoint is included in the repo)
python inference.py --mode eager
python inference.py --mode eager --prompt "MENENIUS:" --num_tokens 200
python inference.py --mode eager --device cuda

# Export to ExecuTorch
python export.py                        # portable (CPU)
python export.py --backend xnnpack      # XNNPACK
python export.py --backend cuda         # CUDA (requires FLA Triton kernels)

# Run inference on exported model (Python)
python inference.py --mode exported
python inference.py --mode export_eager  # export-compatible model in eager (for debugging)

# Verify all modes match
python verify_export.py

# Train from scratch (optional -- auto-detects GPU)
python train.py

# Resume training from checkpoint (set init_from = 'resume' in train.py)
python train.py
```

## C++ Runner

A standalone C++ runner for running the exported .pte without Python or libtorch.

### Build

Requires ExecuTorch to be built first with `cmake --workflow --preset llm-release-cuda` (or `llm-release` for CPU-only).

```bash
# Build with CUDA support
cmake --workflow --preset cuda

# Build CPU-only
cmake --workflow --preset cpu
```

### Run

```bash
# Portable (CPU)
./build/runner --model_path nano_qwen35_moe_portable.pte \
  --prompt "First Citizen:" --num_tokens 20

# CUDA
./build/runner --model_path nano_qwen35_moe_cuda.pte \
  --data_path aoti_cuda_blob.ptd \
  --prompt "First Citizen:" --num_tokens 20

# Options
./build/runner --model_path <.pte> \
  [--data_path <.ptd>]          \  # required for CUDA backend
  [--prompt "text"]             \  # starting text (default: "\nFirst Citizen:\n")
  [--num_tokens 50]             \  # tokens to generate (default: 50)
  [--temperature 0.8]              # sampling temperature, 0 = greedy (default: 0.8)
```

## References

- [vLLM Qwen3.5 implementation](https://github.com/vllm-project/vllm) -- `vllm/model_executor/models/qwen3_5.py`
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) -- the linear attention mechanism
- [nanoGPT](https://github.com/karpathy/nanoGPT) -- style inspiration for single-file model
- [ExecuTorch](https://github.com/pytorch/executorch) -- on-device inference runtime
