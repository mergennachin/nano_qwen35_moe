# nanoQwen3.5MoE

A minimal, educational implementation of the Qwen3.5 MoE (Mixture of Experts) architecture in pure PyTorch. Single-file model definition (~300 lines), trainable on CPU or GPU, exportable to ExecuTorch.

## Files

| File | Description |
|------|-------------|
| `model.py` | Model definition — all components in one file |
| `export_model.py` | Export-compatible model (KV cache, GDN state buffers, grouped MoE experts) |
| `export.py` | Export to ExecuTorch .pte format (portable, XNNPACK, CUDA) |
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
| **Gemma-style RMSNorm** | `x / sqrt(mean(x^2) + eps) * (1 + weight)` — weight initialized to zeros so effective scale starts at 1. |
| **RMSNormGated** | `weight * RMSNorm(x) * silu(z)` — used in GatedDeltaNet output with SiLU gating. |
| **Partial RoPE** | Rotary Position Embeddings on first `partial_rotary_factor` (25%) of head dimensions. Only used in full-attention layers. |
| **Full Attention (GQA + QK-Norm + Output Gate)** | GQA with fewer KV heads. RMSNorm per-head on Q,K before RoPE. Q projection produces Q + gate; output = `attn_output * sigmoid(gate)`. |
| **GatedDeltaNet** | Linear attention via recurrent state matrix. Causal conv1d for local context. L2-normalized Q,K. Mamba-style decay: `g = -exp(A_log) * softplus(a + dt_bias)` with learned `A_log` and `dt_bias`. |
| **SwiGLU MLP** | Gated FFN: `down(SiLU(gate(x)) * up(x))` — 3 weight matrices with SiLU gating. |
| **Sparse MoE** | Router selects top-k of N experts per token. Shared expert with sigmoid gate always runs. |

## Export to ExecuTorch

`export_model.py` transforms the eager model for `torch.export`:

| Change | Why |
|--------|-----|
| KV cache as registered buffers | Full-attention layers need persistent state for autoregressive decode |
| conv_state + recurrent_state buffers | GDN layers need persistent state across tokens |
| `torch.scan` for GDN recurrence | Replaces `for t in range(T)` loop, enabling dynamic sequence lengths |
| Grouped nn.Linear experts | Expert weights as nn.Linear for `quantize_model_()` compatibility. Groups keep each linear small enough for tinygemm int4 packing. |
| `forward(tokens, input_pos)` signature | Matches ExecuTorch's LLM convention |

```bash
# Export to .pte (portable)
python export.py

# Export with XNNPACK backend
python export.py --backend xnnpack

# Export with CUDA backend + int4 quantization
python export.py --backend cuda --qlinear 4w --qlinear-packing-format tile_packed_to_4d --qembedding 8w

# Verify all modes produce identical output
python verify_export.py
```

### Export design

`export.py` is split into `load_and_quantize()` and `export_and_lower()`:

- **`load_and_quantize(args)`** — loads checkpoint, quantizes layer-by-layer on CUDA (each layer moved to CUDA, quantized, moved back to CPU). Peak GPU = 1 bf16 layer. Returns model on CPU.
- **`export_and_lower(model, config, args)`** — `torch.export` traces on CPU (doesn't execute ops), then lowers to the selected backend.

This separation lets you iterate on export without re-quantizing, and test the quantized model eagerly before committing to a full export.

## Tiny Config vs Real Qwen3.5 MoE

| Parameter | Tiny | Real |
|-----------|------|------|
| `n_embd` | 64 | 2048 |
| `n_layer` | 8 | 40 |
| `n_head` / `n_kv_head` | 4 / 2 | 16 / 2 |
| `head_dim` | 16 | 256 |
| `partial_rotary_factor` | 0.25 | 0.25 |
| `n_routed_experts` | 8 | 256 |
| `n_experts_per_tok` | 2 | 8 |
| `full_attention_interval` | 4 | 4 |
| `linear_conv_kernel_dim` | 4 | 4 |
| Total parameters | ~420K | ~35B |

## Differences from Production

| Feature | This Implementation | Production (vLLM/ExecuTorch) |
|---------|-------------------|---------------------|
| GDN recurrence | `torch.scan` (portable) or FLA Triton kernel (CUDA) | Chunked parallel scan (FLA/FlashInfer) |
| MoE dispatch | Grouped nn.Linear + gather | Fused CUDA kernel (FusedMoE) |
| Parallelism | None (single device) | TP, EP, PP, sequence parallel |

The architecture (norms, gating, attention, RoPE) matches the HF reference (`transformers/models/qwen3_5_moe/`) exactly. Real Qwen3.5 checkpoints can be loaded with appropriate key remapping — see `executorch/examples/models/qwen3_5_moe/` for the full-size model.

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

- [HF Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers) — `transformers/models/qwen3_5_moe/`
- [vLLM Qwen3.5 implementation](https://github.com/vllm-project/vllm) — `vllm/model_executor/models/qwen3_5.py`
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — the linear attention mechanism
- [nanoGPT](https://github.com/karpathy/nanoGPT) — style inspiration for single-file model
- [ExecuTorch](https://github.com/pytorch/executorch) — on-device inference runtime
