"""
Export nanoQwen3.5MoE to ExecuTorch .pte format.

Usage:
  python export.py                        # portable (CPU)
  python export.py --backend xnnpack      # XNNPACK
  python export.py --backend cuda         # CUDA (uses FLA Triton kernels for GDN)
  python export.py --backend cuda --qlinear 4w
"""

import argparse
import contextlib
import os

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from export_model import ExportQwen35MoE

_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Load + quantize
# ---------------------------------------------------------------------------


def load_and_quantize(args):
    """Load model from checkpoint, optionally quantize. Returns (model, config)."""
    is_cuda = args.backend == 'cuda'
    use_fla = is_cuda

    if use_fla:
        import executorch.backends.cuda.triton.kernels  # noqa: F401

    print(f"Loading checkpoint...")
    model, config = ExportQwen35MoE.from_checkpoint(
        args.checkpoint, device='cpu', use_fla=use_fla
    )
    model.eval()
    print(f"Model: {config.n_layer} layers, {config.n_embd}d, "
          f"{config.n_routed_experts} experts top-{config.n_experts_per_tok}")

    if args.qlinear or args.qembedding:
        _quantize(model, config, args)
    else:
        dtype = torch.bfloat16 if is_cuda else torch.float32
        model.to(dtype=dtype)

    return model, config


def _to_device_skip_meta(module, device, dtype=None):
    """Move submodules to device, skipping any that have meta-device buffers.

    Uses module.to() on leaf submodules (not p.data = p.data.to()) to
    correctly handle tensor subclasses like Int4TilePackedTo4dTensor.
    """
    for _, submod in module.named_modules():
        has_meta = any(
            b.device.type == "meta" for _, b in submod.named_buffers(recurse=False)
        )
        if has_meta:
            continue
        if list(submod.parameters(recurse=False)):
            if dtype:
                submod.to(device=device, dtype=dtype)
            else:
                submod.to(device=device)


def _quantize(model, config, args):
    """Quantize layer-by-layer on CUDA, keeping the model on CPU.

    Only submodules with parameters (and no meta buffers) are moved to
    CUDA. The quantized model stays on CPU — torch.export traces the
    graph without executing ops, so CUDA is not needed.
    """
    from executorch.extension.llm.export.quantize import quantize_model_

    packing = "tile_packed_to_4d" if args.qlinear == "4w" else None

    # Untie lm_head/embedding for independent quantization
    if model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.wte.weight.clone())

    # Quantize layers (move params to CUDA one layer at a time)
    for i, layer in enumerate(model.layers):
        _to_device_skip_meta(layer, device="cuda", dtype=torch.bfloat16)
        if args.qlinear:
            quantize_model_(
                layer,
                qlinear_config=args.qlinear,
                qlinear_group_size=args.qlinear_group_size,
                qlinear_packing_format=packing,
            )
        _to_device_skip_meta(layer, device="cpu")
        torch.cuda.empty_cache()
        print(f"  Quantized layer {i + 1}/{config.n_layer}", end="\r")
    print()

    # Quantize lm_head
    if args.qlinear:
        print("Quantizing lm_head...")
        model.lm_head.to(device="cuda", dtype=torch.bfloat16)
        wrapper = nn.ModuleDict({"lm_head": model.lm_head})
        quantize_model_(
            wrapper,
            qlinear_config=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
            qlinear_packing_format=packing,
        )
        model.lm_head = wrapper.lm_head
        model.lm_head.to(device="cpu")
        torch.cuda.empty_cache()

    # Quantize embedding (doesn't need CUDA)
    if args.qembedding:
        print(f"Quantizing embeddings ({args.qembedding})...")
        model.wte.to(dtype=torch.bfloat16)
        quantize_model_(model, qembedding_config=args.qembedding)

    # Cast remaining unquantized modules to bfloat16
    model.wte.to(dtype=torch.bfloat16)
    model.ln_f.to(dtype=torch.bfloat16)

    # Cast float buffers (KV caches, conv/recurrent state) to bfloat16.
    # Bool buffers (causal masks) are left untouched.
    for _, buf in model.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(dtype=torch.bfloat16)

    if args.qlinear:
        print(f"Quantized linear layers ({args.qlinear})")


# ---------------------------------------------------------------------------
# Export + lower
# ---------------------------------------------------------------------------


def export_and_lower(model, config, args):
    """Export model to .pte via torch.export + backend lowering."""
    from torch.export import Dim, export
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass

    is_cuda = args.backend == 'cuda'

    # Dynamic shapes
    example_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    example_input_pos = torch.tensor([0, 1], dtype=torch.long)
    seq_dim = Dim("seq_len", min=1, max=config.block_size - 1)
    dynamic_shapes = ({1: seq_dim}, {0: seq_dim})

    print("Exporting with torch.export...")
    ctx = contextlib.nullcontext() if is_cuda else sdpa_kernel([SDPBackend.MATH])
    with torch.no_grad(), ctx:
        exported = export(
            model,
            (example_tokens, example_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    print("Export successful!")

    metadata = {
        "get_max_seq_len": config.block_size,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.n_layer,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }

    if args.backend == 'cuda':
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from torch._inductor.decomposition import conv1d_to_conv2d

        exported = exported.run_decompositions(
            {torch.ops.aten.conv1d.default: conv1d_to_conv2d}
        )

        print("Lowering to ExecuTorch with CUDA...")
        compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]
        et_prog = to_edge_transform_and_lower(
            exported,
            partitioner=[CudaPartitioner(compile_specs)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True,
            ),
            constant_methods=metadata,
        )
        et_program = et_prog.to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=True,
                do_quant_fusion_and_const_prop=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

    elif args.backend == 'xnnpack':
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        print("Lowering to ExecuTorch with XNNPACK...")
        et_prog = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True,
            ),
            constant_methods=metadata,
        )
        et_program = et_prog.to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

    else:
        print("Lowering to ExecuTorch (portable)...")
        edge = to_edge(exported, constant_methods=metadata)
        et_program = edge.to_executorch()

    # Save
    output = args.output or os.path.join(_DIR, f'nano_qwen35_moe_{args.backend}.pte')
    with open(output, 'wb') as f:
        f.write(et_program.buffer)

    if hasattr(et_program, '_tensor_data') and et_program._tensor_data:
        output_dir = os.path.dirname(output)
        et_program.write_tensor_data_to_file(output_dir)
        print(f"Saved tensor data to {output_dir}/")

    print(f"Saved to {output} ({os.path.getsize(output) / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(_DIR, 'checkpoint/ckpt.pt'))
    parser.add_argument('--output', default=None)
    parser.add_argument('--backend', default='portable', choices=['portable', 'xnnpack', 'cuda'])
    parser.add_argument('--qlinear', default=None, choices=['4w', '8w', '8da4w', '8da8w'])
    parser.add_argument('--qlinear-group-size', type=int, default=32)
    parser.add_argument('--qembedding', default=None, choices=['8w'])
    args = parser.parse_args()

    model, config = load_and_quantize(args)
    export_and_lower(model, config, args)


if __name__ == '__main__':
    main()
