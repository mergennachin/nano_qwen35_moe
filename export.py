"""
Export nanoQwen3.5MoE to ExecuTorch .pte format.

Usage:
  python export.py                        # portable (CPU)
  python export.py --backend xnnpack      # XNNPACK
  python export.py --backend cuda         # CUDA (uses FLA Triton kernels for GDN)
"""

import os
import argparse

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.export import Dim, export
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass

from export_model import ExportQwen35MoE

_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(_DIR, 'checkpoint/ckpt.pt'))
    parser.add_argument('--output', default=None)
    parser.add_argument('--backend', default='portable', choices=['portable', 'xnnpack', 'cuda'])
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(_DIR, f'nano_qwen35_moe_{args.backend}.pte')

    is_cuda = args.backend == 'cuda'
    device = 'cuda' if is_cuda else 'cpu'
    dtype = torch.bfloat16 if is_cuda else torch.float32
    use_fla = is_cuda

    if use_fla:
        # Import triggers registration of triton::chunk_gated_delta_rule
        import executorch.backends.cuda.triton.kernels  # noqa: F401

    print(f"Loading checkpoint (device={device}, dtype={dtype})...")
    model, config = ExportQwen35MoE.from_checkpoint(
        args.checkpoint, device=device, use_fla=use_fla
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    print(f"Model: {config.n_layer} layers, {config.n_embd}d, "
          f"{config.n_routed_experts} experts top-{config.n_experts_per_tok}")

    if is_cuda:
        # CUDA/AOTI: static shapes (single-token decode)
        example_tokens = torch.tensor([[0]], dtype=torch.long, device=device)
        example_input_pos = torch.tensor([0], dtype=torch.long, device=device)
        dynamic_shapes = None
    else:
        # Portable/XNNPACK: dynamic shapes via torch.scan
        example_tokens = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        example_input_pos = torch.tensor([0, 1], dtype=torch.long, device=device)
        seq_dim = Dim("seq_len", min=1, max=config.block_size - 1)
        dynamic_shapes = ({1: seq_dim}, {0: seq_dim})

    print("Exporting with torch.export...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        exported = export(
            model,
            (example_tokens, example_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    print("Export successful!")

    if args.backend == 'cuda':
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

        print("Lowering to ExecuTorch with CUDA...")
        compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]
        et_prog = to_edge_transform_and_lower(
            exported,
            partitioner=[CudaPartitioner(compile_specs)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True,
            ),
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
        )
        et_program = et_prog.to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

    else:
        print("Lowering to ExecuTorch (portable)...")
        edge = to_edge(exported)
        et_program = edge.to_executorch()

    with open(args.output, 'wb') as f:
        f.write(et_program.buffer)

    if hasattr(et_program, '_tensor_data') and et_program._tensor_data:
        output_dir = os.path.dirname(args.output)
        et_program.write_tensor_data_to_file(output_dir)
        print(f"Saved tensor data to {output_dir}/")

    print(f"Saved to {args.output} ({os.path.getsize(args.output) / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
