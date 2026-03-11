"""
Export nanoQwen3.5MoE to ExecuTorch .pte format.

Usage:
  python export.py                        # portable (CPU)
  python export.py --backend xnnpack      # XNNPACK
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
    parser.add_argument('--backend', default='portable', choices=['portable', 'xnnpack'])
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(_DIR, f'nano_qwen35_moe_{args.backend}.pte')

    print("Loading checkpoint...")
    model, config = ExportQwen35MoE.from_checkpoint(args.checkpoint)
    model.eval()
    print(f"Model: {config.n_layer} layers, {config.n_embd}d, "
          f"{config.n_routed_experts} experts top-{config.n_experts_per_tok}")

    # example inputs with seq_len > 1 to avoid 0/1 specialization
    example_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    example_input_pos = torch.tensor([0, 1], dtype=torch.long)

    # dynamic shapes
    seq_dim = Dim("seq_len", min=1, max=config.block_size - 1)
    dynamic_shapes = (
        {1: seq_dim},   # tokens: (1, seq_len)
        {0: seq_dim},   # input_pos: (seq_len,)
    )

    print("Exporting with torch.export...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        exported = export(
            model,
            (example_tokens, example_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    print("Export successful!")

    if args.backend == 'xnnpack':
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        print("Lowering to ExecuTorch with XNNPACK...")
        et_prog = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
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
    print(f"Saved to {args.output} ({os.path.getsize(args.output) / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
