"""
Export nanoQwen3.5MoE to ExecuTorch .pte format.

Usage:
  python export.py
  python export.py --checkpoint checkpoint/ckpt.pt --output nano_qwen35_moe.pte
"""

import os
import argparse

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.export import Dim, export
from executorch.exir import to_edge

from export_model import ExportQwen35MoE

_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(_DIR, 'checkpoint/ckpt.pt'))
    parser.add_argument('--output', default=os.path.join(_DIR, 'nano_qwen35_moe.pte'))
    args = parser.parse_args()

    print("Loading checkpoint...")
    model, config = ExportQwen35MoE.from_checkpoint(args.checkpoint)
    model.eval()
    print(f"Model: {config.n_layer} layers, {config.n_embd}d, "
          f"{config.n_routed_experts} experts top-{config.n_experts_per_tok}")

    # example inputs with seq_len > 1 to avoid 0/1 specialization
    example_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    example_input_pos = torch.tensor([0, 1], dtype=torch.long)

    # dynamic shapes: seq_len can vary (prefill + decode)
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

    print("Lowering to edge...")
    edge = to_edge(exported)
    print("Converting to ExecuTorch...")
    et_program = edge.to_executorch()

    with open(args.output, 'wb') as f:
        f.write(et_program.buffer)
    print(f"Saved to {args.output} ({os.path.getsize(args.output) / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
