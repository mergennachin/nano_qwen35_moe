"""
Run inference with a trained nanoQwen3.5MoE model.

Usage:
  python inference.py
  python inference.py --device cuda
  python inference.py --prompt "MENENIUS:" --num_tokens 500
  python inference.py --device cuda --prompt "First Citizen:" --temperature 0.5
"""

import os
import time
import pickle
import argparse

import torch
from model import Qwen35MoE

_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(_DIR, 'out-qwen35moe/ckpt.pt')
DATA_DIR = os.path.join(_DIR, 'data_shakespeare_char')


def main():
    parser = argparse.ArgumentParser(description='Sample from trained Qwen3.5 MoE')
    parser.add_argument('--ckpt', default=CKPT_PATH)
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--prompt', default='\nFirst Citizen:\n')
    parser.add_argument('--num_tokens', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # Load character encoding
    meta_path = os.path.join(args.data_dir, 'meta.pkl')
    assert os.path.exists(meta_path), f"Missing {meta_path} — run train.py first"
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']

    # Load checkpoint
    assert os.path.exists(args.ckpt), f"Missing {args.ckpt} — run train.py first"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt['config']

    pattern = ''.join('F' if t == 'full_attention' else 'L' for t in config.layer_types)
    print(f"Loaded checkpoint: step {ckpt['iter_num']}, val_loss={ckpt['best_val_loss']:.4f}")
    print(f"Model: {config.n_layer} layers ({pattern}), {config.n_embd}d, "
          f"{config.n_routed_experts} experts top-{config.n_experts_per_tok}")

    # Build model and load weights
    model = Qwen35MoE(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    print(f"Parameters: {model.get_num_params():,}")

    # Encode prompt
    prompt_chars = [c for c in args.prompt if c in stoi]
    if not prompt_chars:
        prompt_chars = ['\n']
    prompt_ids = [stoi[c] for c in prompt_chars]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.num_tokens,
                             temperature=args.temperature, top_k=args.top_k)
    dt = time.time() - t0

    text = ''.join(itos[i] for i in out[0].tolist())
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")
    print(f"{args.num_tokens} tokens in {dt:.1f}s ({dt/args.num_tokens*1000:.0f}ms/tok)")


if __name__ == '__main__':
    main()
