"""
Run inference with nanoQwen3.5MoE.

Three modes:
  --mode eager         Use model.py (original eager model)
  --mode export_eager  Use export_model.py (export-compatible, runs in eager)
  --mode exported      Use nano_qwen35_moe.pte (ExecuTorch runtime)

Usage:
  python inference.py
  python inference.py --mode eager --device cuda
  python inference.py --mode export_eager
  python inference.py --mode exported --pte nano_qwen35_moe_portable.pte
  python inference.py --prompt "MENENIUS:" --num_tokens 200
"""

import os
import time
import pickle
import argparse

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(_DIR, 'checkpoint/ckpt.pt')
PTE_PATH = os.path.join(_DIR, 'nano_qwen35_moe_portable.pte')
DATA_DIR = os.path.join(_DIR, 'data_shakespeare_char')


def load_meta(data_dir):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    assert os.path.exists(meta_path), f"Missing {meta_path} — run data_shakespeare_char/prepare.py"
    with open(meta_path, 'rb') as f:
        return pickle.load(f)


def encode_prompt(prompt, stoi):
    chars = [c for c in prompt if c in stoi]
    if not chars:
        chars = ['\n']
    return [stoi[c] for c in chars]


def generate_eager(model, prompt_ids, num_tokens, temperature, top_k, device):
    """Generate using model.py's built-in generate()."""
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=num_tokens,
                             temperature=temperature, top_k=top_k)
    return out[0].tolist()


def generate_with_input_pos(model_fn, prompt_ids, num_tokens, temperature, top_k, device, max_seq_len):
    """Generate using forward(tokens, input_pos) signature (export_model or .pte)."""
    tokens = prompt_ids.copy()

    # truncate prompt if longer than max_seq_len
    start = max(0, len(prompt_ids) - max_seq_len)
    active_prompt = prompt_ids[start:]

    # prefill: feed prompt tokens one at a time
    for i, tok in enumerate(active_prompt):
        t = torch.tensor([[tok]], dtype=torch.long, device=device)
        p = torch.tensor([i], dtype=torch.long, device=device)
        logits = model_fn(t, p)

    # decode: generate new tokens one at a time
    pos = len(active_prompt)
    next_tok = sample_token(logits, temperature, top_k)
    tokens.append(next_tok)

    for _ in range(num_tokens - 1):
        if pos >= max_seq_len:
            break  # KV cache / state buffers are full
        t = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        p = torch.tensor([pos], dtype=torch.long, device=device)
        logits = model_fn(t, p)
        next_tok = sample_token(logits, temperature, top_k)
        tokens.append(next_tok)
        pos += 1

    return tokens


def sample_token(logits, temperature, top_k):
    logits = logits[0, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = -float('Inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def main():
    parser = argparse.ArgumentParser(description='nanoQwen3.5MoE inference')
    parser.add_argument('--mode', default='eager', choices=['eager', 'export_eager', 'exported'])
    parser.add_argument('--ckpt', default=CKPT_PATH)
    parser.add_argument('--pte', default=PTE_PATH)
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--prompt', default='\nFirst Citizen:\n')
    parser.add_argument('--num_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    meta = load_meta(args.data_dir)
    stoi, itos = meta['stoi'], meta['itos']
    prompt_ids = encode_prompt(args.prompt, stoi)

    print(f"Mode: {args.mode}")

    t0 = time.time()

    if args.mode == 'eager':
        from model import Qwen35MoE
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        config = ckpt['config']
        model = Qwen35MoE(config)
        model.load_state_dict(ckpt['model'])
        model.to(args.device)
        model.eval()
        pattern = ''.join('F' if t == 'full_attention' else 'L' for t in config.layer_types)
        print(f"Model: {config.n_layer} layers ({pattern}), {model.get_num_params():,} params")
        print(f"Device: {args.device}")

        tokens = generate_eager(model, prompt_ids, args.num_tokens,
                                args.temperature, args.top_k, args.device)

    elif args.mode == 'export_eager':
        from export_model import ExportQwen35MoE
        model, config = ExportQwen35MoE.from_checkpoint(args.ckpt, device=args.device)
        model.to(args.device)
        model.eval()
        pattern = ''.join('F' if t == 'full_attention' else 'L' for t in config.layer_types)
        print(f"Model: {config.n_layer} layers ({pattern})")
        print(f"Device: {args.device}")

        def model_fn(t, p):
            with torch.no_grad():
                return model(t, p)

        tokens = generate_with_input_pos(model_fn, prompt_ids, args.num_tokens,
                                         args.temperature, args.top_k, args.device,
                                         config.block_size)

    elif args.mode == 'exported':
        from executorch.runtime import Runtime
        assert os.path.exists(args.pte), f"Missing {args.pte} — run export.py first"
        runtime = Runtime.get()
        program = runtime.load_program(args.pte)
        method = program.load_method("forward")
        print(f"Loaded: {args.pte}")
        device = 'cpu'

        # load config to get block_size
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        max_seq_len = ckpt['config'].block_size

        def model_fn(t, p):
            return method.execute([t.cpu(), p.cpu()])[0]

        tokens = generate_with_input_pos(model_fn, prompt_ids, args.num_tokens,
                                         args.temperature, args.top_k, device,
                                         max_seq_len)

    dt = time.time() - t0
    text = ''.join(itos.get(t, '?') for t in tokens)
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")
    print(f"{args.num_tokens} tokens in {dt:.1f}s ({dt/args.num_tokens*1000:.0f}ms/tok)")


if __name__ == '__main__':
    main()
