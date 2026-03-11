"""
Training script for nanoQwen3.5MoE on character-level Shakespeare.

Usage:
  python train.py

Requires: data_shakespeare_char/train.bin, val.bin, meta.pkl
  (run data_shakespeare_char/prepare.py first)
"""

import os
import time
import math
import pickle

import numpy as np
import torch

from model import Qwen35MoE, Qwen35MoEConfig

# -----------------------------------------------------------------------------
# config
out_dir = 'out-qwen35moe'
eval_interval = 50
log_interval = 10
eval_iters = 10
max_iters = 2000
init_from = 'scratch'  # 'scratch' or 'resume'
batch_size = 8
block_size = 32
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 100
min_lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_shakespeare_char')
# -----------------------------------------------------------------------------


def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def sample(model, meta, num_tokens=200):
    model.eval()
    itos = meta['itos']
    stoi = meta['stoi']
    idx = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=num_tokens, temperature=0.8, top_k=10)
    model.train()
    return ''.join(itos[i] for i in out[0].tolist())


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # check data exists
    meta_path = os.path.join(data_dir, 'meta.pkl')
    assert os.path.exists(meta_path), (
        f"Missing {meta_path} — run: python data_shakespeare_char/prepare.py"
    )

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"vocab_size = {vocab_size}")

    # create or resume model
    iter_start = 0
    best_val_loss = 1e9
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')

    if init_from == 'resume' and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = Qwen35MoE(config)
        model.load_state_dict(ckpt['model'])
        iter_start = ckpt['iter_num'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f"Resumed at step {iter_start}, best_val_loss={best_val_loss:.4f}")
    else:
        print("Initializing new model from scratch")
        torch.manual_seed(1337)
        config = Qwen35MoEConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=4,
            n_embd=64,
            n_head=4,
            n_kv_head=2,
            head_dim=16,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            n_routed_experts=4,
            n_experts_per_tok=2,
            expert_intermediate_size=32,
            shared_expert_intermediate_size=32,
        )
        model = Qwen35MoE(config)

    model.to(device)

    # optimizer
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))
    print(f"decay params: {sum(p.numel() for p in decay_params):,}, "
          f"nodecay params: {sum(p.numel() for p in nodecay_params):,}")

    # training loop
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    for iter_num in range(iter_start, max_iters + 1):
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # eval + checkpoint
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter_num:5d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}, lr {lr:.2e}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    ckpt = {
                        'model': model.state_dict(),
                        'config': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
                    print(f"saved checkpoint to {out_dir}/ckpt.pt")

            # sample every 250 steps
            if iter_num % (eval_interval * 5) == 0:
                text = sample(model, meta, num_tokens=200)
                print(f"--- sample at step {iter_num} ---")
                print(text[:300])
                print("---")

        # forward/backward
        X, Y = get_batch('train')
        _, loss = model(X, Y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # logging
        if iter_num % log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"  iter {iter_num:5d}: loss {loss.item():.4f}, {dt*1000:.0f}ms")

    # final sample
    print("\n=== FINAL SAMPLE ===")
    print(sample(model, meta, num_tokens=500))
