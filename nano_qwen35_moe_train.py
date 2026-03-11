"""
Training script for nanoQwen3.5MoE on character-level Shakespeare.
Follows nanoGPT's train.py style — single file, no dependencies beyond torch/numpy.

Usage:
  python qwen3_moe_train.py

Trains a tiny Qwen3.5 MoE on char-level Shakespeare. CPU-only, runs in minutes.
"""

import os
import time
import math
import pickle

import numpy as np
import torch

from nano_qwen35_moe_model import Qwen35MoE, Qwen35MoEConfig

# -----------------------------------------------------------------------------
# config — tuned for tiny model on CPU
out_dir = 'out-qwen35moe'
eval_interval = 50
log_interval = 10
eval_iters = 10
max_iters = 2000
init_from = 'resume'  # 'scratch' or 'resume'
batch_size = 8
block_size = 32          # short sequences for CPU speed (GDN recurrence is O(T))
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 100
min_lr = 1e-4
device = 'cpu'
# -----------------------------------------------------------------------------

# data preparation — embed Shakespeare text directly (no download needed)
data_dir = os.path.join(os.path.dirname(__file__), 'data_shakespeare_char')

SHAKESPEARE = """\
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?

Second Citizen:
Worthy Menenius Agrippa; one that hath always loved
the people.

First Citizen:
He's one honest enough: would all the rest were so!

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have
had inkling this fortnight what we intend to do,
which now we'll show 'em in deeds. They say poor
suitors have strong breaths: they shall know we
have strong arms too.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?

First Citizen:
We cannot, sir, we are undone already.

MENENIUS:
I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them
Against the Roman state, whose course will on
The way it takes, cracking ten thousand curbs
Of more strong link asunder than can ever
Appear in your impediment. For the dearth,
The gods, not the patricians, make it, and
Your knees to them, not arms, must help. Alack,
You are transported by calamity
Thither where more attends you, and you slander
The helms o' the state, who care for you like fathers,
When you curse them as enemies.

First Citizen:
Care for us! True, indeed! They ne'er cared for us
yet: suffer us to famish, and their store-houses
crammed with grain; make edicts for usury, to
support usurers; repeal daily any wholesome act
established against the rich, and provide more
piercing statutes daily, to chain up and restrain
the poor. If the wars eat us not up, they will; and
there's all the love they bear us.

MENENIUS:
Either you must
Confess yourselves wondrous malicious,
Or be accused of folly. I shall tell you
A pretty tale: it may be you have heard it;
But, since it serves my purpose, I will venture
To stale 't a little more.

First Citizen:
Well, I'll hear it, sir: yet you must not think to
fob off our disgrace with a tale: but, an 't please
you, deliver.

MENENIUS:
There was a time when all the body's members
Rebell'd against the belly, thus accused it:
That only like a gulf it did remain
I' the midst o' the body, idle and unactive,
Still cupboarding the viand, never bearing
Like labour with the rest, where the other instruments
Did see and hear, devise, instruct, walk, feel,
And, mutually participate, did minister
Unto the appetite and affection common
Of the whole body. The belly answer'd--

First Citizen:
Well, sir, what answer made the belly?

MENENIUS:
Sir, I shall tell you. With a kind of smile,
Which ne'er came from the lungs, but even thus--
For, look you, I may make the belly smile
As well as speak--it tauntingly replied
To the discontented members, the mutinous parts
That envied his receipt; even so most fitly
As you malign our senators for that
They are not such as you.

First Citizen:
Your belly's answer? What!
The kingly-crowned head, the vigilant eye,
The counsellor heart, the arm our soldier,
Our steed the leg, the tongue our trumpeter.
With other muniments and petty helps
In this our fabric, if that they--

MENENIUS:
What then?
'Fore me, this fellow speaks! What then? what then?

First Citizen:
Should by the cormorant belly be restrain'd,
Who is the sink o' the body,--

MENENIUS:
Well, what then?

First Citizen:
The former agents, if they did complain,
What could the belly answer?

MENENIUS:
I will tell you
If you'll bestow a small--of what you have little--
Patience awhile, you'st hear the belly's answer.

First Citizen:
Ye're long about it.

MENENIUS:
Note me this, good friend;
Your most grave belly was deliberate,
Not rash like his accusers, and thus answer'd:
'True is it, my incorporate friends,' quoth he,
'That I receive the general food at first,
Which you do live upon; and fit it is,
Because I am the store-house and the shop
Of the whole body: but, if you do remember,
I send it through the rivers of your blood,
Even to the court, the heart, to the seat o' the brain;
And, through the cranks and offices of man,
The strongest nerves and small inferior veins
From me receive that natural competency
Whereby they live: and though that all at once,
You, my good friends,'--this says the belly, mark me,--

First Citizen:
Ay, sir; well, well.

MENENIUS:
'Though all at once cannot
See what I do deliver out to each,
Yet I can make my audit up, that all
From me do back receive the flour of all,
And leave me but the bran.' What say you to't?

First Citizen:
It was an answer: how apply you this?

MENENIUS:
The senators of Rome are this good belly,
And you the mutinous members; for examine
Their counsels and their cares, digest things rightly
Touching the weal o' the common, you shall find
No public benefit which you receive
But it proceeds or comes from them to you
And no way from yourselves. What do you think,
You, the great toe of this assembly?

First Citizen:
I the great toe! why the great toe?

MENENIUS:
For that, being one o' the lowest, basest, poorest,
Of this most wise rebellion, thou go'st foremost:
Thou rascal, that art worst in blood to run,
Lead'st first to win some vantage.
But make you ready your stiff bats and clubs:
Rome and her rats are at the point of battle;
The one side must have bale.

MARCIUS:
Thanks. What's the matter, you dissentious rogues,
That, rubbing the poor itch of your opinion,
Make yourselves scabs?

First Citizen:
We have ever your good word.

MARCIUS:
He that will give good words to thee will flatter
Beneath abhorring. What would you have, you curs,
That like nor peace nor war? the one affrights you,
The other makes you proud. He that trusts to you,
Where he should find you lions, finds you hares;
Where foxes, geese: you are no surer, no,
Than is the coal of fire upon the ice,
Or hailstone in the sun. Your virtue is
To make him worthy whose offence subdues him
And curse that justice did it.
Who deserves greatness
Deserves your hate; and your affections are
A sick man's appetite, who desires most that
Which would increase his evil. He that depends
Upon your favours swims with fins of lead
And hews down oaks with rushes. Hang ye! Trust Ye?
With every minute you do change a mind,
And call him noble that was now your hate,
Him vile that was your garland. What's the matter,
That in these several places of the city
You cry against the noble senate, who,
Under the gods, keep you in awe, which else
Would feed on one another? What's their seeking?
"""

def prepare_data():
    os.makedirs(data_dir, exist_ok=True)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        return  # already prepared

    data = SHAKESPEARE
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encoded = [stoi[c] for c in data]
    n = len(encoded)
    train_ids = np.array(encoded[:int(n * 0.9)], dtype=np.uint16)
    val_ids = np.array(encoded[int(n * 0.9):], dtype=np.uint16)

    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))

    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"prepared {n:,} chars, vocab_size={vocab_size}, "
          f"train={len(train_ids):,}, val={len(val_ids):,}")


# -----------------------------------------------------------------------------
# data loading — same as nanoGPT

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
    # start with newline
    idx = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=num_tokens, temperature=0.8, top_k=10)
    model.train()
    return ''.join(itos[i] for i in out[0].tolist())


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    prepare_data()

    # load meta
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
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
        # lr schedule
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # eval
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

            # sample
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
