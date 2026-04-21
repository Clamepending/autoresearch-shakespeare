"""
autoresearch-shakespeare training script. Edit freely to lower val_bpb.

Contract (required by eval.py):
  - Running `uv run python train.py` writes `ckpt.pt` containing a dict:
        {"state_dict": <model state_dict>, "cfg": <dict>}
  - This module must define `build_model(vocab_size, **cfg) -> nn.Module` that
    reconstructs an `nn.Module` whose forward maps (B, T) LongTensor token
    indices -> (B, T, V) float logits. eval.py calls:
        m = train.build_model(vocab_size=V, **ckpt["cfg"])
        m.load_state_dict(ckpt["state_dict"])
  - Training must complete within TIME_BUDGET_S wall-clock seconds
    (soft-enforced in the loop); going over is OK but costs you cycle time.
  - Keep training logic under `if __name__ == "__main__":` so eval.py can
    `import train` without side effects.

Anything else is fair game: architecture, optimizer, LR schedule, batch size,
context length used during training, data augmentation, curriculum, etc.
"""
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- fixed-ish constants (change only with justification) -------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("data/input.txt")
CKPT_PATH = Path("ckpt.pt")
VAL_FRAC = 0.10
CTX_LEN_EVAL = 256   # eval.py uses this; model must accept context >= 256
TIME_BUDGET_S = 300  # 5 minutes wall clock inside the training loop
SEED = 1337


def load_data():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text = DATA_PATH.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(len(data) * (1 - VAL_FRAC))
    return torch.from_numpy(data[:split]), torch.from_numpy(data[split:]), len(chars)


def pick_device():
    # NOTE: default is CPU. On this PyTorch + MPS combo, training this causal
    # transformer on MPS reaches a pathological state where train loss crashes
    # to ~0.02 while val stays at ~uniform (~6 bpb) — the model overfits to
    # per-batch cues that do not generalize. CPU training is ~3–4× slower in
    # steps/sec but produces a generalizing model. Switch to "mps"/"cuda" at
    # your own risk for the speed win.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---- model (edit freely) ----------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = (d_ff + 7) & ~7
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_value = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_value(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, ctx_len=256, rope_base=10000.0, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(ctx_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_complex", torch.polar(torch.ones_like(freqs), freqs), persistent=False)

    def _apply_rope(self, x):
        T = x.size(-2)
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        x_rotated = x_complex * self.rope_complex[:T]
        return torch.view_as_real(x_rotated).flatten(-2)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        # dropout_p intentionally omitted: on CPU, `dropout_p > 0` forces SDPA's
        # math-eager backend and costs ~2.4× per-step. Regularization lives in SwiGLU.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0, attn_dropout=None, ctx_len=256):
        super().__init__()
        if attn_dropout is None:
            attn_dropout = dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, ctx_len=ctx_len, dropout=attn_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=4,
                 ctx_len=256, dropout=0.0, attn_dropout=None):
        super().__init__()
        assert ctx_len >= CTX_LEN_EVAL, "model must support eval ctx length"
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, dropout=dropout, attn_dropout=attn_dropout, ctx_len=ctx_len) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.tok_emb(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return self.head(h)


def build_model(vocab_size, **cfg):
    """Reconstruct the model for eval. Keep this signature: eval.py calls
    `build_model(vocab_size=V, **ckpt["cfg"])`. Edit the Model class and/or
    this function together so they stay in sync."""
    return Model(vocab_size=vocab_size, **cfg)


# ---- training (edit freely) -------------------------------------------------
def get_batch(data, batch_size, ctx_len, device):
    ix = torch.randint(0, len(data) - ctx_len - 1, (batch_size,))
    x = torch.stack([data[i : i + ctx_len] for i in ix]).to(device, non_blocking=True)
    y = torch.stack([data[i + 1 : i + ctx_len + 1] for i in ix]).to(device, non_blocking=True)
    return x, y


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    print(f"device={device}", flush=True)

    train_data, val_data, vocab_size = load_data()
    print(f"vocab={vocab_size} train_tokens={len(train_data)} val_tokens={len(val_data)}", flush=True)

    batch_size = 32
    ctx_len = 384
    model_cfg = dict(d_model=128, n_head=4, n_layer=4, ctx_len=ctx_len, dropout=0.2, attn_dropout=0.1)
    model = build_model(vocab_size=vocab_size, **model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.2f}M", flush=True)

    lr_max = 4e-3
    lr_min = 4e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.05,
                            betas=(0.9, 0.95))

    t0 = time.time()
    step = 0
    model.train()
    while True:
        t_elapsed = time.time() - t0
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(t_elapsed / TIME_BUDGET_S, 1.0)))
        for g in opt.param_groups:
            g['lr'] = lr

        x, y = get_batch(train_data, batch_size, ctx_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step {step} loss {loss.item():.4f} lr {lr:.5f} elapsed {elapsed:.1f}s", flush=True)

        if time.time() - t0 > TIME_BUDGET_S:
            break

    elapsed = time.time() - t0
    print(f"done step={step} elapsed={elapsed:.1f}s final_train_loss={loss.item():.4f}", flush=True)

    model.eval()
    model.cpu()
    torch.save({"state_dict": model.state_dict(), "cfg": model_cfg}, CKPT_PATH)
    print(f"saved {CKPT_PATH}", flush=True)


if __name__ == "__main__":
    main()
