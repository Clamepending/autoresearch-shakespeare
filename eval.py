"""
Frozen val-bpb judge. NEVER edit this file.

Contract with train.py:
  - train.py, when run, writes `ckpt.pt` containing a dict:
        {"state_dict": <model state_dict>, "cfg": <dict>}
  - train.py must define `build_model(vocab_size, **cfg) -> nn.Module` that
    maps (B, T) LongTensor token indices -> (B, T, V) float logits.
  - train.py must be importable without side effects (keep training under
    `if __name__ == "__main__":`).

Evaluation protocol:
  - Dataset: tinyshakespeare, last 10% by char = val.
  - Vocab: sorted unique chars over the full text.
  - Windows: non-overlapping, length CTX_LEN=256, starting at offset 0.
  - Metric: val_bpb = mean cross-entropy (nats) / ln(2). Prints JSON to stdout.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import train  # noqa: F401  (must be importable so torch.load can find classes)

CTX_LEN = 256
VAL_FRAC = 0.10
DATA_PATH = Path("data/input.txt")
CKPT_PATH = Path("ckpt.pt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SEED = 1337


def ensure_data():
    if not DATA_PATH.exists():
        import urllib.request
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text()


def val_tensor():
    text = ensure_data()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(len(data) * (1 - VAL_FRAC))
    return torch.from_numpy(data[split:]), len(chars)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    val, vocab_size = val_tensor()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = train.build_model(vocab_size=vocab_size, **ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # Judge runs on CPU: deterministic, device-independent scoring.
    device = torch.device("cpu")
    model.to(device)

    nll_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        i = 0
        while i + CTX_LEN + 1 <= len(val):
            x = val[i : i + CTX_LEN].unsqueeze(0).to(device)
            y = val[i + 1 : i + CTX_LEN + 1].unsqueeze(0).to(device)
            logits = model(x)
            if logits.size(-1) != vocab_size:
                raise ValueError(
                    f"model vocab {logits.size(-1)} != dataset vocab {vocab_size}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            )
            nll_sum += loss.item()
            n_tokens += y.numel()
            i += CTX_LEN

    nll = nll_sum / n_tokens
    bpb = nll / math.log(2)
    out = {
        "val_bpb": bpb,
        "val_nll": nll,
        "n_tokens": n_tokens,
        "vocab_size": vocab_size,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
