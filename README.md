# autoresearch-shakespeare

Autoresearch reduction for the Remote Vibes v2 research loop: an agent
hillclimbs val-bpb on a tinyshakespeare character-LM by editing **only**
`train.py`. `eval.py` is the fixed judge.

## Rules

- Agents edit `train.py`. Anything inside is fair game (architecture,
  optimizer, schedule, batch size, context length, curriculum, etc.).
- Agents do **not** edit `eval.py` or `pyproject.toml`.
- Agents may not add new pip dependencies.
- Training must complete within `TIME_BUDGET_S` wall-clock seconds
  (soft-enforced inside the training loop in `train.py`).
- Metric: `val_bpb` (bits per character on tinyshakespeare val split, last 10%),
  lower is better. Printed as JSON by `eval.py`.

## Devices

- `eval.py` always runs on CPU for deterministic scoring. Do not change this.
- `train.py` defaults to CPU. MPS on current torch produces a pathological
  training regime for this model (train loss crashes, val stays at uniform).
  Flip to CUDA/MPS at your own risk — the judge is device-independent, so
  you're responsible for any device-specific quirks.

## Run one cycle

```
uv sync
uv run python train.py
uv run python eval.py
```

Each `uv run python eval.py` invocation prints one JSON line like
`{"val_bpb": 1.73, "val_nll": 1.20, "n_tokens": 27904, "vocab_size": 65}`.

## Protocol

See the sibling wiki repo at
[`projects/autoresearch-shakespeare/`](https://github.com/Clamepending/mac-brain/tree/main/projects/autoresearch-shakespeare)
for the move / result / leaderboard / log setup.
