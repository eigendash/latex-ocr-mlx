"""Train the recogniser on artifacts/dataset.npz.

    python scripts/train.py --steps 6000
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latex_ocr import Tokenizer, vocabulary  # noqa: E402
from latex_ocr.model import LatexOCR, ModelConfig, num_params  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(ART / "dataset.npz")
    images, ids = data["train_images"], data["train_ids"].astype(np.int32)
    tok = Tokenizer(vocabulary())
    cfg = ModelConfig(vocab=len(tok))
    mx.random.seed(args.seed)
    model = LatexOCR(cfg)
    mx.eval(model.parameters())
    print(f"{num_params(model) / 1e6:.2f}M parameters, {len(images)} training formulas")

    warmup = 300
    schedule = optim.join_schedules(
        [optim.linear_schedule(0, args.lr, warmup), optim.cosine_decay(args.lr, args.steps - warmup)], [warmup]
    )
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    def loss_fn(model, x, seq):
        logits = model(x, seq[:, :-1])
        target = seq[:, 1:]
        mask = target != tok.pad
        return (nn.losses.cross_entropy(logits, target) * mask).sum() / mask.sum()

    step = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(args.seed)
    start, running = time.perf_counter(), None
    for i in range(1, args.steps + 1):
        pick = rng.integers(0, len(images), args.batch)
        x = mx.array(images[pick].astype(np.float32) / 255)
        seq = ids[pick]
        seq = mx.array(seq[:, : int((seq != tok.pad).sum(1).max())])
        loss, grads = step(model, x, seq)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
        if i % 250 == 0:
            print(f"step {i:5d}  loss {running:.4f}", flush=True)

    ART.mkdir(exist_ok=True)
    model.save_weights(str(ART / "model.safetensors"))
    (ART / "config.json").write_text(json.dumps(asdict(cfg)))
    print(f"saved to {ART / 'model.safetensors'} after {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
