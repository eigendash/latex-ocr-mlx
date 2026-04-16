"""Evaluate the trained model on the held-out splits.

    python scripts/evaluate.py
    python scripts/evaluate.py --beam-samples 200
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latex_ocr import Tokenizer, vocabulary  # noqa: E402
from latex_ocr.decode import beam_decode, greedy_decode  # noqa: E402
from latex_ocr.metrics import score  # noqa: E402
from latex_ocr.model import LatexOCR, ModelConfig  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"


def load():
    cfg = ModelConfig(**json.loads((ART / "config.json").read_text()))
    model = LatexOCR(cfg)
    model.load_weights(str(ART / "model.safetensors"))
    mx.eval(model.parameters())
    return model


def references(tok, ids):
    return [tok.decode_tokens(row[1:]) for row in ids.astype(np.int32)]


def run_greedy(model, tok, images, max_len=72, batch=100):
    out = []
    for i in range(0, len(images), batch):
        x = mx.array(images[i : i + batch].astype(np.float32) / 255)
        out += [tok.decode_tokens(ids) for ids in greedy_decode(model, x, tok.bos, tok.eos, max_len)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam-samples", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    args = ap.parse_args()

    data = np.load(ART / "dataset.npz")
    tok = Tokenizer(vocabulary())
    model = load()

    rows, results = [], {}
    for split in ("test", "hard"):
        images, ids = data[f"{split}_images"], data[f"{split}_ids"]
        refs = references(tok, ids)
        greedy = run_greedy(model, tok, images)
        n = min(args.beam_samples, len(images))
        beams = [
            tok.decode_tokens(
                beam_decode(model, mx.array(images[i].astype(np.float32) / 255), tok.bos, tok.eos, args.beam)
            )
            for i in range(n)
        ]
        g_full, g_sub, b_sub = score(refs, greedy), score(refs[:n], greedy[:n]), score(refs[:n], beams)
        results[split] = {"greedy_all": g_full, "greedy_subset": g_sub, "beam_subset": b_sub, "n": len(refs), "n_beam": n}
        rows += [
            (split, "greedy", len(refs), g_full),
            (split, "greedy", n, g_sub),
            (split, f"beam {args.beam}", n, b_sub),
        ]
        wrong = [(r, p) for r, p in zip(refs, greedy) if r != p][:3]
        print(f"\n{split}: first greedy errors")
        for r, p in wrong:
            print("  ref :", " ".join(r), "\n  pred:", " ".join(p))

    print("\n| split | decoding | formulas | exact match | edit distance | corpus edit distance |")
    print("|---|---|---:|---:|---:|---:|")
    for split, name, n, s in rows:
        print(f"| {split} | {name} | {n} | {s['exact']:.3f} | {s['edit']:.3f} | {s['corpus_edit']:.3f} |")
    (ART / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
