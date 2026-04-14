"""Render a synthetic dataset to artifacts/dataset.npz.

    python scripts/build_dataset.py                 # 16000 train, 500 val, 500 test, 300 harder
    python scripts/build_dataset.py --train 4000

Train, validation and test use depth-2 formulas from disjoint seeds and are
de-duplicated against each other. The "hard" split uses depth-3 formulas with a
longer length cap, to see how the model behaves past the training distribution.
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latex_ocr import FormulaGenerator, Tokenizer, render, vocabulary  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"
MAX_IDS = 72


def make_split(args):
    seed, count, depth, max_tokens, exclude = args
    gen = FormulaGenerator(seed=seed, max_tokens=max_tokens)
    seen, images, formulas = set(exclude), [], []
    while len(formulas) < count:
        tokens = gen.sample(depth=depth)
        key = " ".join(tokens)
        if key in seen:
            continue
        img = render(tokens)
        if img is None:
            continue
        seen.add(key)
        images.append(img)
        formulas.append(tokens)
    return images, formulas


def pack(tok, images, formulas):
    ids = np.zeros((len(formulas), MAX_IDS), dtype=np.int16)
    for i, tokens in enumerate(formulas):
        row = tok.encode(tokens)
        ids[i, : len(row)] = row
    return np.stack(images), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=16000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--hard", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tok = Tokenizer(vocabulary())
    # Evaluation splits are generated first so training can exclude every one of their formulas.
    with Pool(args.workers) as pool:
        val_i, val_f = make_split((101, args.val, 2, 48, []))
        test_i, test_f = make_split((202, args.test, 2, 48, [" ".join(f) for f in val_f]))
        hard_i, hard_f = make_split((303, args.hard, 3, 64, []))
        held = [" ".join(f) for f in val_f + test_f + hard_f]
        shards = args.workers
        parts = pool.map(
            make_split, [(1000 + s, args.train // shards, 2, 48, held) for s in range(shards)]
        )
    train_i = [im for p in parts for im in p[0]]
    train_f = [f for p in parts for f in p[1]]
    # Different shards can produce the same formula; drop repeats so train has no duplicates.
    seen, keep = set(), []
    for k, f in enumerate(train_f):
        key = " ".join(f)
        if key not in seen:
            seen.add(key)
            keep.append(k)
    train_i, train_f = [train_i[k] for k in keep], [train_f[k] for k in keep]

    arrays = {}
    for name, (imgs, fs) in {
        "train": (train_i, train_f), "val": (val_i, val_f), "test": (test_i, test_f), "hard": (hard_i, hard_f)
    }.items():
        arrays[f"{name}_images"], arrays[f"{name}_ids"] = pack(tok, imgs, fs)
        print(f"{name}: {len(fs)} formulas, mean {np.mean([len(f) for f in fs]):.1f} tokens")
    ART.mkdir(exist_ok=True)
    np.savez_compressed(ART / "dataset.npz", **arrays)
    print(f"wrote {ART / 'dataset.npz'}")


if __name__ == "__main__":
    main()
