"""Write the trained weights in a form a non-MLX runtime can load.

    python scripts/export.py             # float16 weights
    python scripts/export.py --float32

Produces artifacts/export/: weights.safetensors (flat tensor names, row-major),
config.json (the model hyperparameters plus the input canvas size), and vocab.json
(the id-to-token table, whose first four entries are the special tokens).
Nothing here is specific to MLX, so the safetensors file can be read from JavaScript
or ONNX tooling. Building a browser runtime is not part of this repo.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latex_ocr import HEIGHT, WIDTH, Tokenizer, vocabulary  # noqa: E402
from latex_ocr.tokenizer import SPECIALS  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    weights = mx.load(str(ART / "model.safetensors"))
    dtype = mx.float32 if args.float32 else mx.float16
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    out = ART / "export"
    out.mkdir(exist_ok=True)
    mx.save_safetensors(str(out / "weights.safetensors"), weights)

    cfg = json.loads((ART / "config.json").read_text())
    cfg.update(canvas_height=HEIGHT, canvas_width=WIDTH, pad=0, bos=1, eos=2, unk=3)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    tok = Tokenizer(vocabulary())
    assert tok.itos[:4] == SPECIALS
    (out / "vocab.json").write_text(json.dumps(tok.itos))

    size = (out / "weights.safetensors").stat().st_size / 1e6
    n = sum(v.size for v in weights.values())
    print(f"{len(weights)} tensors, {n / 1e6:.2f}M values, {size:.1f} MB as {dtype}")


if __name__ == "__main__":
    main()
