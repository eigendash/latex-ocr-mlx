# latex-ocr-mlx

A small image-to-LaTeX recognizer written in MLX. It has 7.6M parameters, trains on a laptop GPU, and is fitted to synthetic formulas rendered with matplotlib, which is the only data this repo uses.

It was inspired by Texo, a 20M-parameter formula recognizer that runs in the browser (technical report: *Texo: Formula Recognition within 20M Parameters*, arXiv:2602.17189). Texo distills a larger model and is fine-tuned on a million real formula images. I did not read its code, only its README and abstract. What I took from it is the premise that a model this small is enough for the task, and the references it builds on: *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models* (arXiv:2109.10282), *UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition* (arXiv:2404.15254), and PP-FormulaNet. This repo is an independent small-scale version trained from scratch on rendered text, so its numbers say nothing about Texo's accuracy and cannot be compared with its benchmark tables.

## Model

The input is a 64 by 256 grayscale ink map. Three strided convolutions reduce it to an 8 by 32 grid of 256-wide vectors, a learned position embedding is added, and four transformer encoder layers mix the 256 grid tokens. The decoder has four pre-norm layers with causal self-attention, cross-attention to the encoder grid, and a feed-forward block. Output logits reuse the token embedding matrix. Total size is 7.56M parameters.

Output tokens come from a command-level tokenizer (`latex_ocr/tokenizer.py`). A control sequence such as `\frac` is one token, a digit is one token, and whitespace is never a token. The vocabulary is 103 symbols plus four special tokens (107 in all), taken from the formula grammar, so unknown tokens do not occur on synthetic data.

## Data

`latex_ocr/grammar.py` generates random formulas directly as token lists: letters, Greek letters and digits; sub- and superscripts; fractions; square roots; sums, products and integrals with limits; trigonometric and logarithm functions; `\left(` and `\right)` pairs; binomials; and accents. Arguments are always braced, so two different token sequences do not draw the same picture. Matrices are missing because mathtext has no matrix environment.

`latex_ocr/render.py` draws each formula with matplotlib's mathtext at one font size and places it at the left of a fixed canvas. Formulas that do not fit are discarded. The splits are:

| split | formulas | source |
|---|---:|---|
| train | 15,836 | depth-2 grammar, de-duplicated against every evaluation split |
| validation | 500 | depth-2, separate seed |
| test | 500 | depth-2, separate seed, disjoint from validation |
| hard | 300 | depth-3 grammar, longer length cap |

The hard split is deeper in nesting but only slightly longer, 23.2 tokens on average against 20.0, because the canvas rejects the very long formulas.

## Training and results

Training ran for 6000 steps at batch size 64 with AdamW, a 300-step warmup and a cosine schedule, from a single seed. The final training loss was 0.18 and was still falling, so the model is probably undertrained rather than at a ceiling. I did not run a longer schedule.

Greedy decoding on all splits, plus beam search with four beams on the first 200 formulas of each split:

| split | decoding | formulas | exact match | edit distance | corpus edit distance |
|---|---|---:|---:|---:|---:|
| test | greedy | 500 | 0.392 | 0.122 | 0.162 |
| test | greedy | 200 | 0.390 | 0.123 | 0.159 |
| test | beam 4 | 200 | 0.400 | 0.114 | 0.151 |
| hard | greedy | 300 | 0.353 | 0.244 | 0.367 |
| hard | greedy | 200 | 0.400 | 0.224 | 0.346 |
| hard | beam 4 | 200 | 0.425 | 0.204 | 0.318 |

Exact match means the predicted token sequence equals the reference. Edit distance is Levenshtein distance over tokens divided by reference length and averaged over formulas, and the corpus version divides total distance by total reference tokens. The beam rows use 200 formulas, so a difference of a point or two in exact match is within noise. Beam search lowers edit distance in both splits by a more consistent margin, around 0.01 to 0.02. On the hard split the first 200 formulas happen to be easier than the full 300, which is why the two greedy rows differ.

About 60% of test formulas contain at least one wrong token. The errors I looked at fall into two groups. Some are small glyph confusions: `j` read as `l`, `B` as `w`. At this resolution a subscript glyph is only a few pixels tall, and the encoder reduces the image by a factor of eight in each direction. Others are structural: a subscript read as a superscript, or a nested group closed in the wrong place. The hard split shows more of the second kind.

These numbers are all on the same synthetic renderer, one font and one size. Nothing here says the model reads scanned pages, photographs or handwriting.

## Export

`scripts/export.py` writes `artifacts/export/` with `weights.safetensors`, `config.json` and `vocab.json`. The tensor names are flat and the layout is row-major, so the file can be read by something other than MLX. The float16 file is 15.1 MB and float32 is 30.2 MB. I did not write a browser runtime, so I have not checked that one can reproduce the MLX outputs.

## Running it

```
pip install -e ".[dev]"
pytest
python scripts/build_dataset.py
python scripts/train.py --steps 6000
python scripts/evaluate.py
python scripts/export.py
```

The 18 tests cover the tokenizer, the grammar, the renderer, the shapes and parameter count, decoder causality, whether the decoder actually uses the image, and decoding. One test trains a tiny copy of the model to memorize six formulas and then checks that greedy decoding reproduces them and that beam search with one beam matches greedy. A bug in the renderer was caught this way: my first version read the PNG's alpha channel, which is solid, so different formulas rendered to the same picture.

## Limits

One seed, one font, one resolution, and a short training schedule. The decoder recomputes the full prefix at every step instead of caching keys and values. Beam search runs one image at a time. There is no data augmentation, no real images, and no support for matrices or multi-line equations.
