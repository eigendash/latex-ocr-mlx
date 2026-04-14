import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from latex_ocr import FormulaGenerator, Tokenizer, render, vocabulary
from latex_ocr.decode import beam_decode, greedy_decode
from latex_ocr.metrics import edit_distance, score
from latex_ocr.model import LatexOCR, ModelConfig, num_params


def tiny_cfg(tok):
    return ModelConfig(vocab=len(tok), dim=64, heads=2, enc_layers=1, dec_layers=2, ff=128, max_len=40)


def test_edit_distance_known_values():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], [1, 2]) == 2
    assert edit_distance(["a", "b"], ["a", "b"]) == 0


def test_score_exact_and_edit():
    s = score([["a", "b"], ["c"]], [["a", "b"], ["d"]])
    assert s["exact"] == 0.5
    assert s["edit"] == 0.5 and s["corpus_edit"] == 1 / 3


def test_shapes_and_default_size_is_in_the_target_range():
    tok = Tokenizer(vocabulary())
    model = LatexOCR(ModelConfig(vocab=len(tok)))
    images = mx.zeros((2, 64, 256))
    assert model.encode(images).shape == (2, 256, 256)
    assert model(images, mx.zeros((2, 7), dtype=mx.int32)).shape == (2, 7, len(tok))
    assert 5e6 < num_params(model) < 20e6


def test_decoder_is_causal():
    tok = Tokenizer(vocabulary())
    mx.random.seed(0)
    model = LatexOCR(tiny_cfg(tok))
    images = mx.random.uniform(shape=(2, 64, 256))
    a = mx.random.randint(0, len(tok), (2, 12))
    b = mx.concatenate([a[:, :6], mx.random.randint(0, len(tok), (2, 6))], axis=1)
    la, lb = np.array(model(images, a)), np.array(model(images, b))
    np.testing.assert_allclose(la[:, :6], lb[:, :6], atol=1e-5)
    assert not np.allclose(la[:, 6:], lb[:, 6:], atol=1e-5)


def test_decoder_actually_reads_the_image():
    tok = Tokenizer(vocabulary())
    mx.random.seed(0)
    model = LatexOCR(tiny_cfg(tok))
    tokens = mx.zeros((1, 5), dtype=mx.int32)
    one = np.array(model(mx.random.uniform(shape=(1, 64, 256)), tokens))
    two = np.array(model(mx.random.uniform(shape=(1, 64, 256)), tokens))
    assert not np.allclose(one, two, atol=1e-4)


def _memorise(steps=250):
    tok = Tokenizer(vocabulary())
    gen = FormulaGenerator(seed=2, max_tokens=14)
    samples = []
    while len(samples) < 6:
        tokens = gen.sample(depth=1, max_terms=2)
        img = render(tokens)
        if img is not None:
            samples.append((img, tokens))
    images = mx.array(np.stack([s[0] for s in samples]).astype(np.float32) / 255)
    width = max(len(s[1]) for s in samples) + 2
    seqs = np.zeros((len(samples), width), dtype=np.int32)
    for i, (_, t) in enumerate(samples):
        ids = tok.encode(t)
        seqs[i, : len(ids)] = ids
    seqs = mx.array(seqs)

    mx.random.seed(1)
    model = LatexOCR(tiny_cfg(tok))
    opt = optim.AdamW(learning_rate=2e-3)

    def loss_fn(model):
        logits = model(images, seqs[:, :-1])
        mask = seqs[:, 1:] != tok.pad
        return (nn.losses.cross_entropy(logits, seqs[:, 1:]) * mask).sum() / mask.sum()

    step = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        loss, grads = step(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    return tok, model, images, samples, float(loss)


def test_tiny_model_memorises_a_handful_of_formulas_and_decoders_agree():
    tok, model, images, samples, loss = _memorise()
    assert loss < 0.1
    greedy = greedy_decode(model, images, tok.bos, tok.eos, max_len=30)
    for ids, (_, tokens) in zip(greedy, samples):
        assert tok.decode_tokens(ids) == tokens
    beam1 = beam_decode(model, images[0], tok.bos, tok.eos, beam=1, max_len=30)
    assert beam1 == greedy[0]
    beam4 = beam_decode(model, images[0], tok.bos, tok.eos, beam=4, max_len=30)
    assert tok.decode_tokens(beam4) == samples[0][1]


def test_greedy_terminates_at_max_len_for_an_untrained_model():
    tok = Tokenizer(vocabulary())
    model = LatexOCR(tiny_cfg(tok))
    out = greedy_decode(model, mx.zeros((2, 64, 256)), tok.bos, tok.eos, max_len=5)
    assert len(out) == 2 and all(len(o) <= 5 for o in out)
