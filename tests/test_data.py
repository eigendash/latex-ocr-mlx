import numpy as np
import pytest

from latex_ocr import FormulaGenerator, Tokenizer, render, tokenize, vocabulary
from latex_ocr.tokenizer import detokenize


def test_tokenize_commands_digits_and_escapes():
    assert tokenize(r"\frac{12}{x} \alpha_i \{") == [
        r"\frac", "{", "1", "2", "}", "{", "x", "}", r"\alpha", "_", "i", r"\{",
    ]


def test_whitespace_does_not_change_tokens():
    assert tokenize("a^{2}+b") == tokenize("a ^ { 2 } + b")


def test_command_stops_at_non_letters():
    assert tokenize(r"\sin2x") == [r"\sin", "2", "x"]
    assert tokenize(r"\alpha\beta") == [r"\alpha", r"\beta"]


def test_encode_decode_roundtrip_and_specials():
    tok = Tokenizer(vocabulary())
    ids = tok.encode(r"\frac { a } { 2 }")
    assert ids[0] == tok.bos and ids[-1] == tok.eos
    assert tok.decode(ids) == r"\frac { a } { 2 }"
    assert tok.decode(ids + [tok.pad, tok.pad]) == r"\frac { a } { 2 }"


def test_unknown_token_maps_to_unk_and_decode_stops_at_eos():
    tok = Tokenizer(vocabulary())
    ids = tok.encode("a \\notatoken b")
    assert tok.unk in ids
    assert tok.decode([tok.bos, *tok.encode("a b", add_special=False), tok.eos, *tok.encode("c", add_special=False)]) == "a b"


def test_vocabulary_has_no_duplicates_and_covers_the_grammar():
    vocab = vocabulary()
    assert len(vocab) == len(set(vocab))
    tok = Tokenizer(vocab)
    gen = FormulaGenerator(seed=1)
    for _ in range(300):
        assert tok.unk not in tok.encode(gen.sample(depth=3))


def test_generator_is_deterministic_and_respects_length_cap():
    a = [FormulaGenerator(seed=7, max_tokens=30).sample() for _ in range(1)]
    b = [FormulaGenerator(seed=7, max_tokens=30).sample() for _ in range(1)]
    assert a == b
    gen = FormulaGenerator(seed=3, max_tokens=24)
    assert all(3 <= len(gen.sample()) <= 24 for _ in range(200))


def test_generated_tokens_survive_tokenize_of_their_text():
    gen = FormulaGenerator(seed=5)
    for _ in range(200):
        tokens = gen.sample(depth=3)
        assert tokenize(detokenize(tokens)) == tokens
        assert tokens.count("{") == tokens.count("}")
        assert tokens.count(r"\left") == tokens.count(r"\right")


def test_render_shape_ink_and_distinct_pictures():
    a = render(tokenize(r"\frac { a } { b }"))
    b = render(tokenize(r"\frac { b } { a }"))
    assert a.shape == (64, 256) and a.dtype == np.uint8
    assert a.max() > 128 and a[:, 200:].max() == 0
    assert not np.array_equal(a, b)


def test_render_rejects_malformed_and_oversized():
    assert render(tokenize(r"\frac { a }")) is None
    assert render(tokenize(" ".join(["x"] * 200))) is None


def test_most_generated_formulas_render():
    gen = FormulaGenerator(seed=11)
    ok = sum(render(gen.sample()) is not None for _ in range(60))
    assert ok >= 40, ok
