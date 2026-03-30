"""Command-level LaTeX tokenizer.

A token is a control sequence (`\\frac`, `\\alpha`), an escaped single character
(`\\{`), a single digit, a single letter, or any other single non-space character.
Whitespace only separates tokens and is never a token itself, so `a^{2}` and
`a ^ { 2 }` tokenize identically. Digits are split one per token on purpose: it
keeps the vocabulary small and makes the model read numbers glyph by glyph.
"""

import re

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[0-9]|[A-Za-z]|\S")


def tokenize(text):
    return _TOKEN.findall(text)


def detokenize(tokens):
    """Canonical text for a token list: one space between tokens."""
    return " ".join(tokens)


class Tokenizer:
    def __init__(self, vocabulary):
        self.itos = SPECIALS + [t for t in vocabulary if t not in SPECIALS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos, self.unk = (self.stoi[t] for t in SPECIALS)

    def __len__(self):
        return len(self.itos)

    def encode(self, text_or_tokens, add_special=True):
        tokens = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else text_or_tokens
        ids = [self.stoi.get(t, self.unk) for t in tokens]
        return [self.bos, *ids, self.eos] if add_special else ids

    def decode_tokens(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.itos[i])
        return out

    def decode(self, ids):
        return detokenize(self.decode_tokens(ids))
