"""A random formula grammar limited to what matplotlib's mathtext can draw.

Formulas are produced directly as token lists, so the tokenizer round-trips them
exactly. Sub- and superscript arguments, fraction parts and so on are always braced,
even for a single symbol, so that two different token sequences do not draw the same
picture. (`a^b` and `a^{b}` would; only the second form is ever generated.)

Matrices are not in the grammar: mathtext has no `matrix` environment.
"""

import random

LETTERS = list("abcdefghijklmnopqrstuvwxyz") + list("ABCDEFGH")
GREEK = [
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\theta", r"\lambda",
    r"\mu", r"\pi", r"\sigma", r"\phi", r"\omega", r"\Gamma", r"\Delta", r"\Omega",
]
BINARY = ["+", "-", "=", r"\times", r"\cdot", r"\leq", r"\geq", r"\neq", r"\approx", r"\pm"]
FUNCTIONS = [r"\sin", r"\cos", r"\log", r"\ln"]
BIG = [r"\sum", r"\int", r"\prod"]
ACCENTS = [r"\hat", r"\vec", r"\bar"]
EXTRA = [r"\infty", r"\partial", r"\to", "(", ")", "[", "]", "|", ",", ".", "/", "!", "'", "<", ">", "{", "}", "^", "_"]
STRUCTURE = [r"\frac", r"\sqrt", r"\binom", r"\left", r"\right"]
DIGITS = list("0123456789")


def vocabulary():
    seen, out = set(), []
    for t in DIGITS + LETTERS + GREEK + BINARY + FUNCTIONS + BIG + ACCENTS + EXTRA + STRUCTURE:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class FormulaGenerator:
    def __init__(self, seed=0, max_tokens=48):
        self.rng = random.Random(seed)
        self.max_tokens = max_tokens

    def symbol(self):
        r = self.rng.random()
        if r < 0.55:
            return [self.rng.choice(LETTERS)]
        if r < 0.80:
            return [self.rng.choice(GREEK)]
        return [self.rng.choice(DIGITS) for _ in range(self.rng.choice([1, 1, 2, 3]))]

    def group(self, body):
        return ["{", *body, "}"]

    def term(self, depth):
        rng = self.rng
        if depth <= 0:
            return self.symbol()
        kind = rng.choices(
            ["symbol", "supsub", "frac", "sqrt", "bigop", "func", "paren", "binom", "accent"],
            weights=[18, 20, 14, 8, 10, 8, 8, 4, 6],
        )[0]
        if kind == "symbol":
            return self.symbol()
        if kind == "supsub":
            base = self.symbol()
            sub, sup = self.expr(depth - 1, 1), self.expr(depth - 1, 1)
            mode = rng.choice(["sup", "sub", "both"])
            out = list(base)
            if mode in ("sub", "both"):
                out += ["_", *self.group(sub)]
            if mode in ("sup", "both"):
                out += ["^", *self.group(sup)]
            return out
        if kind == "frac":
            return [r"\frac", *self.group(self.expr(depth - 1, 2)), *self.group(self.expr(depth - 1, 2))]
        if kind == "sqrt":
            return [r"\sqrt", *self.group(self.expr(depth - 1, 2))]
        if kind == "bigop":
            op = rng.choice(BIG)
            var = rng.choice(LETTERS[:26])
            lower = [var, "=", rng.choice(DIGITS)] if op != r"\int" else self.symbol()
            upper = self.symbol()
            body = self.term(depth - 1)
            tail = ["d", var] if op == r"\int" else []
            return [op, "_", *self.group(lower), "^", *self.group(upper), *body, *tail]
        if kind == "func":
            return [rng.choice(FUNCTIONS), *self.term(depth - 1)]
        if kind == "paren":
            return [r"\left", "(", *self.expr(depth - 1, 3), r"\right", ")"]
        if kind == "binom":
            return [r"\binom", *self.group(self.symbol()), *self.group(self.symbol())]
        return [rng.choice(ACCENTS), *self.group([rng.choice(LETTERS)])]

    def expr(self, depth, max_terms):
        n = self.rng.randint(1, max_terms)
        out = self.term(depth)
        for _ in range(n - 1):
            out += [self.rng.choice(BINARY), *self.term(depth)]
        return out

    def sample(self, depth=2, max_terms=3):
        """One formula as a token list, at most `max_tokens` long."""
        while True:
            tokens = self.expr(depth, max_terms)
            if 3 <= len(tokens) <= self.max_tokens:
                return tokens
