def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(references, predictions):
    """Exact match rate and edit distance, both per token.

    `edit` is the mean over formulas of distance / reference length; `corpus_edit` is
    total distance over total reference tokens, which weights long formulas more.
    """
    n = len(references)
    exact = sum(r == p for r, p in zip(references, predictions)) / n
    dists = [edit_distance(r, p) for r, p in zip(references, predictions)]
    return {
        "exact": exact,
        "edit": sum(d / max(len(r), 1) for d, r in zip(dists, references)) / n,
        "corpus_edit": sum(dists) / sum(len(r) for r in references),
    }
