"""Greedy and beam-search decoding. The decoder is re-run on the whole prefix each step,
which is cheap at these lengths and keeps the code simple."""

import mlx.core as mx
import numpy as np


def greedy_decode(model, images, bos, eos, max_len=64):
    """images: (B, H, W). Returns a list of token-id lists, without bos/eos."""
    memory = model.encode(images)
    B = images.shape[0]
    tokens = mx.full((B, 1), bos, dtype=mx.int32)
    done = np.zeros(B, dtype=bool)
    for _ in range(max_len):
        nxt = model.decoder(tokens, memory)[:, -1].argmax(-1)
        nxt = mx.where(mx.array(done), eos, nxt).astype(mx.int32)
        tokens = mx.concatenate([tokens, nxt[:, None]], axis=1)
        mx.eval(tokens)
        done |= np.array(nxt) == eos
        if done.all():
            break
    out = []
    for row in np.array(tokens)[:, 1:]:
        ids = []
        for t in row:
            if t == eos:
                break
            ids.append(int(t))
        out.append(ids)
    return out


def beam_decode(model, image, bos, eos, beam=4, max_len=64):
    """Beam search for one image (H, W) with length-normalised log-probability."""
    memory = model.encode(image[None])
    seqs = np.array([[bos]], dtype=np.int32)
    scores = np.zeros(1)
    finished = []
    for _ in range(max_len):
        n = len(seqs)
        logits = model.decoder(mx.array(seqs), mx.broadcast_to(memory, (n, *memory.shape[1:])))[:, -1]
        logp = np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        total = scores[:, None] + logp
        flat = np.argpartition(-total.ravel(), min(beam, total.size) - 1)[:beam]
        flat = flat[np.argsort(-total.ravel()[flat])]
        keep_seqs, keep_scores = [], []
        for idx in flat:
            b, tok = divmod(int(idx), logp.shape[1])
            score = total[b, tok]
            if tok == eos:
                finished.append((score / (seqs.shape[1]), list(seqs[b, 1:])))
            else:
                keep_seqs.append(np.append(seqs[b], tok))
                keep_scores.append(score)
        if not keep_seqs or (len(finished) >= beam and max(f[0] for f in finished) > max(keep_scores) / seqs.shape[1]):
            break
        seqs, scores = np.stack(keep_seqs), np.array(keep_scores)
    if not finished:
        finished = [(scores[0] / seqs.shape[1], list(seqs[0, 1:]))]
    return [int(t) for t in max(finished, key=lambda f: f[0])[1]]
