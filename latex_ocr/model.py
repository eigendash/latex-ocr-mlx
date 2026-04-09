"""Image-to-LaTeX encoder-decoder.

The encoder turns a 64x256 ink map into 256 tokens: a small strided-convolution stem
reduces it to an 8x32 grid and a few transformer layers mix the grid. The decoder is
an ordinary pre-norm transformer with causal self-attention and cross-attention to the
encoder grid. Output logits share the token embedding matrix.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .render import HEIGHT, WIDTH


@dataclass
class ModelConfig:
    vocab: int = 120
    dim: int = 256
    heads: int = 4
    enc_layers: int = 4
    dec_layers: int = 4
    ff: int = 1024
    max_len: int = 72


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, context=None, causal=False):
        B, T, D = x.shape
        context = x if context is None else context
        q = self.q(x)
        k, v = mx.split(self.kv(context), 2, axis=-1)

        def split(t):
            return t.reshape(B, t.shape[1], self.heads, -1).transpose(0, 2, 1, 3)

        o = mx.fast.scaled_dot_product_attention(
            split(q), split(k), split(v), scale=(D // self.heads) ** -0.5, mask="causal" if causal else None
        )
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)

    def __call__(self, x):
        return self.down(nn.gelu(self.up(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.heads)
        self.ff = FeedForward(cfg.dim, cfg.ff)

    def __call__(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ff(self.n2(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(cfg.dim) for _ in range(3))
        self.self_attn = Attention(cfg.dim, cfg.heads)
        self.cross_attn = Attention(cfg.dim, cfg.heads)
        self.ff = FeedForward(cfg.dim, cfg.ff)

    def __call__(self, x, memory):
        x = x + self.self_attn(self.n1(x), causal=True)
        x = x + self.cross_attn(self.n2(x), context=memory)
        return x + self.ff(self.n3(x))


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(64, cfg.dim, 2, stride=2)
        self.n_tokens = (HEIGHT // 8) * (WIDTH // 8)
        self.pos = mx.random.normal((self.n_tokens, cfg.dim)) * 0.02
        self.layers = [EncoderLayer(cfg) for _ in range(cfg.enc_layers)]
        self.norm = nn.LayerNorm(cfg.dim)

    def __call__(self, images):
        """images: (B, H, W) floats in [0, 1]."""
        x = images[..., None]
        x = nn.gelu(self.c1(x))
        x = nn.gelu(self.c2(x))
        x = self.c3(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1]) + self.pos
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos = mx.random.normal((cfg.max_len, cfg.dim)) * 0.02
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.dec_layers)]
        self.norm = nn.LayerNorm(cfg.dim)

    def __call__(self, tokens, memory):
        x = self.embed(tokens) + self.pos[: tokens.shape[1]]
        for layer in self.layers:
            x = layer(x, memory)
        return self.embed.as_linear(self.norm(x))


class LatexOCR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, images):
        return self.encoder(images)

    def __call__(self, images, tokens):
        return self.decoder(tokens, self.encoder(images))


def num_params(model):
    return sum(v.size for _, v in tree_flatten(model.parameters()))
