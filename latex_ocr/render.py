"""Draw a formula onto a fixed-size grayscale canvas with matplotlib's mathtext."""

import io

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import font_manager, image, mathtext  # noqa: E402

from .tokenizer import detokenize  # noqa: E402

HEIGHT, WIDTH = 64, 256
MARGIN = 4
_PROP = font_manager.FontProperties(size=14)


def render(tokens, height=HEIGHT, width=WIDTH):
    """Return a (height, width) uint8 ink map (0 background, 255 ink), or None.

    None means the formula is malformed for mathtext or does not fit on the canvas.
    The drawing is placed at the left margin and centred vertically.
    """
    buf = io.BytesIO()
    try:
        mathtext.math_to_image(f"${detokenize(tokens)}$", buf, dpi=100, format="png", prop=_PROP, color="black")
    except Exception:
        return None
    buf.seek(0)
    rgba = image.imread(buf)
    # the PNG has an opaque white background, so ink is darkness, not alpha
    alpha = 1.0 - rgba[..., :3].mean(axis=-1)
    ys, xs = np.nonzero(alpha > 0.02)
    if len(ys) == 0:
        return None
    crop = alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    if h > height - 2 * 2 or w > width - 2 * MARGIN:
        return None
    canvas = np.zeros((height, width), dtype=np.float32)
    top = (height - h) // 2
    canvas[top : top + h, MARGIN : MARGIN + w] = crop
    return (canvas * 255).astype(np.uint8)
