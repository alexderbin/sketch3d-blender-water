"""Measure of pattern repetition in the ground texture (a 2D version of components/materials/bench/repeat.py): autocorrelation of the fine brightness of a texture window
(the large scale removed by a Gaussian σ 20 px); outside the central hump — the highest secondary peak. A tile grid gives strong peaks at the tile
step (like a "grid" over the ground); non-repeating ground — none.
.venv/bin/python -m components.site.bench.repeat2d <relief_tex.png> [window px=1024] → number (0 — does not repeat, 1 — an exact repeat)"""
import sys

import numpy as np
from PIL import Image


def repeat_peak2d(path, win=1024):
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(path).convert("L"), np.float32); H, W = a.shape
    a = a[(H - win) // 2:(H + win) // 2, (W - win) // 2:(W + win) // 2]   # the middle of the plot
    k = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2); k /= k.sum()
    low = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, a); low = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, low)
    p = a - low; p -= p.mean()
    F = np.fft.rfft2(p, s=(2 * win, 2 * win)); ac = np.fft.irfft2(F * np.conj(F))[:win, :win]; ac /= max(ac[0, 0], 1e-6)
    yy, xx = np.mgrid[:win, :win]; r = np.hypot(yy, xx)
    prof = np.array([ac[(r >= q) & (r < q + 1)].mean() for q in range(0, win // 2)])   # the central hump — up to the first dip below zero
    neg = np.nonzero(prof < 0)[0]; r0 = int(neg[0]) if len(neg) else win // 4
    m = (r > r0) & (r < win / 2)
    return float(max(0.0, ac[m].max()))


if __name__ == "__main__":
    print(round(repeat_peak2d(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1024), 3))
