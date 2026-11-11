"""Helpers for horizons, silhouettes and water reflections."""
import numpy as np
import cv2
from common import F32, noise1d, pixel_grid, fbm, sample, rng


def ridge(W, base, amp, cell, seed, octaves=7, sharp=0.0, gain=0.5):
    """Mountain profile: returns y (pixel row of the ridge top) for every column."""
    n = noise1d(W, cell, seed, octaves=octaves, gain=gain)
    if sharp > 0:
        r = noise1d(W, cell * 0.7, seed + 99, octaves=octaves, gain=gain)
        n = (1 - sharp) * n + sharp * (1.5 - np.sqrt(r * r + 0.12) * 2.0)
    return base - amp * n


def silhouette_mask(H, ytop, soft=1.0):
    """Anti-aliased mask (H,W): 1 below the ridge line."""
    rows = np.arange(H, dtype=F32)[:, None]
    return np.clip((rows - ytop[None, :]) / soft + 0.5, 0, 1)


def lake_reflection(img, horizon, seed, ripple=6.0, stretch=1.0, darken=0.65, tint=(0.9, 0.95, 1.0)):
    """Mirror everything above `horizon` (pixel row) onto the water below it."""
    H, W = img.shape[:2]
    S = H / 2160
    px, py = pixel_grid(W, H)
    depth = np.maximum(py - horizon, 0) / max(H - horizon, 1)  # 0 at horizon, 1 at bottom
    n1 = fbm(W, H, 60 * S, 5, seed=seed)
    n2 = fbm(W, H, 25 * S, 4, seed=seed + 1)
    # ripples are squashed vertically by perspective
    amp = ripple * S * (0.15 + 2.5 * depth)
    mx = px + n2 * amp * 0.6
    my = 2 * horizon - py + n1 * amp * stretch
    my = np.minimum(my, horizon - 1)
    refl = cv2.remap(img, mx.astype(F32), my.astype(F32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fres = darken * (0.55 + 0.45 * (1 - depth) ** 3)
    refl = refl * fres[..., None] * np.array(tint, F32)
    out = img.copy()
    m = py >= horizon
    out[m] = refl[m]
    # thin bright line where water meets shore haze
    return out
