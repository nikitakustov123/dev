"""Shared rendering toolkit: noise, colour, stars, post-processing.

Everything is vectorised NumPy / OpenCV so a full 3840x2160 frame
renders in seconds to a couple of minutes on a laptop CPU.
"""
import math
import numpy as np
import cv2
from PIL import Image

F32 = np.float32


def rng(seed):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------
# coordinates
# ----------------------------------------------------------------------------
def coords(W, H):
    """Normalised coords: y in [-1, 1] (up), x in [-W/H, W/H]."""
    x = (np.arange(W, dtype=F32) + 0.5 - W / 2) / (H / 2)
    y = (H / 2 - np.arange(H, dtype=F32) - 0.5) / (H / 2)
    return np.meshgrid(x, y)


def pixel_grid(W, H):
    return np.meshgrid(np.arange(W, dtype=F32), np.arange(H, dtype=F32))


# ----------------------------------------------------------------------------
# noise
# ----------------------------------------------------------------------------
def noise_layer(W, H, cell, seed, rotate=True):
    """Smooth value noise with feature size `cell` pixels (rotated lattice)."""
    r = rng(seed)
    n = max(int(math.hypot(W, H) / cell) + 8, 4)
    src = r.random((n, n), dtype=F32)
    ang = r.uniform(0, 2 * np.pi) if rotate else 0.0
    s = 1.0 / cell
    c, sn = math.cos(ang), math.sin(ang)
    cx, cy = W / 2, H / 2
    ox, oy = r.uniform(0, n), r.uniform(0, n)
    M = np.array([[c * s, -sn * s, ox - (c * cx - sn * cy) * s],
                  [sn * s, c * s, oy - (sn * cx + c * cy) * s]], np.float32)
    return cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_WRAP)


def fbm(W, H, cell, octaves=8, gain=0.5, lac=2.0, seed=0, ridged=False, min_cell=0.75,
        normalize=True):
    """Fractal Brownian motion. Returns z-scored field (mean 0, std 1)."""
    total = np.zeros((H, W), F32)
    amp = 1.0
    for i in range(octaves):
        c = cell / lac ** i
        if c < min_cell:
            break
        L = noise_layer(W, H, c, seed * 7919 + i * 31 + 1)
        if ridged:
            L = (1.0 - np.abs(2.0 * L - 1.0)) ** 2
        total += F32(amp) * L
        amp *= gain
    if normalize:
        total -= total.mean()
        total /= (total.std() + 1e-8)
    return total


def tile_noise(N, cell, octaves, seed, gain=0.5, ridged=False):
    """Tileable fBm texture of size N x N (cell must divide N for perfect tiling)."""
    r = rng(seed)
    total = np.zeros((N, N), F32)
    amp = 1.0
    c = cell
    for i in range(octaves):
        k = max(int(round(N / c)), 2)
        src = r.random((k, k), dtype=F32)
        s = k / N
        ox, oy = r.integers(0, k), r.integers(0, k)
        M = np.array([[s, 0, ox], [0, s, oy]], np.float32)
        L = cv2.warpAffine(src, M, (N, N), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_WRAP)
        if ridged:
            L = (1.0 - np.abs(2.0 * L - 1.0)) ** 2
        total += F32(amp) * L
        amp *= gain
        c /= 2
        if c < 1:
            break
    total -= total.mean()
    total /= total.std() + 1e-8
    return total


def sample(tex, x, y, border=cv2.BORDER_WRAP, interp=cv2.INTER_LINEAR):
    """Sample a texture at float pixel coordinates (arrays of any 2D shape)."""
    return cv2.remap(tex, x.astype(F32), y.astype(F32), interp, borderMode=border)


def noise1d(n, cell, seed, octaves=1, gain=0.5):
    r = rng(seed)
    out = np.zeros(n, F32)
    amp = 1.0
    for o in range(octaves):
        c = cell / 2 ** o
        if c < 1:
            break
        k = int(n / c) + 4
        v = r.random(k).astype(F32)
        t = np.arange(n, dtype=F32) / c + r.uniform(0, 1)
        i = np.floor(t).astype(int)
        f = t - i
        f = f * f * (3 - 2 * f)
        out += F32(amp) * (v[i] * (1 - f) + v[i + 1] * f)
        amp *= gain
    out -= out.mean()
    out /= out.std() + 1e-8
    return out


# ----------------------------------------------------------------------------
# maths helpers
# ----------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return a + (b - a) * t


def blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def fast_blur(img, sigma, factor=4):
    """Large-radius blur done at reduced resolution."""
    H, W = img.shape[:2]
    if sigma < 6:
        return blur(img, sigma)
    sw, sh = max(W // factor, 1), max(H // factor, 1)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def palette(t, stops):
    """stops: list of (pos, (r,g,b)) linear colours. t any shape -> (...,3)."""
    pos = np.array([s[0] for s in stops], F32)
    cols = np.array([s[1] for s in stops], F32)
    t = np.clip(t, pos[0], pos[-1])
    out = np.empty(t.shape + (3,), F32)
    for c in range(3):
        out[..., c] = np.interp(t, pos, cols[:, c])
    return out


def hexcol(h):
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], F32)
    return srgb_to_linear(c)


def srgb_to_linear(c):
    c = np.asarray(c, F32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(F32)


def kelvin(T):
    """Approximate black-body colour (linear RGB, max channel ~1) for T in Kelvin."""
    T = np.asarray(T, F32) / 100.0
    r = np.where(T <= 66, 255.0, 329.698727446 * np.power(np.maximum(T - 60, 1e-3), -0.1332047592))
    g = np.where(T <= 66, 99.4708025861 * np.log(np.maximum(T, 1e-3)) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(T - 60, 1e-3), -0.0755148492))
    b = np.where(T >= 66, 255.0, np.where(T <= 19, 0.0,
                 138.5177312231 * np.log(np.maximum(T - 10, 1e-3)) - 305.0447927307))
    rgb = np.stack([r, g, b], -1)
    rgb = np.clip(rgb, 0, 255) / 255.0
    return srgb_to_linear(rgb)


def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


# ----------------------------------------------------------------------------
# stars
# ----------------------------------------------------------------------------
def splat(buf, x, y, val):
    """Bilinear splat of values (n,3) at float pixel coords into buf (H,W,3)."""
    H, W = buf.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(F32)[:, None]
    fy = (y - y0).astype(F32)[:, None]
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = yi[m] * W + xi[m]
        v = val[m] * w[m]
        for c in range(3):
            buf[..., c].reshape(-1)[:] += np.bincount(idx, weights=v[:, c], minlength=H * W).astype(F32)
    return buf


def starfield(W, H, n, seed, density=None, bright=1.0, spikes=True, min_t=2800, max_t=14000):
    """Realistic star layer with PSF, halo and diffraction spikes for bright stars."""
    S = H / 2160
    r = rng(seed)
    if density is None:
        x = r.uniform(0, W, n)
        y = r.uniform(0, H, n)
    else:
        # rejection sample against a density map in [0,1]
        xs, ys = [], []
        need = n
        while need > 0:
            m = need * 3
            cx = r.uniform(0, W, m)
            cy = r.uniform(0, H, m)
            d = density[np.clip(cy.astype(int), 0, H - 1), np.clip(cx.astype(int), 0, W - 1)]
            keep = r.random(m) < d
            xs.append(cx[keep][:need])
            ys.append(cy[keep][:need])
            need -= min(int(keep.sum()), need)
        x = np.concatenate(xs)[:n]
        y = np.concatenate(ys)[:n]
    mag = (r.pareto(1.35, len(x)) + 1.0)
    mag = np.minimum(mag, 600)
    T = np.exp(r.uniform(np.log(min_t), np.log(max_t), len(x)))
    col = kelvin(T)
    col = col / col.max(axis=1, keepdims=True)
    val = col * (mag[:, None] * 0.035 * bright).astype(F32)
    buf = np.zeros((H, W, 3), F32)
    splat(buf, x, y, val)
    out = blur(buf, 0.6 * S) * 2.0
    hot = np.maximum(buf - 0.5 * bright, 0)
    out += blur(hot, 2.0 * S) * 4.0
    out += fast_blur(hot, 12 * S) * 12.0
    if spikes:
        vhot = np.maximum(buf - 2.5 * bright, 0)
        L = max(int(90 * S) | 1, 9)
        k = np.zeros((L, L), F32)
        k[L // 2, :] = np.exp(-np.abs(np.arange(L) - L // 2) / (L / 7))
        k[:, L // 2] = k[L // 2, :]
        k /= k.sum() / 6
        out += cv2.filter2D(vhot, -1, k)
    return out


# ----------------------------------------------------------------------------
# post
# ----------------------------------------------------------------------------
def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def bloom(img, strength=0.15, threshold=0.8, radii=(4, 12, 32, 80)):
    H, W = img.shape[:2]
    S = H / 2160
    bright = np.maximum(img - threshold, 0)
    acc = np.zeros_like(img)
    for rr in radii:
        acc += fast_blur(bright, rr * S)
    return img + acc * (strength / len(radii))


def vignette(img, amount=0.35, power=2.2):
    H, W = img.shape[:2]
    X, Y = coords(W, H)
    r = np.sqrt((X / (W / H)) ** 2 + Y ** 2) / math.sqrt(2)
    v = 1 - amount * r ** power
    return img * v[..., None]


def grain(img, amount=0.012, seed=1):
    r = rng(seed)
    n = r.standard_normal(img.shape[:2], dtype=F32) * amount
    return img + n[..., None]


def finish(img, exposure=1.0, bloom_amt=0.12, bloom_thr=0.9, vig=0.3, grain_amt=0.008,
           tonemap=True, saturation=1.0, contrast=1.0):
    img = img.astype(F32) * F32(exposure)
    if bloom_amt > 0:
        img = bloom(img, bloom_amt, bloom_thr)
    if vig > 0:
        img = vignette(img, vig)
    if tonemap:
        img = aces(img)
    else:
        img = np.clip(img, 0, 1)
    if saturation != 1.0:
        lum = (img * np.array([0.2126, 0.7152, 0.0722], F32)).sum(-1, keepdims=True)
        img = np.clip(lum + (img - lum) * saturation, 0, 1)
    out = linear_to_srgb(img)
    if contrast != 1.0:
        out = np.clip(0.5 + (out - 0.5) * contrast, 0, 1)
    if grain_amt > 0:
        out = grain(out, grain_amt)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def save(img8, path, quality=95):
    Image.fromarray(img8).save(path, quality=quality, subsampling=0, optimize=True)
