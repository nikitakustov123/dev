"""Mathematical / abstract scenes."""
import math
import os
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from common import *


# ----------------------------------------------------------------------------
# 23  Clifford strange attractor
# ----------------------------------------------------------------------------
def attractor(W, H):
    S = H / 2160
    a, b, c, d = -1.7, 1.8, -1.9, -0.4
    rg = rng(401)
    N = 2_000_000
    x = rg.uniform(-1, 1, N)
    y = rg.uniform(-1, 1, N)
    dens = np.zeros(H * W)
    acc = np.zeros((3, H * W))
    xmin, xmax, ymin, ymax = -2.9, 2.9, -1.95, 1.95
    sc = min(W * 0.94 / (xmax - xmin), H * 0.94 / (ymax - ymin))
    ox = W / 2 - (xmin + xmax) / 2 * sc
    oy = H / 2 + (ymin + ymax) / 2 * sc
    steps = 110 if S > 0.5 else 40
    for i in range(steps):
        xn = np.sin(a * y) + c * np.cos(a * x)
        yn = np.sin(b * x) + d * np.cos(b * y)
        if i > 8:
            ang = np.arctan2(yn - y, xn - x)
            spd = np.hypot(xn - x, yn - y)
            px = (xn * sc + ox).astype(np.int64)
            py = (oy - yn * sc).astype(np.int64)
            m = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            idx = py[m] * W + px[m]
            dens += np.bincount(idx, minlength=H * W)
            t = (ang[m] / (2 * np.pi) + 0.5)
            s_ = np.clip(spd[m] / 3.5, 0, 1)
            cols = palette(((t + 0.35 * s_) % 1.0).astype(F32), [(0, (0.1, 0.4, 1.0)), (0.25, (0.0, 0.9, 0.9)), (0.5, (1.0, 0.85, 0.3)),
                                                                  (0.75, (1.0, 0.2, 0.5)), (1, (0.1, 0.4, 1.0))])
            for ch in range(3):
                acc[ch] += np.bincount(idx, weights=cols[:, ch], minlength=H * W)
        x, y = xn, yn
    dens = dens.reshape(H, W)
    col = (acc / np.maximum(dens.reshape(-1), 1)).T.reshape(H, W, 3)
    v = np.log1p(dens) / np.log1p(np.percentile(dens[dens > 0], 99.7))
    v = np.clip(v, 0, 1.5) ** 1.6
    img = (col * v[..., None] * 1.4).astype(F32)
    img = img + blur(img, 2 * S) * 0.4
    X, Y = coords(W, H)
    bg = palette(np.clip(np.sqrt(X * X + Y * Y) / 2, 0, 1), [(0, hexcol('#0b0f24')), (1, hexcol('#010104'))])
    img += bg
    return finish(img, exposure=1.2, bloom_amt=0.3, bloom_thr=0.7, vig=0.3, saturation=1.1)


# ----------------------------------------------------------------------------
# 24  Harmonograph light painting
# ----------------------------------------------------------------------------
def harmonograph(W, H):
    S = H / 2160
    rg = rng(411)
    total = 36_000_000 if S > 0.5 else 6_000_000
    T = 520.0
    buf = np.zeros((H, W, 3), F32)
    p = rg.uniform(0, 2 * np.pi, 6)
    chunk = 3_000_000
    for s0 in range(0, total, chunk):
        t = np.linspace(s0, s0 + chunk, chunk, endpoint=False) / total * T
        e1, e2, e3 = np.exp(-0.0042 * t), np.exp(-0.0031 * t), np.exp(-0.0050 * t)
        x = 0.95 * np.sin(2.0 * t + p[0]) * e1 + 0.55 * np.sin(3.003 * t + p[1]) * e2 + 0.12 * np.sin(7.01 * t + p[4]) * e3
        y = 0.62 * np.sin(3.0 * t + p[2]) * e2 + 0.30 * np.sin(2.002 * t + p[3]) * e1 + 0.08 * np.sin(5.02 * t + p[5]) * e3
        # slow rotation adds depth
        rot = 0.0009 * t
        xr = x * np.cos(rot) - y * np.sin(rot)
        yr = x * np.sin(rot) + y * np.cos(rot)
        px = W / 2 + xr * H * 0.62
        py = H / 2 - yr * H * 0.62
        u = (t / T).astype(F32)
        col = palette(u, [(0, (1.0, 0.85, 0.5)), (0.2, (1.0, 0.3, 0.45)), (0.45, (0.6, 0.2, 1.0)), (0.7, (0.1, 0.6, 1.0)), (1, (0.1, 1.0, 0.8))])
        w = (0.35 + 0.65 * np.exp(-t / T * 2)).astype(F32)[:, None] * 0.006 * (W * H / (3840 * 2160)) * (36_000_000 / total)
        splat(buf, px, py, col * w)
    img = blur(buf, 0.5 * S) * 1.0
    img = img * 0.8 + blur(img, 3 * S) * 1.2 + fast_blur(img, 25 * S) * 2.0
    img += palette(np.clip(np.arange(H, dtype=F32) / H, 0, 1), [(0, (0.002, 0.002, 0.008)), (1, (0.006, 0.002, 0.01))])[:, None, :]
    return finish(img, exposure=1.5, bloom_amt=0.25, bloom_thr=0.7, vig=0.35, saturation=1.15)


# ----------------------------------------------------------------------------
# escape-time fractals (multi-process)
# ----------------------------------------------------------------------------
def _escape_band(args):
    kind, x0, x1, y0, y1, W, h_rows, H, row0, maxit, cj = args
    xs = np.linspace(x0, x1, W)
    ys = np.linspace(y0, y1, H)[row0:row0 + h_rows]
    Xc, Yc = np.meshgrid(xs, ys)
    C = Xc + 1j * Yc
    n = C.size
    Cf = C.ravel()
    if kind == 'mandel':
        z = np.zeros(n, complex)
        c = Cf.copy()
    else:
        z = Cf.copy()
        c = np.full(n, cj)
    dz = np.ones(n, complex) if kind == 'julia' else np.zeros(n, complex)
    trap = np.full(n, 1e9)
    out_n = np.full(n, -1.0)
    out_u = np.zeros(n, complex)
    out_trap = np.zeros(n)
    idx = np.arange(n)
    R2 = 1e6
    for it in range(maxit):
        if kind == 'mandel':
            dz = 2 * z * dz + 1
        else:
            dz = 2 * z * dz
        z = z * z + c
        trap = np.minimum(trap, np.minimum(np.abs(z.real), np.abs(z.imag)))
        a2 = z.real * z.real + z.imag * z.imag
        esc = a2 > R2
        if esc.any():
            e = idx[esc]
            ze = z[esc]
            out_n[e] = it + 1 - np.log2(np.log(np.abs(ze)))
            out_u[e] = ze / dz[esc]
            out_trap[e] = trap[esc]
            keep = ~esc
            z, dz, c, idx, trap = z[keep], dz[keep], c[keep], idx[keep], trap[keep]
            if len(idx) == 0:
                break
    return out_n.reshape(h_rows, W), out_u.reshape(h_rows, W), out_trap.reshape(h_rows, W)


def escape_time(kind, W, H, cx, cy, width, maxit, cj=0j):
    height = width * H / W
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy + height / 2, cy - height / 2
    bands = 32
    rows = [(i * H) // bands for i in range(bands + 1)]
    args = [(kind, x0, x1, y0, y1, W, rows[i + 1] - rows[i], H, rows[i], maxit, cj) for i in range(bands)]
    workers = max(os.cpu_count() or 1, 1)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_escape_band, args))
    n = np.vstack([r[0] for r in res])
    u = np.vstack([r[1] for r in res])
    tr = np.vstack([r[2] for r in res])
    return n, u, tr


def emboss(u, angle=45, h2=1.6):
    un = u / (np.abs(u) + 1e-30)
    v = np.exp(1j * math.radians(angle))
    t = (un.real * v.real + un.imag * v.imag + h2) / (1 + h2)
    return np.clip(t, 0, 1)


# ----------------------------------------------------------------------------
# 25  Mandelbrot deep zoom
# ----------------------------------------------------------------------------
def mandelbrot(W, H):
    S = H / 2160
    n, u, tr = escape_time('mandel', W, H, -0.743643887037151, 0.131825904205330, 0.00042, 2400 if S > 0.5 else 900)
    inside = n < 0
    t = emboss(u, 50, 1.4)
    ns = np.log(np.maximum(n, 1)) * 2.2
    col = palette((ns % 1.0).astype(F32), [(0, hexcol('#06103a')), (0.18, hexcol('#1d5fbf')), (0.36, hexcol('#e8f4ff')),
                                           (0.5, hexcol('#ffb02e')), (0.68, hexcol('#8a1a10')), (0.85, hexcol('#1a0620')), (1, hexcol('#06103a'))])
    img = col * (0.15 + 0.95 * t[..., None] ** 1.5)
    img += (t ** 30)[..., None] * 0.6
    img[inside] = 0
    return finish(img.astype(F32), exposure=1.2, bloom_amt=0.12, bloom_thr=1.0, vig=0.25, tonemap=False, saturation=1.1)


# ----------------------------------------------------------------------------
# 26  Julia set with orbit traps
# ----------------------------------------------------------------------------
def julia(W, H):
    S = H / 2160
    n, u, tr = escape_time('julia', W, H, 0.0, 0.0, 3.3, 1200 if S > 0.5 else 500, cj=complex(-0.7269, 0.1889))
    inside = n < 0
    t = emboss(u, 135, 1.2)
    ns = np.sqrt(np.maximum(n, 0)) * 0.45
    trap = np.exp(-tr * 18)
    col = palette((ns % 1.0).astype(F32), [(0, hexcol('#020818')), (0.3, hexcol('#0c4a6e')), (0.55, hexcol('#3ad1c8')),
                                           (0.75, hexcol('#f7e6b0')), (0.9, hexcol('#c06a2a')), (1, hexcol('#020818'))])
    img = col * (0.1 + 0.95 * t[..., None] ** 2)
    img += trap[..., None] * np.array([1.0, 0.75, 0.35], F32) * 0.6 * t[..., None]
    img += (t ** 40)[..., None] * 0.5
    img[inside] = np.array([0.0, 0.0, 0.0], F32)
    img = img.astype(F32)
    return finish(img, exposure=1.15, bloom_amt=0.2, bloom_thr=0.8, vig=0.3, tonemap=False, saturation=1.1)


# ----------------------------------------------------------------------------
# 27  Reaction-diffusion (Gray-Scott)
# ----------------------------------------------------------------------------
def reaction_diffusion(W, H):
    S = H / 2160
    gw, gh = W // 4, H // 4
    rg = rng(421)
    A = np.ones((gh, gw), F32)
    B = np.zeros((gh, gw), F32)
    for _ in range(int(860 * gw * gh / (960 * 540)) + 20):
        x, y = rg.integers(5, gw - 5), rg.integers(5, gh - 5)
        r = rg.integers(2, 6)
        B[y - r:y + r, x - r:x + r] = 1
    k9 = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]], F32)
    gx = np.linspace(0, 1, gw, dtype=F32)[None, :]
    gy = np.linspace(0, 1, gh, dtype=F32)[:, None]
    warp = cv2.resize(fbm(gw // 4, gh // 4, gh / 8, 4, seed=422), (gw, gh)) * 0.08
    feed = (0.0545 + 0.002 * warp).astype(F32) + 0 * gx
    kill = (0.0605 + 0.003 * gx + 0.0015 * gy + 0.0015 * warp).astype(F32)
    iters = 6000
    for i in range(iters):
        LA = cv2.filter2D(A, -1, k9, borderType=cv2.BORDER_WRAP if False else cv2.BORDER_REFLECT)
        LB = cv2.filter2D(B, -1, k9, borderType=cv2.BORDER_REFLECT)
        ABB = A * B * B
        A += 1.0 * LA - ABB + feed * (1 - A)
        B += 0.5 * LB + ABB - (kill + feed) * B
    h = cv2.resize(B, (W, H), interpolation=cv2.INTER_CUBIC)
    h = blur(h, 1.0 * S)
    gyy, gxx = np.gradient(h)
    k = 90 / S * 0.5
    nx, ny = -gxx * k, -gyy * k
    nl = np.sqrt(nx * nx + ny * ny + 1)
    L = np.array([-0.5, -0.55, 0.67])
    L /= np.linalg.norm(L)
    ndl = np.clip((nx * L[0] + ny * L[1] + L[2]) / nl, 0, 1)
    hv = np.array([-0.25, -0.3, 1.0])
    hv /= np.linalg.norm(hv)
    spec = np.clip((nx * hv[0] + ny * hv[1] + hv[2]) / nl, 0, 1) ** 60
    hn = np.clip(h / max(h.max(), 1e-6), 0, 1)
    X, Y = coords(W, H)
    hue = (hn * 0.9 + 0.15 * X + 0.1 * Y) % 1.0
    col = palette(hue.astype(F32), [(0, hexcol('#0a1a4a')), (0.2, hexcol('#1b7f9a')), (0.4, hexcol('#3fe0c0')), (0.6, hexcol('#f6d36b')),
                                    (0.8, hexcol('#e0457b')), (1, hexcol('#0a1a4a'))])
    base = col * (0.08 + 0.92 * hn[..., None])
    img = base * (0.25 + 0.85 * ndl[..., None]) + spec[..., None] * 0.8
    return finish(img.astype(F32), exposure=1.1, bloom_amt=0.15, bloom_thr=0.8, vig=0.3, saturation=1.1)


# ----------------------------------------------------------------------------
# 28  Black-and-gold marble (multi-level domain warping)
# ----------------------------------------------------------------------------
def marble(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    A = H * 0.12
    q1 = fbm(W, H, H * 0.6, 8, seed=431)
    q2 = fbm(W, H, H * 0.6, 8, seed=432)
    r1s = fbm(W, H, H * 0.45, 9, seed=433)
    r2s = fbm(W, H, H * 0.45, 9, seed=434)
    r1 = sample(r1s, px + A * q1, py + A * q2, border=cv2.BORDER_REFLECT)
    r2 = sample(r2s, px + A * q2 * 1.3, py - A * q1 * 1.3, border=cv2.BORDER_REFLECT)
    fs = fbm(W, H, H * 0.35, 11, gain=0.55, seed=435)
    f = sample(fs, px + A * 1.6 * r1, py + A * 1.6 * r2, border=cv2.BORDER_REFLECT)
    vs = fbm(W, H, H * 0.25, 11, gain=0.55, seed=436, ridged=True)
    v = sample(vs, px + A * 1.2 * r1, py + A * 1.2 * r2, border=cv2.BORDER_REFLECT)
    base = palette(np.clip(f * 0.18 + 0.5, 0, 1), [(0, hexcol('#020303')), (0.4, hexcol('#0a1714')), (0.6, hexcol('#16302a')),
                                                    (0.8, hexcol('#3a5a52')), (1, hexcol('#c8d8d0'))])
    cloud = smoothstep(0.6, 2.6, f) * 0.35
    base = base + cloud[..., None] * np.array([0.4, 0.5, 0.48], F32)
    # gold veins: thin where the ridged field peaks; plus contour lines of f
    vein = smoothstep(2.0, 2.9, v)
    contour = np.exp(-((np.mod(f * 2.2, 1.0) - 0.5) ** 2) / 0.0015) * smoothstep(-0.4, 1.2, f) * 0.8
    gmask = np.clip(vein + contour, 0, 1)
    gy, gx = np.gradient(blur(gmask, 1.5 * S))
    shine = np.clip(0.55 + (-gx - gy) * 12 / S * 0.25, 0, 1.5)
    sparkle = (rng(437).random((H, W)) > 0.9985).astype(F32)
    sparkle = blur(sparkle, 0.8 * S) * 25 * gmask
    gold = np.array([1.0, 0.68, 0.22], F32)
    img = base * (1 - gmask[..., None]) + gmask[..., None] * gold * (0.4 + 1.2 * shine[..., None])
    img += sparkle[..., None] * np.array([1.0, 0.9, 0.7], F32)
    # polished surface reflection
    X, Y = coords(W, H)
    img += (np.exp(-((X + 0.6) ** 2 + (Y - 0.5) ** 2) * 1.2) * 0.04)[..., None]
    return finish(img.astype(F32), exposure=1.2, bloom_amt=0.25, bloom_thr=0.9, vig=0.3)


# ----------------------------------------------------------------------------
# 29  Amethyst geode slice
# ----------------------------------------------------------------------------
def geode(W, H):
    S = H / 2160
    from scipy.spatial import cKDTree
    X, Y = coords(W, H)
    px, py = pixel_grid(W, H)
    rg = rng(441)
    th = np.arctan2(Y, X)
    r = np.sqrt(X * X + Y * Y)
    n_th = noise1d(4096, 300, 442, octaves=5)
    ti = ((th / (2 * np.pi) + 0.5) * 4095).astype(int)
    Rb = 0.86 * (1 + 0.07 * n_th[ti] / 2.5) * np.array(1.0)
    warp = fbm(W, H, H * 0.2, 7, seed=443)
    rn = r / Rb + 0.025 * warp
    # agate bands
    bandn = fbm(W, H, H * 0.05, 6, seed=444)
    bp = rn * 26 + 1.2 * bandn + 0.6 * warp
    bands = 0.5 + 0.5 * np.sin(bp * np.pi)
    fine = 0.5 + 0.5 * np.sin(bp * np.pi * 7.3)
    bcol = palette(np.clip((1 - rn) / 0.45, 0, 1), [(0, hexcol('#3b2a4a')), (0.25, hexcol('#8a6ab8')), (0.45, hexcol('#e8e0f0')),
                                                     (0.7, hexcol('#5a3a9a')), (1, hexcol('#c8b8e8'))])
    agate = bcol * (0.55 + 0.45 * bands[..., None]) * (0.85 + 0.15 * fine[..., None])
    white = smoothstep(0.92, 0.99, bands) * 0.5
    agate = agate + white[..., None] * np.array([0.9, 0.9, 1.0], F32)
    # crystal cavity: Voronoi facets
    cav = rn < 0.55
    nseeds = 5000
    sx = rg.uniform(-1.0, 1.0, nseeds) * 0.62 * W / H
    sy = rg.uniform(-1.0, 1.0, nseeds) * 0.62
    tree = cKDTree(np.stack([sx, sy], 1))
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    dd, ii = tree.query(pts, k=2, workers=-1)
    d1 = dd[:, 0].reshape(H, W)
    d2 = dd[:, 1].reshape(H, W)
    cid = ii[:, 0].reshape(H, W)
    edge = (d2 - d1)
    # each crystal: a pyramid facet pointing toward the cavity centre
    vx = (X - sx[cid])
    vy = (Y - sy[cid])
    fa = np.arctan2(vy, vx)
    facet = np.floor((fa + np.pi) / (2 * np.pi) * 6).astype(int)
    fang = (facet + 0.5) / 6 * 2 * np.pi - np.pi
    tilt = 0.6 * np.clip(np.sqrt(vx * vx + vy * vy) / 0.03, 0, 1)
    nx = np.cos(fang) * tilt
    ny = np.sin(fang) * tilt
    nz = np.ones_like(nx)
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    L = np.array([-0.5, 0.6, 0.62])
    L /= np.linalg.norm(L)
    ndl = np.clip((nx * L[0] + ny * L[1] + nz * L[2]) / nl, 0, 1)
    hv = np.array([-0.25, 0.3, 1.0])
    hv /= np.linalg.norm(hv)
    spec = np.clip((nx * hv[0] + ny * hv[1] + nz * hv[2]) / nl, 0, 1) ** 80
    depth = np.clip(rn / 0.55, 0, 1)
    cshade = np.random.default_rng(445).uniform(0.6, 1.2, nseeds)[cid]
    ccol = palette(depth.astype(F32), [(0, hexcol('#1a0630')), (0.5, hexcol('#5a1a8a')), (0.85, hexcol('#b070e0')), (1, hexcol('#e8d8ff'))])
    crystal = ccol * (0.25 + 0.9 * ndl[..., None]) * cshade[..., None] + spec[..., None] * 2.5
    crystal *= smoothstep(0.0, 0.004, edge)[..., None] * 0.7 + 0.3
    crystal *= (0.35 + 0.65 * depth)[..., None]
    sp = (rng(446).random((H, W)) > 0.9993) & cav
    crystal += blur(sp.astype(F32), 1.0 * S)[..., None] * 30 * spec[..., None] ** 0.2
    rind = (rn > 0.985) & (rn < 1.06)
    rock_t = fbm(W, H, 8 * S, 6, seed=447)
    rock = np.array([0.18, 0.15, 0.13], F32) * (0.6 + 0.3 * rock_t[..., None])
    img = np.zeros((H, W, 3), F32)
    bgc = palette(np.clip(r / 2, 0, 1), [(0, hexcol('#1a1a1e')), (1, hexcol('#050506'))])
    img[:] = bgc
    # drop shadow
    sh = smoothstep(1.1, 0.95, r / Rb)
    img *= (1 - 0.6 * blur(sh, 30 * S))[..., None]
    m_ag = smoothstep(0.995, 0.985, rn) * smoothstep(0.545, 0.555, rn)
    m_rd = smoothstep(1.065, 1.055, rn) * smoothstep(0.98, 0.99, rn)
    m_cv = 1 - smoothstep(0.545, 0.555, rn)
    img = img * (1 - m_rd[..., None]) + rock * m_rd[..., None]
    img = img * (1 - m_ag[..., None]) + agate * m_ag[..., None]
    img = img * (1 - m_cv[..., None]) + crystal * m_cv[..., None]
    # polished-face gloss on agate
    gl = np.exp(-((X + 0.3) ** 2 + (Y - 0.45) ** 2) * 3) * 0.15 * m_ag
    img += gl[..., None]
    return finish(img.astype(F32), exposure=1.15, bloom_amt=0.25, bloom_thr=0.9, vig=0.3, saturation=1.1)


# ----------------------------------------------------------------------------
# 30  Hyperspace jump
# ----------------------------------------------------------------------------
def hyperspace(W, H):
    S = H / 2160
    X, Y = coords(W, H)
    cx, cy = 0.05, 0.02
    dx, dy = X - cx, Y - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-4
    th = np.arctan2(dy, dx)
    N = 2048
    tex = tile_noise(N, N / 16, 8, 451)
    tex2 = tile_noise(N, N / 64, 6, 452)
    v = 1.0 / r
    u = (th / (2 * np.pi) + 0.5) * N
    wall = sample(tex, u + v * 120, v * 90)
    wall2 = sample(tex2, u * 2 - v * 200, v * 220)
    tun = np.exp(0.9 * wall) * (0.5 + 0.5 * np.exp(0.6 * wall2))
    fade = smoothstep(0.03, 0.5, r) * np.exp(-r * 0.6)
    cols = palette(np.clip(0.5 + 0.2 * wall, 0, 1).astype(F32), [(0, hexcol('#1a0a5a')), (0.5, hexcol('#2a6aff')), (1, hexcol('#c050ff'))])
    img = cols * (tun * fade * 0.25)[..., None]
    # star streaks
    rg = rng(453)
    streak = np.zeros((H, W, 3), F32)
    n = 1800
    for i in range(n):
        a = rg.uniform(-np.pi, np.pi)
        r0 = rg.uniform(0.04, 1.6) ** 1.3
        ln = r0 * rg.uniform(0.3, 1.2) + 0.02
        r1 = r0 + ln
        x0 = W / 2 + (cx + r0 * math.cos(a)) * H / 2
        y0 = H / 2 - (cy + r0 * math.sin(a)) * H / 2
        x1 = W / 2 + (cx + r1 * math.cos(a)) * H / 2
        y1 = H / 2 - (cy + r1 * math.sin(a)) * H / 2
        tcol = rg.random()
        c = (0.55, 0.75, 1.0) if tcol < 0.6 else ((0.85, 0.55, 1.0) if tcol < 0.85 else (1.0, 1.0, 1.0))
        b = rg.uniform(0.15, 1.0) * min(1.0, r0 * 3)
        thick = max(int((1 + 2.5 * r0) * S * 1.5), 1)
        cv2.line(streak, (int(x0), int(y0)), (int(x1), int(y1)), tuple(cc * b for cc in c), thick, cv2.LINE_AA)
    img += streak * 1.2 + blur(streak, 3 * S) * 0.8 + fast_blur(streak, 18 * S) * 0.4
    core = np.exp(-r / 0.035) * 8 + np.exp(-r / 0.15) * 1.2
    img += core[..., None] * np.array([0.7, 0.85, 1.0], F32)
    # chromatic aberration (radial)
    px, py = pixel_grid(W, H)
    ccx, ccy = W / 2 + cx * H / 2, H / 2 - cy * H / 2
    out = img.copy()
    for ch, s_ in ((0, 1.006), (2, 0.994)):
        mx = (ccx + (px - ccx) / s_).astype(F32)
        my = (ccy + (py - ccy) / s_).astype(F32)
        out[..., ch] = cv2.remap(img[..., ch], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return finish(out, exposure=1.1, bloom_amt=0.35, bloom_thr=0.7, vig=0.45, saturation=1.15)
