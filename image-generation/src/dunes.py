"""14  Desert dunes at golden hour -- height-field ray marching with
ray-marched sun shadows, wind ripples and aerial perspective."""
import math
import numpy as np
import cv2
from common import *


def dune_heightmap(N=4096, seed=141):
    yy, xx = np.meshgrid(np.arange(N, dtype=F32), np.arange(N, dtype=F32), indexing='ij')
    w1 = tile_noise(N, N / 4, 6, seed)
    w2 = tile_noise(N, N / 8, 6, seed + 1)
    h = np.zeros((N, N), F32)
    for (freq, amp, ang, wamp, sd) in ((9, 1.0, 0.25, 0.07, 0), (20, 0.3, 0.6, 0.04, 1), (45, 0.07, -0.2, 0.03, 2)):
        # tile-safe direction: integer cycles along both axes
        cx_, cz_ = round(freq * math.cos(ang)), round(freq * math.sin(ang))
        u = (xx * cx_ + yy * cz_) / N + wamp * (w1 if sd % 2 == 0 else w2) * (1 + sd)
        s = u - np.floor(u)
        # asymmetric dune: long windward slope, steep slip face, rounded crest
        wind = smoothstep(0.0, 0.78, s)
        slip = 1 - smoothstep(0.78, 1.0, s) ** 0.7
        prof = np.where(s < 0.78, wind, slip)
        prof = prof ** 1.3
        h += amp * prof
    h += 0.12 * tile_noise(N, N / 16, 6, seed + 5)
    h = cv2.GaussianBlur(h, (0, 0), 2.0)
    return h


def dunes(W, H):
    S = H / 2160
    N = 4096
    hm = dune_heightmap(N)
    world = 900.0          # metres covered by the tile
    hs = 16.0              # height scale (m)
    tex_k = N / world
    hmap = hm * hs
    gzm, gxm = np.gradient(hmap)
    gxm *= tex_k
    gzm *= tex_k
    X, Y = coords(W, H)
    f = 1.5
    pitch = math.radians(-11)
    cp, sp = math.cos(pitch), math.sin(pitch)
    dx = X
    dy0 = Y
    dz0 = np.full_like(X, f)
    dy = dy0 * cp + dz0 * sp
    dz = -dy0 * sp + dz0 * cp
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / n, dy / n, dz / n
    ox, oy, oz = 137.0, 0.0, 41.0
    oy = float(hmap.max()) + 22.0

    def hgt(x, z):
        return sample(hmap, x * tex_k, z * tex_k)

    t = np.full(X.shape, 2.0, F32)
    hit = np.zeros(X.shape, bool)
    tmax = 4000.0
    prev_t = t.copy()
    for i in range(260):
        act = ~hit & (t < tmax)
        if not act.any():
            break
        px_ = ox + dx * t
        py_ = oy + dy * t
        pz_ = oz + dz * t
        h = hgt(px_, pz_)
        dh = py_ - h
        newhit = act & (dh < 0)
        hit |= newhit
        step = np.maximum(dh * 0.45, 0.02 + t * 0.004)
        prev_t = np.where(act & ~newhit, t, prev_t)
        t = np.where(act & ~newhit, t + step, t)
    # refine with bisection between prev_t and t
    lo, hi = prev_t.copy(), t.copy()
    for i in range(8):
        mid = (lo + hi) * 0.5
        h = hgt(ox + dx * mid, oz + dz * mid)
        below = (oy + dy * mid) < h
        hi = np.where(hit & below, mid, hi)
        lo = np.where(hit & ~below, mid, lo)
    t = np.where(hit, hi, t)
    pxh, pyh, pzh = ox + dx * t, oy + dy * t, oz + dz * t

    sun = np.array([-0.75, 0.16, 0.62])
    sun /= np.linalg.norm(sun)
    gx = sample(gxm, pxh * tex_k, pzh * tex_k)
    gz = sample(gzm, pxh * tex_k, pzh * tex_k)
    # wind ripples (band-limited by distance)
    footprint = t * 2 / H / f
    rip_k = 2 * np.pi / 0.35
    rw = np.exp(-(rip_k * footprint * 0.5) ** 2) * 0.12
    warp = sample(hm, pxh * tex_k * 3.0, pzh * tex_k * 3.0) * 6
    phase = (pxh * 0.8 + pzh * 0.6) * rip_k + warp * 4
    gx = gx + rw * np.cos(phase) * 0.8 * rip_k * 0.02
    gz = gz + rw * np.cos(phase) * 0.6 * rip_k * 0.02
    nx, ny, nz = -gx, np.ones_like(gx), -gz
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    ndl = np.clip(nx * sun[0] + ny * sun[1] + nz * sun[2], 0, 1)
    # shadow rays
    sh = np.ones(X.shape, F32)
    st = np.full(X.shape, 0.5, F32)
    for i in range(70):
        qx, qy, qz = pxh + sun[0] * st, pyh + 0.05 + sun[1] * st, pzh + sun[2] * st
        d = qy - hgt(qx, qz)
        sh = np.minimum(sh, np.clip(8 * d / st + 0.5, 0, 1))
        st = st + np.maximum(np.abs(d) * 0.5, 0.3 + st * 0.04)
    sh = np.where(hit, sh, 1.0)
    sand = np.array([0.80, 0.42, 0.18], F32)
    alb = sand * (0.9 + 0.1 * sample(hm, pxh * tex_k * 7, pzh * tex_k * 7)[..., None])
    sunc = np.array([1.0, 0.72, 0.45], F32) * 3.2
    skyc = np.array([0.35, 0.45, 0.75], F32) * 0.6
    lit = alb * (sunc * (ndl * sh)[..., None] + skyc * (0.5 + 0.5 * ny)[..., None])
    # rim / sheen on crests facing the sun
    view = -(dx * nx + dy * ny + dz * nz)
    sheen = (1 - np.clip(view, 0, 1)) ** 4 * ndl * sh
    lit += sheen[..., None] * np.array([1.0, 0.7, 0.4], F32) * 0.6

    # sky
    el = np.clip(dy, -0.1, 1)
    sky = palette(np.clip(el / 0.5, 0, 1) ** 0.6, [(0, hexcol('#f6c08a')), (0.25, hexcol('#c9a2a0')), (0.6, hexcol('#5f7fb5')), (1, hexcol('#284a8a'))])
    cosg = dx * sun[0] + dy * sun[1] + dz * sun[2]
    ang = np.arccos(np.clip(cosg, -1, 1))
    sky += (np.exp(-ang / 0.05) * 6 + np.exp(-ang / 0.4) * 0.8 + smoothstep(0.012, 0.009, ang) * 60)[..., None] * np.array([1.0, 0.75, 0.45], F32)
    fog = 1 - np.exp(-t / 3200.0)
    fogc = np.array([1.0, 0.62, 0.38], F32) * 0.85
    col = lit * (1 - fog[..., None]) + fogc * fog[..., None]
    img = np.where(hit[..., None], col, sky).astype(F32)
    return finish(img, exposure=0.85, bloom_amt=0.2, bloom_thr=1.2, vig=0.3, saturation=1.2, contrast=1.08)
