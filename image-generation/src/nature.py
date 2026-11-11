"""Nature scenes."""
import math
import numpy as np
import cv2
from common import *
from landscape import ridge, silhouette_mask, lake_reflection


def radial_blur(img, cx, cy, passes=8, length=0.5):
    """Zoom blur toward (cx,cy) in pixels -- used for god rays.
    Each pass averages the image with a copy scaled toward the centre; the
    scale offsets double every pass so the total streak length is `length`."""
    H, W = img.shape[:2]
    out = img.astype(F32).copy()
    px, py = pixel_grid(W, H)
    for k in range(passes):
        s = 1 - length * (2 ** k) / (2 ** passes)
        mx = (cx + (px - cx) * s).astype(F32)
        my = (cy + (py - cy) * s).astype(F32)
        out = 0.5 * (out + cv2.remap(out, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


# ----------------------------------------------------------------------------
# 08  Aurora borealis over a lake
# ----------------------------------------------------------------------------
def aurora(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    hor = int(H * 0.66)
    t = np.clip(py / hor, 0, 1)
    sky = palette(t, [(0, hexcol('#01030a')), (0.6, hexcol('#04122a')), (1, hexcol('#0b3a4a'))]) * 0.5
    sky += starfield(W, H, int(50000 * W * H / (3840 * 2160)), seed=81, bright=0.6)
    aur = np.zeros((H, W, 3), F32)
    rg = rng(82)
    for k, (yb, amp, br) in enumerate(((0.47, 0.08, 1.0), (0.36, 0.07, 0.5), (0.56, 0.04, 0.4))):
        xs = np.arange(W, dtype=F32)
        base = H * (yb + amp * noise1d(W, W / 1.5, 83 + k, octaves=3) / 2.0) + H * 0.04 * np.sin(xs / W * 2 * np.pi * (1.5 + k) + k)
        # coordinate along the curtain (arc length-ish) for ray structure
        along = np.cumsum(np.sqrt(1 + np.gradient(base) ** 2)).astype(F32)
        ray_n = np.interp(along, np.arange(W, dtype=F32), noise1d(W, 7 * S, 90 + k, octaves=4))
        ray_n2 = np.interp(along, np.arange(W, dtype=F32), noise1d(W, 120 * S, 95 + k, octaves=3))
        h = (base[None, :] - py) / (H * 0.32)  # height above lower edge (0..)
        lower = np.exp(-np.maximum(-h, 0) * 60)
        prof = np.where(h > 0, np.exp(-np.maximum(h, 0) * 3.2) * (1 - np.exp(-np.maximum(h, 0) * 50)), 0)
        rays = (0.55 + 0.45 * np.clip(ray_n[None, :] * 0.5 + 0.5, 0, 1) ** 2)
        mod = np.exp(0.8 * ray_n2)[None, :]
        I = prof * rays * mod * br
        green = np.array([0.15, 1.0, 0.45], F32)
        red = np.array([0.8, 0.15, 0.6], F32)
        cmix = smoothstep(0.15, 0.9, h)[..., None]
        col = green * (1 - cmix) + red * cmix
        edge = np.exp(-np.abs(h) * 120)[..., None] * np.array([0.6, 0.2, 0.9], F32) * 0.5 * mod[..., None]
        aur += (I[..., None] * col + edge * br) * 0.22
    aur = cv2.blur(aur, (1, max(int(9 * S), 1)))
    aur = aur * 0.85 + blur(aur, 2 * S) * 0.15
    sky = sky + aur * (py < hor)[..., None]
    sky += fast_blur(aur, 60 * S) * 0.35
    # snowy mountains
    m1 = ridge(W, hor - 0.09 * H, 0.06 * H, 900 * S, 86, octaves=10, sharp=0.35)
    mk1 = silhouette_mask(H, m1, 1.0)
    snow = fbm(W, H, 30 * S, 8, seed=87)
    below = np.clip(py - m1[None, :], 0, None)
    lit = (np.exp(-below / (4 * S)) * 0.8 + np.exp(-below / (60 * S)) * 0.25) * (0.8 + 0.2 * snow)
    mcol = (np.array([0.003, 0.006, 0.009], F32) + lit[..., None] * np.array([0.04, 0.2, 0.13], F32))
    img = sky * (1 - mk1[..., None]) + mcol * mk1[..., None]
    m2 = ridge(W, hor - 0.015 * H, 0.025 * H, 300 * S, 88, octaves=8)
    tn = noise1d(W, 9 * S, 89, octaves=2)
    m2 = m2 - np.maximum(tn, 0) * 0.012 * H * (noise1d(W, 300 * S, 90) > 0)
    mk2 = silhouette_mask(H, m2, 1.0)
    img = img * (1 - mk2[..., None]) + np.array([0.002, 0.004, 0.006], F32) * mk2[..., None]
    img[hor:] = 0
    img = lake_reflection(img, hor, 89, ripple=4.0, darken=0.7)
    return finish(img, exposure=1.7, bloom_amt=0.3, bloom_thr=0.4, vig=0.35, saturation=1.15)


# ----------------------------------------------------------------------------
# 09  Misty mountain ranges at golden hour
# ----------------------------------------------------------------------------
def misty_mountains(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    sx, sy = W * 0.62, H * 0.40
    t = np.clip(py / H, 0, 1)
    sky = palette(t, [(0, hexcol('#1d3566')), (0.22, hexcol('#5d6fa0')), (0.38, hexcol('#e9a07a')), (0.48, hexcol('#ffcf94')), (1, hexcol('#ffe0b8'))])
    d = np.sqrt((px - sx) ** 2 + (py - sy) ** 2) / H
    sky += (np.exp(-d * 9) * 1.2 + np.exp(-d * 40) * 3 + (d < 0.028) * 6)[..., None] * np.array([1.0, 0.85, 0.6], F32)
    clouds = fbm(W, H, H * 0.35, 8, seed=91)
    cl = smoothstep(0.4, 1.8, clouds) * smoothstep(0.45 * H, 0.05 * H, py)
    sky = sky * (1 - 0.3 * cl[..., None]) + cl[..., None] * np.array([1.0, 0.75, 0.6], F32) * 0.35 * (1 + 2 * np.exp(-d * 5))[..., None]
    img = sky.copy()
    n_layers = 8
    fogc = np.array([0.95, 0.66, 0.52], F32)
    rg = rng(92)
    for i in range(n_layers):
        f = i / (n_layers - 1)
        base = H * (0.44 + 0.50 * f ** 1.2)
        amp = H * (0.06 + 0.10 * f)
        prof = ridge(W, base, amp, (1400 - 900 * f) * S, 93 + i, octaves=10, sharp=0.3)
        if i >= n_layers - 3:
            # pine trees along the near ridges
            ntree = int(W / (14 * S * (1.6 - f)))
            xs = np.sort(rg.uniform(0, W, ntree))
            tm = np.zeros((H, W), np.uint8)
            for x0 in xs:
                y0 = prof[int(min(x0, W - 1))] + 4 * S
                th = rg.uniform(40, 110) * S * (0.6 + f)
                tw = th * rg.uniform(0.18, 0.28)
                pts = np.array([[x0 - tw, y0], [x0, y0 - th], [x0 + tw, y0]], np.int32)
                cv2.fillPoly(tm, [pts], 255, lineType=cv2.LINE_AA)
            tmf = tm.astype(F32) / 255
        else:
            tmf = 0
        m = np.maximum(silhouette_mask(H, prof, 1.0), tmf)
        dark = np.array([0.015, 0.02, 0.045], F32) * (0.5 + 0.8 * (1 - f))
        haze = (1 - f) ** 2.4 * 0.9
        col = dark * (1 - haze) + fogc * haze * 0.85
        # top-edge rim light toward the sun
        rim = np.exp(-np.abs(py - prof[None, :]) / (5 * S)) * np.exp(-np.abs(px - sx) / (0.5 * W)) * (1 - haze)
        if i < n_layers - 3:
            col = col + rim[..., None] * np.array([1.0, 0.7, 0.4], F32) * 0.25
        img = img * (1 - m[..., None]) + col * m[..., None]
        # valley mist below this ridge
        mist_n = fbm(W, H, 300 * S, 6, seed=120 + i)
        mist = smoothstep(prof[None, :] + 0.02 * H, prof[None, :] + 0.16 * H, py) * (0.3 + 0.2 * mist_n) * (0.38 - 0.3 * f)
        mist = np.clip(mist, 0, 1)[..., None]
        img = img * (1 - mist) + fogc * (0.75 + 0.25 * np.exp(-d * 3))[..., None] * mist
    # god rays from the sun
    occl = (img.mean(-1) > 0.6).astype(F32) * np.exp(-d * 3)
    rays = radial_blur(np.dstack([occl] * 3), sx, sy, passes=8, length=0.6)
    img += rays * np.array([1.0, 0.8, 0.55], F32) * 0.4
    # birds
    for k in range(9):
        bx, by = W * rg.uniform(0.25, 0.5), H * rg.uniform(0.18, 0.32)
        s = rg.uniform(6, 12) * S
        pts = np.array([[bx - s, by - s * 0.3], [bx - s * 0.4, by - s * 0.45], [bx, by], [bx + s * 0.4, by - s * 0.45], [bx + s, by - s * 0.3]], np.int32)
        cv2.polylines(img, [pts], False, (0.05, 0.05, 0.08), max(int(2 * S), 1), lineType=cv2.LINE_AA)
    return finish(img, exposure=0.95, bloom_amt=0.2, bloom_thr=0.9, vig=0.3, saturation=1.05)


# ----------------------------------------------------------------------------
# 11  Forest with god rays
# ----------------------------------------------------------------------------
def pine(mask, x0, yb, h, w, rg, S):
    """Jagged conifer silhouette drawn into a uint8 mask."""
    tiers = int(rg.integers(9, 15))
    left, right = [], []
    top = yb - h
    for k in range(tiers + 1):
        t = k / tiers
        y = top + t * h * 0.92
        ww = w * (0.08 + 0.92 * t ** 0.9) * rg.uniform(0.75, 1.15)
        droop = h * 0.025
        left += [(x0 - ww, y + droop), (x0 - ww * 0.35, y - h * 0.01)]
        right += [(x0 + ww, y + droop), (x0 + ww * 0.35, y - h * 0.01)]
    pts = [(x0, top - h * 0.03)] + right + [(x0 + w * 0.06, yb), (x0 - w * 0.06, yb)] + left[::-1]
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255, lineType=cv2.LINE_AA)


def forest(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    sx, sy = W * 0.56, H * 0.16
    d = np.sqrt((px - sx) ** 2 + (py - sy) ** 2) / H
    t = np.clip(py / H, 0, 1)
    sky = palette(t, [(0, hexcol('#8aa0b8')), (0.35, hexcol('#e9d6b0')), (0.6, hexcol('#f3e2c0')), (1, hexcol('#c8c0a8'))])
    sky = sky * 0.55 + (np.exp(-d * 6) * 0.8 + np.exp(-d * 30) * 3 + (d < 0.02) * 8)[..., None] * np.array([1.0, 0.88, 0.65], F32)
    img = sky.copy()
    fogc = np.array([0.62, 0.55, 0.42], F32)
    rg = rng(112)
    n_layers = 8
    sky_vis = np.ones((H, W), F32)
    for i in range(n_layers):
        f = i / (n_layers - 1)
        yb = H * (0.56 + 0.5 * f ** 1.3)
        hmin, hmax = H * (0.08 + 0.45 * f), H * (0.16 + 0.75 * f)
        count = int(W / (S * (40 + 160 * f)))
        mask = np.zeros((H, W), np.uint8)
        for _ in range(count):
            x0 = rg.uniform(-0.05, 1.05) * W
            h = rg.uniform(hmin, hmax)
            pine(mask, x0, yb + rg.uniform(-0.01, 0.02) * H, h, h * rg.uniform(0.16, 0.24), rg, S)
        ground = smoothstep(yb - 0.01 * H, yb + 0.01 * H, py)
        m = np.maximum(mask.astype(F32) / 255, ground)
        haze = (1 - f) ** 1.7 * 0.93
        dark = np.array([0.012, 0.022, 0.016], F32)
        col = dark * (1 - haze) + fogc * haze * (0.65 + 0.35 * np.exp(-d * 2))[..., None]
        img = img * (1 - m[..., None]) + col * m[..., None]
        sky_vis *= (1 - m)
        mist = smoothstep(yb - 0.12 * H, yb + 0.02 * H, py) * (0.35 - 0.25 * f) * (0.7 + 0.3 * fbm(W, H, 400 * S, 5, seed=200 + i))
        mist = np.clip(mist, 0, 1)[..., None]
        img = img * (1 - mist) + fogc * mist
    # volumetric god rays through gaps in the canopy
    src = (sky_vis * np.exp(-d * 1.2) * (1 + 3 * np.exp(-d * 8)))[..., None] * np.ones(3, F32)
    rays = radial_blur(src, sx, sy, passes=9, length=0.7)
    img += rays * np.array([1.0, 0.85, 0.6], F32) * 0.9
    # dust motes in the light
    n = int(6000 * W * H / (3840 * 2160))
    x = rg.uniform(0, W, n)
    y = rg.uniform(0, H, n)
    lv = rays[np.clip(y.astype(int), 0, H - 1), np.clip(x.astype(int), 0, W - 1), 0]
    buf = np.zeros((H, W, 3), F32)
    splat(buf, x, y, (np.ones((n, 3)) * (lv * rg.uniform(0.3, 1.5, n))[:, None]).astype(F32))
    img += blur(buf, 1.2 * S) * 4
    return finish(img, exposure=1.0, bloom_amt=0.25, bloom_thr=0.9, vig=0.35, saturation=1.1)


# ----------------------------------------------------------------------------
# 12  Aerial terrain (satellite view)
# ----------------------------------------------------------------------------
def aerial_terrain(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    q1 = fbm(W, H, H * 0.5, 6, seed=151)
    q2 = fbm(W, H, H * 0.5, 6, seed=152)
    base = fbm(W, H, H * 0.7, 11, gain=0.5, seed=153)
    rid = fbm(W, H, H * 0.35, 12, gain=0.52, seed=154, ridged=True)
    ridw = sample(rid, px + q1 * H * 0.08, py + q2 * H * 0.08, border=cv2.BORDER_REFLECT)
    h = base * 0.55 + ridw * 0.6 * smoothstep(-0.6, 1.2, base)
    lo, hi = np.percentile(h, [3, 99.7])
    h = (h - lo) / (hi - lo) * 2.5 - 0.55
    # cavities / erosion gullies
    det = fbm(W, H, 25 * S, 6, seed=155, ridged=True)
    h = h + 0.06 * det * smoothstep(0, 1, h)
    sea = 0.0
    gy, gx = np.gradient(blur(h, 0.8 * S))
    k = 25.0 / S
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(h)
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    L = np.array([-0.6, -0.55, 0.58])
    L /= np.linalg.norm(L)
    shade = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    slope = 1 - nz
    hn = np.clip(h / 2.5, 0, 1)
    land = palette(hn, [(0, hexcol('#d8c99a')), (0.03, hexcol('#7f9a4a')), (0.2, hexcol('#4f7a35')), (0.38, hexcol('#3f5f30')), (0.5, hexcol('#7b7258')),
                        (0.62, hexcol('#8a8478')), (0.75, hexcol('#f4f6fa')), (1, hexcol('#ffffff'))])
    rock = hexcol('#6a645a')
    land = land * (1 - smoothstep(0.25, 0.6, slope)[..., None]) + rock * smoothstep(0.25, 0.6, slope)[..., None] * (1 - smoothstep(0.62, 0.78, hn))[..., None]
    snow = smoothstep(0.62, 0.72, hn + 0.1 * det) * (1 - smoothstep(0.55, 0.8, slope))
    land = land * (1 - snow[..., None]) + np.array([0.95, 0.97, 1.0], F32) * snow[..., None]
    veg = fbm(W, H, 12 * S, 5, seed=156)
    land *= (0.85 + 0.15 * veg[..., None])
    cav = h - blur(h, 6 * S)
    ao = np.clip(1 + cav * 3, 0.5, 1.2)
    col = land * (0.25 + 0.95 * shade[..., None]) * ao[..., None]
    # water
    depth = np.clip(sea - h, 0, None)
    water = palette(np.clip(depth / 0.9, 0, 1), [(0, hexcol('#3fb7b0')), (0.15, hexcol('#1b7a96')), (1, hexcol('#06203a'))])
    wn = fbm(W, H, 300 * S, 6, seed=157)
    water *= (0.92 + 0.08 * wn[..., None])
    foam = np.exp(-depth / 0.012) * (h < sea) * (0.6 + 0.4 * fbm(W, H, 4 * S, 3, seed=158))
    wm = smoothstep(sea + 0.005, sea - 0.005, h)
    col = col * (1 - wm[..., None]) + (water + foam[..., None] * 0.6) * wm[..., None]
    # clouds and their shadows
    cl = fbm(W, H, H * 0.18, 11, gain=0.6, seed=159)
    cmask = smoothstep(1.7, 2.8, cl) * 0.9
    sh = np.roll(np.roll(cmask, int(60 * S), 0), int(70 * S), 1)
    col *= (1 - 0.55 * sh[..., None])
    col = col * (1 - cmask[..., None]) + np.array([1.0, 1.0, 1.0], F32) * (0.85 + 0.15 * cl[..., None] / 3) * cmask[..., None]
    return finish(col, exposure=1.05, bloom_amt=0.05, bloom_thr=1.0, vig=0.25, tonemap=False, saturation=1.1, contrast=1.05)


# ----------------------------------------------------------------------------
# 13  Underwater light and caustics
# ----------------------------------------------------------------------------
def caustic(x, y, it=5, inten=0.005):
    TAU = 6.28318530718
    px_ = np.mod(x * TAU, TAU) - 250.0
    py_ = np.mod(y * TAU, TAU) - 250.0
    ix, iy = px_.copy(), py_.copy()
    c = np.ones_like(x)
    t0 = 1.7
    for n in range(it):
        t = t0 * (1.0 - (3.5 / float(n + 1)))
        ix_n = px_ + np.cos(t - ix) + np.sin(t + iy)
        iy_n = py_ + np.sin(t - iy) + np.cos(t + ix)
        ix, iy = ix_n, iy_n
        c += 1.0 / np.sqrt((px_ / (np.sin(ix + t) / inten)) ** 2 + (py_ / (np.cos(iy + t) / inten)) ** 2)
    c /= it
    c = 1.17 - c ** 1.4
    return np.clip(np.abs(c), 0, None) ** 8


def underwater(W, H):
    S = H / 2160
    X, Y = coords(W, H)
    px, py = pixel_grid(W, H)
    t = np.clip(py / H, 0, 1)
    img = palette(t, [(0, hexcol('#6fe0e8')), (0.25, hexcol('#1b8db0')), (0.6, hexcol('#0a3f6e')), (1, hexcol('#041a33'))])
    # light shafts from the surface
    ang = np.arctan2(X - 0.2, (1.6 - Y))
    nr = noise1d(4096, 18, 161, octaves=4)
    ai = np.clip((ang + 1.2) / 2.4 * 4095, 0, 4095).astype(int)
    shafts = np.clip(nr[ai] * 0.5 + 0.3, 0, None) ** 2 * smoothstep(1.0, -0.6, -Y) * np.exp(-(1 - Y) * 0.6)
    img += shafts[..., None] * np.array([0.6, 0.95, 1.0], F32) * 0.35
    # sea floor in perspective
    hor = 0.62 * H
    fl = py > hor
    z = (H * 0.5) / np.maximum(py - hor, 1)
    fx = X * z * 0.9
    fz = z * 1.2
    floor_n = fbm(W, H, 120 * S, 7, seed=162)
    sand = palette(np.clip(0.5 + 0.15 * floor_n, 0, 1), [(0, hexcol('#9c8a64')), (1, hexcol('#e0cfa0'))])
    ca = caustic(fx * 0.9, fz * 0.9)
    ca2 = caustic(fx * 0.9 + 0.013, fz * 0.9 + 0.007)
    ca = blur(ca, 0.5 * S) * np.exp(-(z - 0.5) * 0.25)[..., None][..., 0]
    floor_col = sand * (0.35 + 1.4 * np.clip(ca, 0, 3)[..., None] * np.array([0.9, 1.0, 1.0], F32) + 0.5 * np.clip(ca2, 0, 3)[..., None] * np.array([0.1, 0.3, 0.5], F32))
    fog = 1 - np.exp(-z * 0.18)
    floor_col = floor_col * (1 - fog[..., None]) + img * fog[..., None]
    fblend = smoothstep(hor, hor + 6 * S, py)
    img = img * (1 - fblend[..., None]) + floor_col * fblend[..., None]
    # rocks and coral silhouettes
    rg = rng(163)
    for side in (0, 1):
        prof = ridge(W, H * (0.80 if side == 0 else 0.86), H * 0.08, 500 * S, 164 + side, octaves=9)
        xs = np.arange(W)
        cut = (xs / W) if side == 0 else (1 - xs / W)
        prof = prof + smoothstep(0.15, 0.45, cut) * H * 0.6
        m = silhouette_mask(H, prof, 1.0)
        tex = fbm(W, H, 15 * S, 6, seed=170 + side)
        top_lit = np.exp(-np.clip(py - prof[None, :], 0, None) / (25 * S))
        rc = np.array([0.04, 0.06, 0.07], F32) * (1 + 0.5 * tex[..., None]) + top_lit[..., None] * np.array([0.1, 0.25, 0.25], F32)
        dist_fog = 0.25 if side == 0 else 0.1
        rc = rc * (1 - dist_fog) + img * dist_fog
        img = img * (1 - m[..., None]) + rc * m[..., None]
        # coral fans / anemone dots
        for _ in range(25):
            x0 = rg.uniform(0, W)
            if (x0 / W if side == 0 else 1 - x0 / W) > 0.25:
                continue
            y0 = prof[int(x0)]
            col = rg.choice([[1.0, 0.35, 0.3], [1.0, 0.6, 0.2], [0.8, 0.3, 0.8], [1.0, 0.85, 0.3]])
            for b in range(40):
                a = rg.uniform(-2.6, -0.5)
                ln = rg.uniform(20, 90) * S
                x1, y1 = x0 + math.cos(a) * ln, y0 + math.sin(a) * ln
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), tuple(float(c) * 0.5 for c in col), max(int(2 * S), 1), cv2.LINE_AA)
                cv2.circle(img, (int(x1), int(y1)), max(int(3 * S), 1), tuple(float(c) * 0.9 for c in col), -1, cv2.LINE_AA)
    # school of fish
    n = 900
    th = rg.uniform(0, 2 * np.pi, n)
    rr = rg.normal(1, 0.18, n)
    cx, cy = W * 0.62, H * 0.38
    fxs = cx + np.cos(th) * rr * W * 0.16
    fys = cy + np.sin(th) * rr * H * 0.12 + rg.normal(0, 0.02, n) * H
    for i in range(n):
        s = rg.uniform(14, 24) * S * (0.7 + 0.5 * rr[i])
        dirx, diry = -np.sin(th[i]), np.cos(th[i]) * 0.75
        nrm = math.hypot(dirx, diry)
        dirx, diry = dirx / nrm, diry / nrm
        ang = math.degrees(math.atan2(diry, dirx))
        shade = 0.35 + 0.25 * math.sin(th[i] * 3)
        c = (0.08 * shade, 0.18 * shade, 0.25 * shade)
        cv2.ellipse(img, (int(fxs[i]), int(fys[i])), (int(s), max(int(s * 0.32), 1)), ang, 0, 360, c, -1, cv2.LINE_AA)
        tx, ty = fxs[i] - dirx * s * 1.1, fys[i] - diry * s * 1.1
        pts = np.array([[fxs[i] - dirx * s * 0.8, fys[i] - diry * s * 0.8], [tx - diry * s * 0.4, ty + dirx * s * 0.4], [tx + diry * s * 0.4, ty - dirx * s * 0.4]], np.int32)
        cv2.fillPoly(img, [pts], c, cv2.LINE_AA)
        cv2.ellipse(img, (int(fxs[i]), int(fys[i] - s * 0.08)), (int(s * 0.8), max(int(s * 0.1), 1)), ang, 180, 360, (0.5 * shade, 0.7 * shade, 0.75 * shade), -1, cv2.LINE_AA)
    # snell's window at the top
    sw = caustic(X * 0.6, Y * 0.3 + 3.0, it=4, inten=0.01)
    img += (np.clip(sw, 0, 2) * smoothstep(0.82, 1.0, Y))[..., None] * np.array([0.8, 1.0, 1.0], F32) * 0.6
    # marine snow
    m = int(4000 * W * H / (3840 * 2160))
    buf = np.zeros((H, W, 3), F32)
    splat(buf, rg.uniform(0, W, m), rg.uniform(0, H, m), np.full((m, 3), 0.25, F32) * rg.uniform(0.2, 1, (m, 1)).astype(F32))
    img += blur(buf, 1.2 * S) * 5 + fast_blur(buf, 8 * S) * 20
    return finish(img, exposure=1.15, bloom_amt=0.25, bloom_thr=0.8, vig=0.45, saturation=1.1)


# ----------------------------------------------------------------------------
# 15  Lightning storm
# ----------------------------------------------------------------------------
def bolt_path(p0, p1, rg, depth, disp):
    pts = [p0, p1]
    for d in range(depth):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            ln = math.hypot(b[0] - a[0], b[1] - a[1])
            nx, ny = -(b[1] - a[1]) / (ln + 1e-6), (b[0] - a[0]) / (ln + 1e-6)
            off = rg.normal(0, disp * ln)
            new += [(mx + nx * off, my + ny * off), b]
        pts = new
    return pts


def lightning(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    rg = rng(181)
    bx0, by0 = W * 0.47, -0.02 * H
    bx1, by1 = W * 0.55, H * 0.80
    main = bolt_path((bx0, by0), (bx1, by1), rg, 9, 0.22)
    mask = np.zeros((H, W), F32)
    cv2.polylines(mask, [np.array(main, np.int32)], False, 1.0, max(int(4 * S), 1), cv2.LINE_AA)
    branches = []
    for k in range(18):
        i = rg.integers(len(main) // 10, len(main) * 3 // 4)
        a = main[i]
        ang = math.atan2(main[min(i + 8, len(main) - 1)][1] - a[1], main[min(i + 8, len(main) - 1)][0] - a[0]) + rg.choice([-1, 1]) * rg.uniform(0.4, 1.1)
        ln = rg.uniform(0.08, 0.3) * H
        b = (a[0] + math.cos(ang) * ln, a[1] + math.sin(ang) * ln)
        br = bolt_path(a, b, rg, 7, 0.25)
        cv2.polylines(mask, [np.array(br, np.int32)], False, 0.55, max(int(2 * S), 1), cv2.LINE_AA)
        for k2 in range(2):
            j = rg.integers(1, len(br) - 1)
            ang2 = ang + rg.uniform(-0.9, 0.9)
            ln2 = ln * rg.uniform(0.2, 0.5)
            c = (br[j][0] + math.cos(ang2) * ln2, br[j][1] + math.sin(ang2) * ln2)
            br2 = bolt_path(br[j], c, rg, 6, 0.25)
            cv2.polylines(mask, [np.array(br2, np.int32)], False, 0.3, max(int(1 * S), 1), cv2.LINE_AA)
    # clouds lit from the bolt
    cl = fbm(W, H, H * 0.45, 11, seed=182)
    cl2 = fbm(W, H, H * 0.12, 9, seed=183, ridged=True)
    cl3 = fbm(W, H, H * 0.06, 8, seed=185)
    dens = smoothstep(-1.0, 1.6, cl + 0.25 * cl3) * smoothstep(0.78 * H, 0.1 * H, py)
    d = np.sqrt((px - W * 0.5) ** 2 + (py - H * 0.18) ** 2) / H
    light = np.exp(-d * 3.0) * 1.5 + 0.06
    sky = palette(np.clip(py / H, 0, 1), [(0, hexcol('#0b0d1a')), (0.6, hexcol('#1b1f38')), (1, hexcol('#2a2540'))]) * 0.5
    cloudc = np.array([0.55, 0.5, 0.85], F32)
    edge = np.clip(dens - blur(dens, 10 * S), 0, None)
    shade = np.clip(0.5 + 0.5 * (cl3 - blur(cl3, 25 * S)) * 2, 0, 1.5)
    img = sky * (1 - dens[..., None] * 0.6) + (dens * light * (0.3 + 0.7 * shade))[..., None] * cloudc * 0.5
    img += (edge * light * 2)[..., None] * np.array([0.8, 0.8, 1.0], F32)
    # the bolt
    bolt = mask[..., None] * np.array([1.0, 0.95, 1.0], F32) * 8
    glow1 = blur(mask, 5 * S)[..., None] * np.array([0.7, 0.6, 1.0], F32) * 4
    glow2 = fast_blur(mask, 50 * S)[..., None] * np.array([0.5, 0.45, 1.0], F32) * 6
    img += bolt + glow1 + glow2
    # landscape with town lights
    hor = H * 0.80
    prof = ridge(W, hor, 0.03 * H, 900 * S, 184, octaves=8)
    gm = silhouette_mask(H, prof, 1.0)
    ground = np.array([0.006, 0.006, 0.01], F32) + (np.exp(-np.abs(px - bx1) / (0.25 * W)) * np.exp(-(py - hor) / (0.08 * H)) * 0.06)[..., None] * np.array([0.6, 0.6, 1.0], F32)
    img = img * (1 - gm[..., None]) + ground * gm[..., None]
    n = int(1800 * W * H / (3840 * 2160))
    lx = rg.uniform(0.05, 0.95, n) * W
    ly = hor + np.abs(rg.normal(0, 0.05, n)) * H + 0.01 * H
    lc = np.array([[1.0, 0.7, 0.35]], F32) * rg.uniform(0.2, 1.0, (n, 1)).astype(F32)
    buf = np.zeros((H, W, 3), F32)
    splat(buf, lx, ly, lc)
    img += (blur(buf, 0.8 * S) * 2 + blur(buf, 3 * S) * 3) * gm[..., None]
    # rain
    rain = np.zeros((H, W), F32)
    for _ in range(int(3000 * W / 3840)):
        x0, y0 = rg.uniform(-0.1 * W, W), rg.uniform(0, H)
        ln = rg.uniform(30, 90) * S
        cv2.line(rain, (int(x0), int(y0)), (int(x0 + ln * 0.18), int(y0 + ln)), rg.uniform(0.02, 0.08), 1, cv2.LINE_AA)
    img += (rain * (0.4 + light))[..., None] * np.array([0.7, 0.7, 1.0], F32)
    return finish(img, exposure=1.1, bloom_amt=0.3, bloom_thr=0.8, vig=0.4)


# ----------------------------------------------------------------------------
# 16  Sakura tree
# ----------------------------------------------------------------------------
def sakura(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    rg = rng(191)
    t = np.clip(py / H, 0, 1)
    bg = palette(t, [(0, hexcol('#9cc3e6')), (0.55, hexcol('#f6d6e0')), (0.8, hexcol('#ffe9dc')), (1, hexcol('#f2c9b8'))])
    d = np.sqrt((px - W * 0.78) ** 2 + (py - H * 0.25) ** 2) / H
    bg += (np.exp(-d * 4) * 0.6)[..., None] * np.array([1.0, 0.9, 0.75], F32)
    # distant bokeh blossoms
    bok = np.zeros((H, W), np.uint8)
    for _ in range(220):
        cv2.circle(bok, (int(rg.uniform(0, W)), int(rg.uniform(0, H * 0.8))), int(rg.uniform(15, 70) * S), int(rg.uniform(20, 70)), -1, cv2.LINE_AA)
    bok = blur(bok.astype(F32) / 255, 6 * S)
    img = bg + bok[..., None] * np.array([1.0, 0.6, 0.75], F32) * 0.8
    # distant hills
    hill = ridge(W, H * 0.80, H * 0.05, 1500 * S, 192, octaves=6)
    hm = silhouette_mask(H, hill, 1.5)
    img = img * (1 - hm[..., None]) + hm[..., None] * np.array([0.55, 0.6, 0.7], F32) * (0.8 + 0.2 * t[..., None])
    hill2 = ridge(W, H * 0.9, H * 0.03, 1200 * S, 193, octaves=6)
    hm2 = silhouette_mask(H, hill2, 1.5)
    gtex = fbm(W, H, 6 * S, 6, seed=194)
    grass = np.array([0.12, 0.24, 0.07], F32) * (0.85 + 0.25 * gtex[..., None]) * (0.7 + 0.5 * smoothstep(0.85 * H, H, py))[..., None]
    img = img * (1 - hm2[..., None]) + grass * hm2[..., None]

    # tree on uint8 layers (anti-aliased drawing)
    wood = np.zeros((H, W), np.uint8)
    flowers = []

    def branch(x, y, ang, ln, th, depth):
        if depth == 0 or ln < 6 * S:
            for _ in range(rg.integers(10, 20)):
                flowers.append((x + rg.normal(0, 30 * S), y + rg.normal(0, 26 * S)))
            return
        segs = 6
        cx, cy, a = x, y, ang
        for s in range(segs):
            a += rg.normal(0, 0.12)
            nx, ny = cx + math.cos(a) * ln / segs, cy - math.sin(a) * ln / segs
            tt = th * (1 - 0.35 * s / segs)
            cv2.line(wood, (int(cx), int(cy)), (int(nx), int(ny)), 255, max(int(tt), 1), cv2.LINE_AA)
            cx, cy = nx, ny
            if depth <= 3 and rg.random() < 0.5:
                flowers.append((cx + rg.normal(0, 18 * S), cy + rg.normal(0, 15 * S)))
        nb = 2 if rg.random() < 0.5 else 3
        spread = [-0.55, 0.5, 0.05]
        for i in range(nb):
            da = spread[i] * rg.uniform(0.6, 1.2) + rg.normal(0, 0.08)
            # bias branches back toward vertical / outward for a rounded crown
            branch(cx, cy, a + da - 0.12 * (a - math.radians(90)), ln * rg.uniform(0.7, 0.8), th * 0.64, depth - 1)

    branch(W * 0.42, H * 1.02, math.radians(88), H * 0.22, 95 * S, 10)
    woodf = wood.astype(F32) / 255
    bark = fbm(W, H, 5 * S, 5, seed=195)
    wcol = np.array([0.06, 0.035, 0.03], F32) * (1 + 0.4 * bark[..., None]) + (np.clip(px - W * 0.36, 0, None) / W)[..., None] * 0.03
    img = img * (1 - woodf[..., None]) + wcol * woodf[..., None]
    # blossoms
    fl = np.zeros((H, W, 3), np.uint8)
    fa = np.zeros((H, W), np.uint8)
    pal = [(255, 183, 197), (255, 205, 215), (250, 160, 185), (255, 228, 234), (240, 140, 170)]
    for (x, y) in flowers:
        if not (0 <= x < W and 0 <= y < H):
            continue
        r = rg.uniform(5, 10) * S
        base = np.array(pal[rg.integers(0, len(pal))], float) * rg.uniform(0.8, 1.05)
        rot = rg.uniform(0, 72)
        for p in range(5):
            a = math.radians(rot + p * 72)
            cxp, cyp = x + math.cos(a) * r * 0.9, y + math.sin(a) * r * 0.9
            c = tuple(int(min(v, 255)) for v in base * rg.uniform(0.9, 1.05))
            cv2.ellipse(fl, (int(cxp), int(cyp)), (max(int(r * 0.85), 1), max(int(r * 0.6), 1)), math.degrees(a), 0, 360, c, -1, cv2.LINE_AA)
            cv2.ellipse(fa, (int(cxp), int(cyp)), (max(int(r * 0.85), 1), max(int(r * 0.6), 1)), math.degrees(a), 0, 360, 255, -1, cv2.LINE_AA)
        cv2.circle(fl, (int(x), int(y)), max(int(r * 0.3), 1), (200, 60, 100), -1, cv2.LINE_AA)
    flc = srgb_to_linear(fl.astype(F32) / 255)
    faf = fa.astype(F32) / 255
    # light from the upper right
    lit = 0.75 + 0.35 * np.clip((px - W * 0.2) / W + (H * 0.6 - py) / H, 0, 1)
    img = img * (1 - faf[..., None]) + flc * lit[..., None] * faf[..., None]
    img += fast_blur(faf, 30 * S)[..., None] * np.array([1.0, 0.6, 0.7], F32) * 0.12
    # falling petals with motion blur
    pet = np.zeros((H, W), np.uint8)
    for _ in range(600):
        x, y = rg.uniform(0, W), rg.uniform(0, H)
        r = rg.uniform(4, 9) * S
        cv2.ellipse(pet, (int(x), int(y)), (max(int(r), 1), max(int(r * 0.55), 1)), rg.uniform(0, 180), 0, 360, 255, -1, cv2.LINE_AA)
    k = np.zeros((int(15 * S) | 1, int(15 * S) | 1), F32)
    cv2.line(k, (0, 0), (k.shape[1] - 1, k.shape[0] - 1), 1.0, 1)
    k /= k.sum()
    petf = cv2.filter2D(pet.astype(F32) / 255, -1, k)
    img = img * (1 - petf[..., None]) + petf[..., None] * np.array([1.0, 0.7, 0.78], F32)
    return finish(img, exposure=1.0, bloom_amt=0.12, bloom_thr=0.9, vig=0.25, tonemap=True, saturation=1.05)


# ----------------------------------------------------------------------------
# 17  Snowflake (Reiter crystal growth model)
# ----------------------------------------------------------------------------
def snowflake(W, H):
    S = H / 2160
    n = 501 if S > 0.5 else 301
    alpha, beta, gamma = 1.0, 0.4, 0.0001
    s = np.full((n, n), beta, np.float64)
    c = n // 2
    s[c, c] = 1.0
    # hexagonal neighbourhood in axial coordinates
    def nsum(a):
        return (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) +
                np.roll(np.roll(a, 1, 0), -1, 1) + np.roll(np.roll(a, -1, 0), 1, 1))
    ii, jj = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    hexdist = np.maximum(np.maximum(np.abs(ii), np.abs(jj)), np.abs(ii + jj))
    limit = int(n * 0.46)
    for it in range(40000):
        frozen = s >= 1
        rec = frozen | (nsum(frozen.astype(np.float64)) > 0)
        u = np.where(rec, 0.0, s)
        v = np.where(rec, s + gamma, 0.0)
        u = u + alpha / 2 * (nsum(u) / 6 - u)
        s = u + v
        s[hexdist >= n // 2 - 1] = beta
        if it % 200 == 0 and (hexdist[s >= 1].max() if (s >= 1).any() else 0) >= limit:
            break
    ice = np.clip(s - 1, 0, None) * (s >= 1)
    frozen = (s >= 1).astype(F32)
    # axial -> cartesian render
    size = int(H * 0.95)
    sc = size / n
    sq = math.sqrt(3) / 2
    # dst = A * src: x = sc*(j + i/2), y = sc*(i*sq)
    M = np.array([[sc, sc * 0.5, 0], [0, sc * sq, 0]], np.float64)
    M[0, 2] = W / 2 - (sc * c + sc * 0.5 * c)
    M[1, 2] = H / 2 - sc * sq * c
    def warp(a):
        return cv2.warpAffine(a.astype(F32), M, (W, H), flags=cv2.INTER_CUBIC, borderValue=0)
    fz = np.clip(warp(frozen), 0, 1)
    th = warp(ice)
    th = blur(th, 1.0 * S)
    fz = blur(fz, 0.8 * S)
    gy, gx = np.gradient(blur(th * 4 + fz * 0.5, 2 * S))
    k = 30
    nx, ny = -gx * k, -gy * k
    nl = np.sqrt(nx * nx + ny * ny + 1)
    L = np.array([-0.5, -0.6, 0.62])
    L /= np.linalg.norm(L)
    diff = (nx * L[0] + ny * L[1] + L[2]) / nl
    spec = np.clip(diff, 0, 1) ** 40
    X, Y = coords(W, H)
    rr = np.sqrt(X * X + Y * Y)
    bg = palette(np.clip(rr / 1.6, 0, 1), [(0, hexcol('#123a6a')), (0.5, hexcol('#071a38')), (1, hexcol('#02060f'))])
    rg = rng(201)
    bok = np.zeros((H, W), np.uint8)
    for _ in range(160):
        cv2.circle(bok, (int(rg.uniform(0, W)), int(rg.uniform(0, H))), int(rg.uniform(10, 60) * S), int(rg.uniform(15, 70)), -1, cv2.LINE_AA)
    bg += blur(bok.astype(F32) / 255, 4 * S)[..., None] * np.array([0.4, 0.7, 1.0], F32) * 0.1
    edges = np.clip(np.abs(gx) + np.abs(gy), 0, None) * 6
    icecol = (np.array([0.35, 0.6, 0.95], F32) * (0.3 + 0.7 * np.clip(diff, 0, 1))[..., None]
              + np.clip(th * 1.5, 0, 1)[..., None] * np.array([0.6, 0.85, 1.0], F32) * 0.6
              + spec[..., None] * 3 + edges[..., None] * np.array([0.7, 0.9, 1.0], F32))
    img = bg * (1 - fz[..., None] * 0.85) + icecol * fz[..., None]
    img += fast_blur(fz, 25 * S)[..., None] * np.array([0.3, 0.6, 1.0], F32) * 0.2
    return finish(img, exposure=1.2, bloom_amt=0.3, bloom_thr=0.7, vig=0.3)
