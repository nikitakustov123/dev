"""Space scenes."""
import math
import numpy as np
import cv2
from common import *
from landscape import ridge, silhouette_mask, lake_reflection


# ----------------------------------------------------------------------------
# 01  Emission nebula
# ----------------------------------------------------------------------------
def nebula(W, H):
    S = H / 2160
    X, Y = coords(W, H)
    px, py = pixel_grid(W, H)
    q1 = fbm(W, H, H * 0.55, 7, seed=11)
    q2 = fbm(W, H, H * 0.55, 7, seed=12)
    A = H * 0.16
    f = fbm(W, H, H * 0.5, 11, gain=0.55, seed=13)
    g = fbm(W, H, H * 0.35, 11, gain=0.55, seed=14)
    fw = sample(f, px + q1 * A, py + q2 * A, border=cv2.BORDER_REFLECT)
    gw = sample(g, px + q2 * A * 1.4, py - q1 * A * 1.4, border=cv2.BORDER_REFLECT)
    big = fbm(W, H, H * 1.2, 3, seed=15)
    # overall shape: a diagonal cloud with a bright cavity
    shape = np.exp(-((X * 0.55 + Y * 0.6) ** 2) / 0.35) * 0.9 + 0.25 * np.exp(-(X ** 2 + Y ** 2) / 2.5)
    shape *= np.exp(0.45 * big)
    cav = np.exp(-(((X - 0.35) ** 2) / 0.10 + ((Y - 0.05) ** 2) / 0.07))  # ionised bubble
    ha = np.exp(1.15 * fw - 0.6) * shape
    o3 = np.exp(1.3 * gw - 1.0) * shape * (0.4 + 1.8 * cav)
    ha = ha * (1 - 0.55 * cav)
    # filaments: ridged noise in the warped domain
    rid = fbm(W, H, H * 0.25, 10, seed=16, ridged=True)
    ridw = sample(rid, px + q1 * A * 0.7, py + q2 * A * 0.7, border=cv2.BORDER_REFLECT)
    fil = np.clip(ridw, 0, None) ** 2 * shape
    # dust lanes
    dust = fbm(W, H, H * 0.4, 10, gain=0.55, seed=17)
    dustw = sample(dust, px - q2 * A, py + q1 * A, border=cv2.BORDER_REFLECT)
    dmask = smoothstep(0.2, 1.6, dustw) * smoothstep(-0.5, 0.8, big + 0.3)
    absorb = np.exp(-2.6 * dmask)

    img = np.zeros((H, W, 3), F32)
    img += ha[..., None] * np.array([1.0, 0.16, 0.22], F32) * 0.55
    img += o3[..., None] * np.array([0.10, 0.55, 0.85], F32) * 0.55
    img += fil[..., None] * np.array([1.0, 0.45, 0.35], F32) * 0.10
    img += (ha * o3)[..., None] ** 0.5 * np.array([0.9, 0.75, 0.6], F32) * 0.25
    # light from the central cluster
    d2 = (X - 0.35) ** 2 + (Y - 0.05) ** 2
    img *= (0.55 + 0.9 / (1 + d2 * 9))[..., None]
    # rim lighting of dust edges facing the cluster
    edge = np.clip(dmask - blur(dmask, 6 * S), 0, None)
    img += edge[..., None] * np.array([1.0, 0.6, 0.4], F32) * 1.5 * (0.6 / (1 + d2 * 6))[..., None]
    img *= absorb[..., None]
    img += np.array([0.004, 0.003, 0.008], F32)

    dens = np.clip(0.25 + 0.75 * np.exp(-d2 * 30), 0, 1)
    st = starfield(W, H, int(60000 * (W * H) / (3840 * 2160)), seed=18, bright=1.0)
    cl = starfield(W, H, int(2500 * (W * H) / (3840 * 2160)), seed=19, density=np.exp(-d2 * 60).astype(F32),
                   bright=2.5, min_t=9000, max_t=30000)
    img += st * (absorb ** 0.7)[..., None] + cl
    return finish(img, exposure=1.35, bloom_amt=0.25, bloom_thr=0.6, vig=0.35, saturation=1.15)


# ----------------------------------------------------------------------------
# 02  Spiral galaxy
# ----------------------------------------------------------------------------
def galaxy(W, H):
    S = H / 2160
    X, Y = coords(W, H)
    rot, inc = math.radians(28), math.radians(58)
    ci = math.cos(inc)

    def to_disk(x, y):
        xr = x * math.cos(rot) + y * math.sin(rot)
        yr = (-x * math.sin(rot) + y * math.cos(rot)) / ci
        return xr, yr

    xr, yr = to_disk(X, Y)
    r = np.sqrt(xr ** 2 + yr ** 2) + 1e-4
    th = np.arctan2(yr, xr)
    n1 = fbm(W, H, H * 0.25, 8, seed=21)
    n2 = fbm(W, H, H * 0.08, 7, seed=22)
    pitch = math.radians(21)
    phase = th - np.log(r + 0.03) / math.tan(pitch) + 0.35 * n1
    arms = (0.5 + 0.5 * np.cos(2 * phase)) ** 3
    spurs = (0.5 + 0.5 * np.cos(4 * phase + 1.3 * n2)) ** 6 * 0.45
    clump = np.exp(0.55 * n2)
    disk = np.exp(-r / 0.30) * smoothstep(0.0, 0.08, r)
    armlight = (arms + spurs) * clump * disk * smoothstep(0.05, 0.25, r) * 1.6
    bulge = np.exp(-(r / 0.09) ** 0.9) * 3.2 + np.exp(-(r / 0.26) ** 1.2) * 0.6
    # dust lane on the inner edge of the arms
    dphase = phase - 0.42
    dust = (0.5 + 0.5 * np.cos(2 * dphase)) ** 8 * smoothstep(0.06, 0.25, r) * np.exp(-r / 0.7)
    dust *= np.clip(0.6 + 0.6 * n2, 0, 1.5)
    absorb = np.exp(-3.2 * dust)

    young = np.array([0.55, 0.72, 1.0], F32)
    old = np.array([1.0, 0.82, 0.55], F32)
    img = (armlight[..., None] * young * 1.6 + (disk * 0.35)[..., None] * old + bulge[..., None] * np.array([1.0, 0.86, 0.62], F32))
    img *= absorb[..., None]

    # resolved stars along arms (particles)
    rg = rng(23)
    N = int(2_500_000 * W * H / (3840 * 2160))
    rr = rg.exponential(0.30, N * 3) + 0.03
    tt = rg.uniform(-np.pi, np.pi, N * 3)
    ph = tt - np.log(rr + 0.03) / math.tan(pitch)
    p_acc = (0.5 + 0.5 * np.cos(2 * ph)) ** 4 * 0.9 + 0.1
    keep = rg.random(N * 3) < p_acc
    rr, tt = rr[keep][:N], tt[keep][:N]
    jitter = rg.normal(0, 0.02, (2, len(rr)))
    dx = rr * np.cos(tt) + jitter[0]
    dy = rr * np.sin(tt) + jitter[1]
    sx = dx * math.cos(rot) - dy * ci * math.sin(rot)
    sy = dx * math.sin(rot) + dy * ci * math.cos(rot)
    pxs = sx * H / 2 + W / 2
    pys = H / 2 - sy * H / 2
    T = np.exp(rg.uniform(np.log(3500), np.log(20000), len(rr)))
    col = kelvin(T)
    col = col / col.max(1, keepdims=True)
    b = (rg.pareto(2.2, len(rr)) + 0.3) * 0.02
    buf = np.zeros((H, W, 3), F32)
    splat(buf, pxs, pys, (col * b[:, None]).astype(F32))
    buf = blur(buf, 0.7 * S) * 1.8 + blur(np.maximum(buf - 0.15, 0), 3 * S) * 4
    img += buf * absorb[..., None] ** 0.6

    # HII regions
    m = rg.random(len(rr)) < 0.0012
    hb = np.zeros((H, W, 3), F32)
    splat(hb, pxs[m], pys[m], np.tile(np.array([[1.0, 0.25, 0.45]], F32), (m.sum(), 1)) * rg.uniform(0.3, 1.5, (m.sum(), 1)).astype(F32))
    img += blur(hb, 2.0 * S) * 2.5 + blur(hb, 6 * S) * 6

    # background
    img += starfield(W, H, int(25000 * W * H / (3840 * 2160)), seed=24, bright=0.8)
    # a few tiny background galaxies
    for k in range(25):
        gx, gy = rg.uniform(0, W), rg.uniform(0, H)
        gs = rg.uniform(3, 12) * S
        ga = rg.uniform(0, np.pi)
        e = np.zeros((int(gs * 8) | 1, int(gs * 8) | 1), F32)
        cv2.ellipse(e, (e.shape[1] // 2, e.shape[0] // 2), (int(gs), int(gs * rg.uniform(0.25, 0.8))), math.degrees(ga), 0, 360, 1.0, -1)
        e = blur(e, gs * 0.4) * rg.uniform(0.05, 0.25)
        y0, x0 = int(gy) - e.shape[0] // 2, int(gx) - e.shape[1] // 2
        if 0 <= y0 and 0 <= x0 and y0 + e.shape[0] < H and x0 + e.shape[1] < W:
            img[y0:y0 + e.shape[0], x0:x0 + e.shape[1]] += e[..., None] * np.array([1.0, 0.85, 0.7], F32)
    img += np.array([0.002, 0.002, 0.004], F32)
    return finish(img, exposure=1.2, bloom_amt=0.3, bloom_thr=0.7, vig=0.35, saturation=1.1)


# ----------------------------------------------------------------------------
# planet renderer (used by 03 and 06)
# ----------------------------------------------------------------------------
def gas_texture(seed, palette_stops, N=2048, storms=True):
    t1 = tile_noise(N, N / 16, 8, seed)
    t2 = tile_noise(N, N / 8, 8, seed + 1)
    yy, xx = np.meshgrid(np.arange(N // 2, dtype=F32), np.arange(N, dtype=F32), indexing='ij')
    lat = (yy / (N // 2) - 0.5) * np.pi
    # turbulent band structure
    wx = xx + 25 * sample(t1, xx, yy * 2)
    wy = yy + 10 * sample(t2, xx * 2, yy * 2) + 6 * np.sin(xx / N * 2 * np.pi * 6 + lat * 8)
    band = np.sin(lat * 9 + 0.25 * sample(t1, wx, wy * 2)) * 0.5
    band += np.sin(lat * 23 + 0.6 * sample(t2, wx, wy * 3)) * 0.25
    band += 0.07 * sample(t1, wx * 3, wy * 12)
    tex = palette(band * 0.5 + 0.5, palette_stops)
    if storms:
        r = rng(seed)
        for k in range(6):
            cx, cy = r.uniform(0, N), r.uniform(0.25, 0.75) * N / 2
            a, b = r.uniform(25, 90), r.uniform(12, 35)
            d = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2
            sw = np.exp(-d * 1.5)
            swirl = 0.5 + 0.5 * np.sin(np.sqrt(d) * 9 - np.arctan2(yy - cy, xx - cx) * 2)
            col = np.array(r.choice([[0.75, 0.32, 0.15], [0.95, 0.9, 0.85], [0.5, 0.25, 0.12]]), F32)
            tex = tex * (1 - sw[..., None] * 0.85) + sw[..., None] * col * (0.7 + 0.3 * swirl[..., None])
    return tex.astype(F32)


def render_planet(W, H, cx, cy, R, L, tex, tilt=0.3, spin=0.0, rings=None, atmo=(0.4, 0.6, 1.0),
                  atmo_strength=1.0, ambient=0.0):
    """Orthographic planet with optional rings. cx,cy,R in pixels.
    Returns colour (H,W,3) and alpha coverage (H,W) of planet+rings."""
    px, py = pixel_grid(W, H)
    x = (px - cx) / R
    y = (cy - py) / R
    d2 = x * x + y * y
    inside = d2 < 1.0
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    L = np.array(L, F32)
    L /= np.linalg.norm(L)
    ndl = x * L[0] + y * L[1] + z * L[2]
    # texture lookup in planet frame (axial tilt about z, then spin)
    ct, st = math.cos(tilt), math.sin(tilt)
    xp = x * ct + y * st
    yp = -x * st + y * ct
    lat = np.arcsin(np.clip(yp, -1, 1))
    lon = np.arctan2(xp, z) + spin
    TH, TW = tex.shape[:2]
    u = (lon / (2 * np.pi) % 1.0) * TW
    v = (lat / np.pi + 0.5) * TH
    surf = np.stack([sample(tex[..., c].copy(), u, v, border=cv2.BORDER_REFLECT) for c in range(3)], -1)
    diff = smoothstep(-0.12, 0.6, ndl) * np.clip(ndl * 0.6 + 0.6, 0, 1)
    limb = z ** 0.35
    col = surf * (diff * limb)[..., None]
    rim = (1 - z) ** 2.5 * smoothstep(-0.25, 0.35, ndl)
    col += rim[..., None] * np.array(atmo, F32) * 0.9 * atmo_strength
    col += surf * ambient
    edge = np.sqrt(d2)
    alpha = np.clip((1 - edge) * R + 0.5, 0, 1)
    # outer atmosphere halo
    halo = np.exp(-np.maximum(edge - 1, 0) * 28) * (edge >= 0.999) * smoothstep(-0.4, 0.5, (x * L[0] + y * L[1]) / (edge + 1e-6))
    color = col * alpha[..., None]
    color += halo[..., None] * np.array(atmo, F32) * 0.35 * atmo_strength * (1 - alpha)[..., None]
    cover = alpha.copy()

    if rings is not None:
        r_in, r_out, nrm, profile, rcol = rings
        nrm = np.array(nrm, F32)
        nrm /= np.linalg.norm(nrm)
        zr = -(nrm[0] * x + nrm[1] * y) / nrm[2]
        rr = np.sqrt(x * x + y * y + zr * zr)
        visible = (~inside) | (zr > z)
        idx = (rr - r_in) / (r_out - r_in)
        onring = (idx >= 0) & (idx <= 1)
        pidx = np.clip(idx, 0, 1) * (len(profile) - 1)
        opac = np.interp(pidx, np.arange(len(profile)), profile).astype(F32) * onring
        ring_rgb = palette(np.clip(idx, 0, 1), rcol)
        # planet shadow on rings
        Qx, Qy, Qz = x, y, zr
        b = Qx * L[0] + Qy * L[1] + Qz * L[2]
        c = rr * rr - 1
        disc = b * b - c
        shadow = (disc > 0) & ((-b - np.sqrt(np.maximum(disc, 0))) > 0)
        sh = np.where(shadow, 0.04, 1.0)
        lit = abs(float(nrm @ L)) * 0.8 + 0.25
        ring_col = ring_rgb * (lit * sh)[..., None]
        a = (opac * visible).astype(F32)
        a = np.clip(a, 0, 1)
        # ring shadow on planet
        nl = float(nrm @ L)
        if abs(nl) > 1e-4:
            t = -(nrm[0] * x + nrm[1] * y + nrm[2] * z) / nl
            Sx, Sy, Sz = x + t * L[0], y + t * L[1], z + t * L[2]
            srr = np.sqrt(Sx * Sx + Sy * Sy + Sz * Sz)
            sidx = (srr - r_in) / (r_out - r_in)
            sop = np.interp(np.clip(sidx, 0, 1) * (len(profile) - 1), np.arange(len(profile)), profile)
            sop = sop * ((sidx >= 0) & (sidx <= 1) & (t > 0) & inside)
            color *= (1 - 0.85 * sop)[..., None]
        color = color * (1 - a[..., None]) + ring_col * a[..., None]
        cover = np.maximum(cover, a)
    return color.astype(F32), cover.astype(F32)


def ring_profile(seed, n=4000):
    r = rng(seed)
    p = 0.55 + 0.25 * noise1d(n, 300, seed, octaves=6) / 2.5
    p += 0.12 * noise1d(n, 40, seed + 1, octaves=4) / 2.5
    p += 0.08 * np.sin(np.arange(n) * 0.9)
    t = np.linspace(0, 1, n)
    p *= smoothstep(0.0, 0.05, t) * (1 - smoothstep(0.94, 1.0, t))
    # gaps
    for g, w, depth in ((0.58, 0.035, 0.97), (0.86, 0.008, 0.9), (0.22, 0.02, 0.6), (0.4, 0.006, 0.8), (0.7, 0.004, 0.7)):
        p *= 1 - depth * np.exp(-((t - g) / w) ** 2)
    p[: int(n * 0.2)] *= 0.35  # faint inner C ring
    return np.clip(p, 0, 1).astype(F32)


# ----------------------------------------------------------------------------
# 03  Ringed giant
# ----------------------------------------------------------------------------
def ringed_planet(W, H):
    S = H / 2160
    tex = gas_texture(31, [(0, hexcol('#5a3a22')), (0.3, hexcol('#b98a5a')), (0.5, hexcol('#e8d3ac')),
                           (0.7, hexcol('#c7a77a')), (1, hexcol('#f4ead6'))])
    img = np.zeros((H, W, 3), F32)
    img += starfield(W, H, int(30000 * W * H / (3840 * 2160)), seed=32, bright=0.7)
    X, Y = coords(W, H)
    neb = np.exp(1.2 * fbm(W, H, H * 0.6, 8, seed=33)) * 0.004
    img += neb[..., None] * np.array([0.5, 0.3, 1.0], F32)
    rings = (1.25, 2.35, (0.0, 0.94, 0.33), ring_profile(34),
             [(0, hexcol('#6d655b')), (0.3, hexcol('#b7a68c')), (0.6, hexcol('#ddd1b8')), (1, hexcol('#8a8273'))])
    col, a = render_planet(W, H, W * 0.47, H * 0.53, H * 0.33, (-0.85, 0.3, 0.05), tex, tilt=0.38,
                           rings=rings, atmo=(0.9, 0.75, 0.5), atmo_strength=0.6, ambient=0.004)
    img = img * (1 - a[..., None]) + col * 1.6
    # small moon
    mtex = palette(fbm(512, 256, 30, 6, seed=35) * 0.2 + 0.5, [(0, (0.25, 0.24, 0.23)), (1, (0.7, 0.68, 0.65))])
    mc, ma = render_planet(W, H, W * 0.86, H * 0.2, H * 0.035, (-0.85, 0.3, 0.05), mtex, atmo=(0, 0, 0), atmo_strength=0)
    img = img * (1 - ma[..., None]) + mc * 1.4
    return finish(img, exposure=1.25, bloom_amt=0.15, bloom_thr=0.8, vig=0.3)


# ----------------------------------------------------------------------------
# 05  Milky Way over a mountain lake
# ----------------------------------------------------------------------------
def milky_way(W, H):
    S = H / 2160
    X, Y = coords(W, H)
    px, py = pixel_grid(W, H)
    hor = int(H * 0.68)
    # galactic band: tilted line
    ang = math.radians(62)
    d = (X - 0.25) * math.sin(ang) - (Y - 0.0) * math.cos(ang)
    along = (X - 0.25) * math.cos(ang) + (Y) * math.sin(ang)
    n = fbm(W, H, H * 0.3, 10, seed=51)
    n2 = fbm(W, H, H * 0.12, 9, seed=52)
    width = 0.28 + 0.06 * n
    band = np.exp(-(d / width) ** 2) * (0.6 + 0.5 * np.exp(-((along + 0.1) / 0.9) ** 2))
    core = np.exp(-((along + 0.45) ** 2) / 0.12 - (d ** 2) / 0.05)  # galactic centre glow
    glow = band * np.exp(0.5 * n2) + core * 1.5
    dust = fbm(W, H, H * 0.18, 10, seed=53, ridged=True)
    dlane = smoothstep(0.2, 1.8, dust + 1.2 * np.exp(-(d / 0.06) ** 2) - 0.2) * band
    absorb = np.exp(-2.4 * dlane)
    sky = np.zeros((H, W, 3), F32)
    t = np.clip((H - py) / H, 0, 1)
    sky += palette(1 - (py / hor), [(0, hexcol('#3a2a3e') * 0.25), (0.15, hexcol('#1a2240') * 0.18),
                                     (0.6, hexcol('#070b1a') * 0.6), (1, hexcol('#02030a'))])
    # airglow near the horizon
    sky += (np.exp(-np.abs(hor - py) / (H * 0.08)) * 0.03)[..., None] * np.array([0.3, 0.6, 0.25], F32)
    mw = glow[..., None] * np.array([0.85, 0.78, 0.7], F32) * 0.11
    mw += core[..., None] * np.array([1.0, 0.75, 0.45], F32) * 0.08
    pink = smoothstep(1.2, 2.5, n2) * band
    mw += pink[..., None] * np.array([1.0, 0.2, 0.35], F32) * 0.03
    sky += mw * absorb[..., None]
    dens = np.clip(0.18 + 1.2 * band * absorb, 0, 1).astype(F32)
    stars = starfield(W, H, int(160000 * W * H / (3840 * 2160)), seed=54, density=dens, bright=0.75)
    sky += stars * absorb[..., None] ** 0.5
    # mountains
    m1 = ridge(W, hor - 0.12 * H, 0.06 * H, 900 * S, 55, octaves=8, sharp=0.6)
    m2 = ridge(W, hor - 0.03 * H, 0.04 * H, 500 * S, 56, octaves=8, sharp=0.5)
    far = silhouette_mask(H, m1, 1.0)
    near = silhouette_mask(H, m2, 1.0)
    farcol = np.array([0.004, 0.006, 0.012], F32)
    # faint starlight on the far range
    img = sky * (1 - far[..., None]) + farcol * far[..., None]
    img = img * (1 - near[..., None]) + np.array([0.0015, 0.002, 0.004], F32) * near[..., None]
    # conifer tree line on the near shore
    from nature import pine
    rg = rng(57)
    tm = np.zeros((H, W), np.uint8)
    x = 0.0
    while x < W:
        h = rg.uniform(0.015, 0.06) * H * (1.0 if rg.random() < 0.85 else 1.6)
        pine(tm, x, hor + 2, h, h * rg.uniform(0.18, 0.26), rg, S)
        x += rg.uniform(4, 28) * S
    tmf = tm.astype(F32) / 255
    img = img * (1 - tmf[..., None])
    img[hor:] = 0
    img = lake_reflection(img, hor, 59, ripple=3.0, darken=0.55)
    return finish(img, exposure=1.6, bloom_amt=0.2, bloom_thr=0.5, vig=0.4, saturation=1.1)


# ----------------------------------------------------------------------------
# 06  Alien world: giant planet rising over crystal mountains
# ----------------------------------------------------------------------------
def alien_world(W, H):
    S = H / 2160
    X, Y = coords(W, H)
    px, py = pixel_grid(W, H)
    hor = H * 0.72
    t = np.clip(py / hor, 0, 1)
    sky = palette(t, [(0, hexcol('#0b0420')), (0.45, hexcol('#3a1257')), (0.8, hexcol('#c0487a')), (1, hexcol('#ffb38a'))])
    sky *= 0.6
    sky += starfield(W, H, int(20000 * W * H / (3840 * 2160)), seed=61, bright=0.6) * (1 - t)[..., None] ** 2
    tex = gas_texture(62, [(0, hexcol('#123a5a')), (0.35, hexcol('#2f7f9a')), (0.55, hexcol('#9fd9d0')),
                           (0.75, hexcol('#3b6ea8')), (1, hexcol('#e0f4ff'))])
    rings = (1.3, 2.0, (0.25, 0.96, 0.18), ring_profile(63), [(0, hexcol('#8899aa')), (1, hexcol('#ddeeff'))])
    pc, pa = render_planet(W, H, W * 0.62, H * 0.42, H * 0.36, (0.6, -0.2, 0.75), tex, tilt=-0.25,
                           rings=rings, atmo=(0.4, 0.8, 1.0), atmo_strength=1.2, ambient=0.01)
    # atmospheric extinction near horizon
    ext = smoothstep(hor * 0.6, hor, py)[..., None]
    pc = pc * (1 - 0.6 * ext) + ext * 0.0
    img = sky * (1 - pa[..., None] * (1 - 0.35 * ext)) + pc * 1.5
    # two small moons
    for (mx, my, mr, s) in ((0.18, 0.2, 0.05, 64), (0.3, 0.33, 0.022, 65)):
        mt = palette(fbm(512, 256, 20, 6, seed=s) * 0.2 + 0.5, [(0, (0.3, 0.2, 0.25)), (1, (0.9, 0.8, 0.75))])
        mc, ma = render_planet(W, H, W * mx, H * my, H * mr, (0.6, -0.2, 0.75), mt, atmo=(1, 0.5, 0.5), atmo_strength=0.3)
        img = img * (1 - ma[..., None]) + mc * 1.3
    # haze band
    img += (np.exp(-np.abs(py - hor) / (0.05 * H)) * 0.25)[..., None] * np.array([1.0, 0.45, 0.4], F32)
    # mountain layers with rim light and glowing crystal spires
    fog = np.array([0.55, 0.22, 0.35], F32)
    layers = [(hor - 0.07 * H, 0.08 * H, 1200, 0.55), (hor + 0.02 * H, 0.08 * H, 900, 0.35),
              (hor + 0.12 * H, 0.09 * H, 700, 0.18), (hor + 0.24 * H, 0.10 * H, 600, 0.0)]
    rg = rng(67)
    for i, (base, amp, cell, fogk) in enumerate(layers):
        prof = ridge(W, base, amp, cell * S, 66 + i, octaves=10, sharp=0.75)
        m = silhouette_mask(H, prof, 1.0)
        below = np.clip((py - prof[None, :]) / (0.03 * H), 0, 1)
        slope = np.gradient(prof) / S
        rim = np.clip(-slope * 0.25, 0, 1)[None, :] * np.exp(-np.clip(py - prof[None, :], 0, None) / (6 * S)) * (py >= prof[None, :])
        tex = fbm(W, H, 80 * S, 7, seed=80 + i, ridged=True)
        c = np.array([0.012, 0.008, 0.02], F32) * (0.7 + 0.3 * tex[..., None])
        c = c + rim[..., None] * np.array([0.6, 0.9, 1.0], F32) * 0.6
        c = c * (1 - fogk) + fog * fogk * (1 - 0.5 * below[..., None])
        img = img * (1 - m[..., None]) + c * m[..., None]
        if i >= 1:
            sp = np.zeros((H, W), F32)
            for _ in range(int(5 * (i + 1))):
                x0 = rg.uniform(0, W)
                y0 = prof[int(np.clip(x0, 0, W - 1))] + rg.uniform(0.0, 0.05) * H
                h = rg.uniform(0.03, 0.11) * H * (0.6 + 0.3 * i)
                w = h * rg.uniform(0.06, 0.14)
                lean = rg.uniform(-0.25, 0.25) * h
                pts = np.array([[x0 - w, y0], [x0 + lean, y0 - h], [x0 + w, y0], [x0 + lean * 0.2, y0 + w]], np.int32)
                cv2.fillPoly(sp, [pts], 1.0, lineType=cv2.LINE_AA)
            # gradient: brighter at the tips
            glow = sp * np.clip(1 - (py - (prof[None, :] - 0.1 * H)) / (0.25 * H), 0.2, 1)
            ccol = np.array([0.15, 0.85, 1.0], F32) * (1 - fogk)
            img = img * (1 - sp[..., None]) + (glow[..., None] * ccol * 1.4 + sp[..., None] * np.array([0.01, 0.03, 0.05], F32))
            img += fast_blur(glow, 20 * S)[..., None] * ccol * 0.6
    return finish(img, exposure=1.2, bloom_amt=0.25, bloom_thr=0.6, vig=0.35, saturation=1.1)


# ----------------------------------------------------------------------------
# 07  Total solar eclipse
# ----------------------------------------------------------------------------
def eclipse(W, H):
    S = H / 2160
    X, Y = coords(W, H)
    px, py = pixel_grid(W, H)
    cx, cy, R = 0.0, 0.12, 0.2
    dx, dy = X - cx, Y - cy
    r = np.sqrt(dx * dx + dy * dy) / R
    th = np.arctan2(dy, dx)
    N = 2048
    t1 = tile_noise(N, N / 64, 8, 71)
    t2 = tile_noise(N, N / 16, 6, 72)
    u = (th / (2 * np.pi) + 0.5) * N
    lr = np.log(np.maximum(r, 1e-3))
    streak = sample(t1, u, lr * 40) * 0.6 + sample(t2, u * 2, lr * 18) * 0.4
    streak2 = sample(t1, u * 4 + 300, lr * 25)
    eq = np.abs(np.sin(th - 0.35))
    helmet = np.exp(-((np.cos(2 * (th - 0.35)) + 1) * 0.8)) * 1.5 + 0.3
    corona = np.power(np.maximum(r, 1.0), -3.0) * (0.35 + 0.65 * np.exp(0.9 * streak)) * helmet
    corona += np.power(np.maximum(r, 1.0), -2.0) * 0.08 * np.exp(0.7 * streak2) * (eq ** 2)
    corona += np.power(np.maximum(r, 1.0), -6) * 2.5
    corona *= (r > 1.0)
    img = np.zeros((H, W, 3), F32)
    # twilight sky
    t = np.clip(py / H, 0, 1)
    sky = palette(t, [(0, hexcol('#020514')), (0.5, hexcol('#0a1838')), (0.82, hexcol('#3a3a5a')), (0.9, hexcol('#f08a3a')), (1, hexcol('#ffd27a'))])
    img += sky * 0.35
    img += starfield(W, H, int(4000 * W * H / (3840 * 2160)), seed=73, bright=0.6, spikes=False)
    img += corona[..., None] * np.array([0.85, 0.9, 1.0], F32) * 1.2
    # prominences
    prom = sample(t2, u * 3, lr * 120 + 50)
    pm = smoothstep(1.4, 2.4, prom) * np.exp(-np.maximum(r - 1, 0) / 0.035) * (r > 1)
    chrom = np.exp(-np.abs(r - 1.0) / 0.006) * smoothstep(-0.2, 0.6, np.cos(th - 2.4)) * (r > 1)
    img += (pm * 6 + chrom * 2.0)[..., None] * np.array([1.0, 0.12, 0.25], F32)
    # moon disk
    m = np.clip((r - 1.0) * R * H / 2 + 0.5, 0, 1)
    img *= m[..., None]
    img += (1 - m)[..., None] * np.array([0.0008, 0.0008, 0.0012], F32)
    # diamond ring
    ang = math.radians(48)
    bx, by = cx + R * math.cos(ang), cy + R * math.sin(ang)
    bpx, bpy = (bx * H / 2 + W / 2), (H / 2 - by * H / 2)
    d = np.sqrt((px - bpx) ** 2 + (py - bpy) ** 2) / S
    bead = np.exp(-d / 4.0) * 120 + np.exp(-d / 30) * 8 + np.exp(-d / 160) * 1.0
    L = int(1200 * S) | 1
    spk = np.zeros((H, W), F32)
    for a in (0, 60, 120):
        k = np.zeros((L, L), F32)
        cv2.line(k, (0, L // 2), (L - 1, L // 2), 1.0, 1)
        Mk = cv2.getRotationMatrix2D((L / 2, L / 2), a + 10, 1)
        k = cv2.warpAffine(k, Mk, (L, L))
        k *= np.exp(-np.hypot(*np.meshgrid(np.arange(L) - L / 2, np.arange(L) - L / 2)) / (L / 12))
        src = np.zeros((H, W), F32)
        src[int(bpy), int(bpx)] = 1.0
        spk += cv2.filter2D(src, -1, k) * 3
    img += (bead * (m * 0.9 + 0.1) + spk * m)[..., None] * np.array([1.0, 0.95, 0.88], F32)
    # horizon silhouette
    prof = ridge(W, H * 0.9, H * 0.04, 700 * S, 74, octaves=7, sharp=0.4)
    sm = silhouette_mask(H, prof, 1.0)
    img *= (1 - sm[..., None])
    img += sm[..., None] * np.array([0.002, 0.002, 0.003], F32)
    return finish(img, exposure=1.1, bloom_amt=0.35, bloom_thr=0.7, vig=0.3)
