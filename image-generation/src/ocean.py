"""10  Ocean at sunset -- physically inspired water: a spectrum of directional
waves (band-limited per pixel to avoid aliasing), Fresnel reflection of a
procedural sky with perspective clouds, and a roughness-aware sun glitter path.
"""
import math
import numpy as np
from common import *


SUN = np.array([0.0, 0.055, 1.0])
SUN = SUN / np.linalg.norm(SUN)


def sky_color(dx, dy, dz, cloud_tex, S):
    """Sky radiance for direction arrays."""
    el = np.clip(dy, -0.05, 1)
    t = np.clip(el / 0.45, 0, 1) ** 0.55
    base = palette(t, [(0, hexcol('#ff8a4a') * 0.9), (0.1, hexcol('#f0605a') * 0.7), (0.3, hexcol('#9a3a6a') * 0.6),
                       (0.6, hexcol('#3a2a6a') * 0.6), (1, hexcol('#0c1030'))])
    cosg = np.clip(dx * SUN[0] + dy * SUN[1] + dz * SUN[2], -1, 1)
    ang = np.arccos(cosg)
    glow = np.exp(-ang / 0.04) * 5 + np.exp(-ang / 0.25) * 0.8 + np.exp(-ang / 1.0) * 0.12
    sun_disk = smoothstep(0.0105, 0.0085, ang) * 120
    col = base + (glow + sun_disk)[..., None] * np.array([1.0, 0.62, 0.32], F32)
    # clouds on a plane at altitude 1
    up = np.maximum(dy, 0.004)
    cx = dx / up
    cz = dz / up
    N = cloud_tex.shape[0]
    u = cx * 120 + 3000.0
    v = cz * 120
    c1 = sample(cloud_tex, u, v)
    c2 = sample(cloud_tex, u * 3.1 + 517, v * 3.1)
    dens = smoothstep(-0.2, 1.4, c1 * 0.8 + c2 * 0.35) * smoothstep(0.01, 0.12, dy)
    near = np.exp(-ang / 0.45)[..., None]
    lit = np.exp(-ang / 0.6) * 1.8 + 0.3
    warm = palette(np.clip(dens, 0, 1), [(0, hexcol('#ffd090')), (0.5, hexcol('#ff8a6a')), (1, hexcol('#b05a7a'))])
    cool = palette(np.clip(dens, 0, 1), [(0, hexcol('#ff9aa0')), (0.5, hexcol('#8a5aa0')), (1, hexcol('#3a3a7a'))])
    ccol = warm * near + cool * (1 - near)
    edge = np.clip(1 - dens, 0, 1) * dens * 4
    cloud_rgb = ccol * (lit * (0.4 + edge))[..., None]
    a = np.clip(dens * 0.95, 0, 1)[..., None]
    return col * (1 - a) + cloud_rgb * a


def ocean_sunset(W, H):
    S = H / 2160
    X, Y = coords(W, H)
    yh = -0.08
    f = 1.6
    dx = X.astype(np.float64)
    dy = (Y - yh).astype(np.float64)
    dz = np.full_like(dx, f)
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / n, dy / n, dz / n
    cloud_tex = tile_noise(2048, 256, 9, 101, gain=0.55)

    img = sky_color(dx, dy, dz, cloud_tex, S).astype(F32)

    water = dy < -1e-4
    cam_h = 3.0
    t = cam_h / np.maximum(-dy, 1e-5)
    x = t * dx
    z = t * dz
    pix = 2.0 / H / f
    fp = t * pix / np.maximum(-dy, 0.02) ** 0.5  # footprint (m), grazing-stretched
    rg = rng(102)
    nw = 56
    gx = np.zeros_like(x)
    gz = np.zeros_like(x)
    lost = np.zeros_like(x)
    wind = math.radians(12)
    lams = np.exp(np.linspace(np.log(70), np.log(0.06), nw))
    for i in range(nw):
        lam = lams[i]
        k = 2 * np.pi / lam
        th = wind + rg.normal(0, 0.55)
        kx, kz = k * math.cos(th), k * math.sin(th)
        A = lam * 0.016 / math.sqrt(nw / 10)
        filt = np.exp(-(k * fp * 1.0) ** 2)
        ph = rg.uniform(0, 2 * np.pi)
        c = np.cos(kx * x + kz * z + ph)
        gx += A * kx * c * filt
        gz += A * kz * c * filt
        lost += (A * k) ** 2 * 0.5 * (1 - filt ** 2)
    nx, ny, nz = -gx, np.ones_like(gx), -gz
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    dn = dx * nx + dy * ny + dz * nz
    rx, ry, rz = dx - 2 * dn * nx, dy - 2 * dn * ny, dz - 2 * dn * nz
    ry = np.abs(ry)  # rays that would go into the water reflect off the next wave
    refl = sky_color(rx, ry, rz, cloud_tex, S)
    cosi = np.clip(-dn, 0, 1)
    F = 0.02 + 0.98 * (1 - cosi) ** 5
    # glitter: roughness-aware sun specular
    sig2 = 0.0015 + lost
    rs = rx * SUN[0] + ry * SUN[1] + rz * SUN[2]
    spec = np.exp(-(1 - np.clip(rs, -1, 1)) / (sig2 * 0.5)) * (0.02 / sig2) * 30
    # deep water colour with subsurface glow toward the sun
    sss = np.clip(gx * 0 + (dx * SUN[0] + dz * SUN[2]), 0, 1) ** 4 * np.clip(-gz * 3 + 0.2, 0, 1) * 0.06
    body = np.array([0.004, 0.012, 0.025], F32)
    wcol = refl * F[..., None] + (body + sss[..., None] * np.array([0.1, 0.5, 0.45], F32)) * (1 - F)[..., None]
    wcol += spec[..., None] * np.array([1.0, 0.6, 0.3], F32) * F[..., None]
    # haze toward the horizon
    hz = 1 - np.exp(-t / 2500)
    horizon_col = sky_color(dx, np.full_like(dy, 0.002), dz, cloud_tex, S)
    wcol = wcol * (1 - hz[..., None]) + horizon_col * hz[..., None] * 0.9
    aa = smoothstep(-1e-4, -2.5 * pix, dy)
    img = img * (1 - aa[..., None]) + wcol.astype(F32) * aa[..., None]
    return finish(img, exposure=0.85, bloom_amt=0.3, bloom_thr=1.0, vig=0.35, saturation=1.1)
