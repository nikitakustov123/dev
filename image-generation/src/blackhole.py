"""04  Schwarzschild black hole with a thin accretion disk.

Real null-geodesic ray tracing: the photon orbit equation
    d^2u/dphi^2 = 3 M u^2 - u      (u = 1/r, M = 1)
is integrated once for every impact parameter b, then every pixel looks up
where its photon crosses the disk plane (primary image, the lensed image of
the far side of the disk and the photon ring). Escaped rays sample a lensed
star map; Doppler beaming and gravitational redshift colour the disk.
"""
import math
import numpy as np
from scipy.ndimage import map_coordinates
from common import *


def geodesic_table(bmax=28.0, nb=5000, dphi=0.004, phimax=3.25 * np.pi):
    b = np.linspace(0.004, bmax, nb)
    npf = int(phimax / dphi) + 1
    U = np.empty((nb, npf), np.float32)
    Wd = np.empty((nb, npf), np.float32)
    u = np.zeros(nb)
    w = 1.0 / b
    status = np.zeros(nb, np.int8)  # 0 active, 1 escaped, 2 captured
    phi_esc = np.full(nb, -1.0)

    def f(u, w):
        return w, 3 * u * u - u

    for j in range(npf):
        U[:, j] = np.where(status == 0, u, np.where(status == 1, -1.0, 0.7))
        Wd[:, j] = np.where(status == 0, w, 0.0)
        k1u, k1w = f(u, w)
        k2u, k2w = f(u + 0.5 * dphi * k1u, w + 0.5 * dphi * k1w)
        k3u, k3w = f(u + 0.5 * dphi * k2u, w + 0.5 * dphi * k2w)
        k4u, k4w = f(u + dphi * k3u, w + dphi * k3w)
        un = u + dphi / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        wn = w + dphi / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        act = status == 0
        esc = act & (un < 0) & (j > 0)
        phi_esc[esc] = (j + u[esc] / (u[esc] - un[esc])) * dphi
        status[esc] = 1
        cap = act & (un >= 0.5)
        status[cap] = 2
        u = np.where(act, un, u)
        w = np.where(act, wn, w)
    # rays that were still orbiting at the end of the table are treated as captured
    return b, U, Wd, phi_esc, dphi


def sky_map(seed, Wm=4096, Hm=2048):
    r = rng(seed)
    n = 220000
    z = r.uniform(-1, 1, n)
    lon = r.uniform(-np.pi, np.pi, n)
    lat = np.arcsin(z)
    x = (lon / (2 * np.pi) + 0.5) * Wm
    y = (0.5 - lat / np.pi) * Hm
    mag = np.minimum(r.pareto(1.4, n) + 1, 500)
    T = np.exp(r.uniform(np.log(3000), np.log(15000), n))
    col = kelvin(T)
    col = col / col.max(1, keepdims=True)
    buf = np.zeros((Hm, Wm, 3), F32)
    splat(buf, x, y, (col * (mag * 0.05)[:, None]).astype(F32))
    buf = blur(buf, 0.6) * 0.8 + blur(np.maximum(buf - 0.6, 0), 2.5) * 2
    # faint galactic band
    yy, xx = np.meshgrid(np.linspace(-1, 1, Hm), np.linspace(-1, 1, Wm), indexing='ij')
    band = np.exp(-((yy - 0.25 * np.sin(xx * np.pi)) / 0.12) ** 2)
    neb = np.exp(0.8 * fbm(Wm, Hm, Hm * 0.2, 8, seed=seed + 1))
    buf += (band * neb * 0.02)[..., None] * np.array([0.8, 0.7, 1.0], F32)
    return buf


def black_hole(W, H):
    S = H / 2160
    b_tab, U, Wd, phi_esc, dphi = geodesic_table()
    nb, npf = U.shape
    db = b_tab[1] - b_tab[0]

    X, Y = coords(W, H)
    scale = 12.5
    roll = math.radians(-9)
    bx = (X * math.cos(roll) - Y * math.sin(roll)) * scale
    by = (X * math.sin(roll) + Y * math.cos(roll)) * scale
    elev = math.radians(7.0)
    c = np.array([0, math.sin(elev), math.cos(elev)])
    Xs = np.array([1.0, 0, 0])
    Ys = np.array([0, math.cos(elev), -math.sin(elev)])
    n = np.array([0, 1.0, 0])

    b = np.sqrt(bx * bx + by * by).astype(np.float64) + 1e-6
    ex = (bx * Xs[0] + by * Ys[0]) / b
    ey = (bx * Xs[1] + by * Ys[1]) / b
    ez = (bx * Xs[2] + by * Ys[2]) / b
    nc = n @ c
    ne = ey  # n = (0,1,0)
    phi0 = np.mod(np.arctan2(-nc, ne), np.pi)
    bi = (b - b_tab[0]) / db

    r_in, r_out = 4.2, 17.0
    Tmax = 6400.0
    N = 2048
    tex1 = tile_noise(N, N / 48, 8, 41)
    tex2 = tile_noise(N, N / 12, 6, 42)

    img = np.zeros((H, W, 3), F32)
    trans = np.ones((H, W), F32)
    for k in range(3):
        phi = phi0 + k * np.pi
        pj = phi / dphi
        coords_ = np.array([bi.ravel(), pj.ravel()])
        u = map_coordinates(U, coords_, order=1, mode='nearest').reshape(H, W)
        du = map_coordinates(Wd, coords_, order=1, mode='nearest').reshape(H, W)
        valid = (u > 1e-4) & (u < 0.5) & (pj < npf - 1)
        r = np.where(valid, 1.0 / np.maximum(u, 1e-4), 0.0)
        hit = valid & (r > r_in * 0.92) & (r < r_out * 1.1)
        if not hit.any():
            continue
        cp, sp = np.cos(phi), np.sin(phi)
        Px = r * (cp * c[0] + sp * ex)
        Pz = r * (cp * c[2] + sp * ez)
        # photon travel direction (towards camera) = -dP/dphi
        rp = -du / np.maximum(u * u, 1e-8)
        dx = rp * (cp * c[0] + sp * ex) + r * (-sp * c[0] + cp * ex)
        dy = rp * (cp * c[1] + sp * ey) + r * (-sp * c[1] + cp * ey)
        dz = rp * (cp * c[2] + sp * ez) + r * (-sp * c[2] + cp * ez)
        dn = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
        kx, kz = -dx / dn, -dz / dn
        vx, vz = Pz / np.maximum(r, 1e-6), -Px / np.maximum(r, 1e-6)
        beta = np.clip(np.sqrt(1.0 / np.maximum(r - 2.0, 0.5)), 0, 0.72)
        gamma = 1 / np.sqrt(1 - beta ** 2)
        delta = 1.0 / (gamma * (1 - beta * (vx * kx + vz * kz)))
        g = delta * np.sqrt(np.clip(1 - 2.0 / np.maximum(r, 2.01), 0.01, 1))
        x = r_in / np.maximum(r, 1e-3)
        prof = np.clip(x ** 2.6 * (1 - np.sqrt(np.clip(x, 0, 1))) ** 0.6, 0, None) * 9
        psi = np.arctan2(Pz, Px)
        lr = np.log(np.maximum(r, 1e-3))
        su = (psi / (2 * np.pi) + 0.5) * N + lr * 260
        sv = lr * N * 0.42
        turb = sample(tex1, su, sv) * 0.7 + sample(tex2, su * 0.5 + 400, sv * 0.5) * 0.5
        dens = np.exp(0.55 * turb)
        Tobs = Tmax * (x ** 0.75) ** 0.6 * g * 1.15
        col = kelvin(np.clip(Tobs, 1000, 40000))
        col = col / (col.max(-1, keepdims=True) + 1e-6)
        I = prof * dens * g ** 3.2
        edge = smoothstep(r_in * 0.92, r_in * 1.08, r) * (1 - smoothstep(r_out * 0.7, r_out * 1.1, r))
        alpha = np.clip(edge * (0.55 + 0.4 * smoothstep(-1, 1, turb)), 0, 0.97) * hit
        emis = col * (I * edge)[..., None].astype(F32)
        img += (trans * alpha)[..., None] * emis
        trans *= (1 - alpha)

    # background: lensed stars for escaped rays
    pe = np.interp(b, b_tab, phi_esc)
    escaped = (pe > 0) & (b > 5.15)
    cp, sp = np.cos(pe), np.sin(pe)
    ox = cp * c[0] + sp * ex
    oy = cp * c[1] + sp * ey
    oz = cp * c[2] + sp * ez
    sky = sky_map(43)
    Hm, Wm = sky.shape[:2]
    lon = np.arctan2(ox, oz)
    lat = np.arcsin(np.clip(oy, -1, 1))
    su = (lon / (2 * np.pi) + 0.5) * Wm
    sv = (0.5 - lat / np.pi) * Hm
    bg = np.stack([sample(sky[..., i].copy(), su, sv) for i in range(3)], -1)
    # soften the hard shadow edge (anti-alias)
    shadow_aa = smoothstep(5.19, 5.24, b)
    img += bg * (escaped * trans * shadow_aa)[..., None]
    return finish(img, exposure=1.35, bloom_amt=0.6, bloom_thr=0.5, vig=0.35, saturation=1.25)
