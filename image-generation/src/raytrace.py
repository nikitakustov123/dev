"""21  Studio still life: a vectorised Whitted-style ray tracer.

Chrome, gold, glass and lacquered spheres on a polished floor, lit by
softbox area lights. Features: Fresnel reflections, refraction with total
internal reflection, analytic soft shadows, contact ambient occlusion and
4x jittered supersampling.
"""
import math
import numpy as np
from common import *

CHROME, GOLD, GLASS, LACQUER, EMIT = 0, 1, 2, 3, 4

SPHERES = [
    # centre, radius, material, colour
    ((0.0, 1.25, 0.0), 1.25, CHROME, (0.96, 0.96, 0.98)),
    ((-2.75, 0.85, -0.6), 0.85, GLASS, (0.92, 0.97, 1.0)),
    ((2.7, 0.95, -0.2), 0.95, GOLD, (1.0, 0.76, 0.33)),
    ((-1.35, 0.45, -2.1), 0.45, LACQUER, (0.7, 0.04, 0.05)),
    ((1.25, 0.4, -2.4), 0.4, LACQUER, (0.03, 0.12, 0.7)),
    ((0.25, 0.22, -2.9), 0.22, EMIT, (6.0, 2.2, 0.6)),
    ((-0.45, 0.18, -3.4), 0.18, EMIT, (0.4, 3.0, 6.0)),
    ((-4.8, 0.7, 2.8), 0.7, CHROME, (0.96, 0.96, 0.98)),
    ((4.9, 0.8, 3.2), 0.8, LACQUER, (0.9, 0.85, 0.8)),
    ((1.6, 0.55, 3.6), 0.55, GOLD, (1.0, 0.76, 0.33)),
    ((-2.2, 0.35, 2.4), 0.35, LACQUER, (0.05, 0.4, 0.15)),
    ((3.9, 0.3, -1.9), 0.3, CHROME, (0.96, 0.96, 0.98)),
    ((-3.9, 0.28, -2.2), 0.28, GOLD, (1.0, 0.76, 0.33)),
]

C = np.array([s[0] for s in SPHERES], np.float64)
R = np.array([s[1] for s in SPHERES], np.float64)
MAT = np.array([s[2] for s in SPHERES])
COL = np.array([s[3] for s in SPHERES], np.float64)


def _n(v):
    return v / np.linalg.norm(v)


LIGHTS = [
    # direction, up-hint, half-size (u,v), radiance colour
    (_n(np.array([-0.55, 0.75, -0.55])), np.array([0, 1.0, 0]), (0.42, 0.26), np.array([1.0, 0.97, 0.92]) * 5.0),
    (_n(np.array([0.05, 1.0, 0.25])), np.array([0, 0, 1.0]), (0.9, 0.07), np.array([1.0, 1.0, 1.0]) * 3.0),
    (_n(np.array([0.85, 0.3, 0.75])), np.array([0, 1.0, 0]), (0.1, 0.45), np.array([0.25, 0.75, 1.0]) * 6.0),
    (_n(np.array([-0.85, 0.3, 0.75])), np.array([0, 1.0, 0]), (0.1, 0.45), np.array([1.0, 0.25, 0.65]) * 6.0),
]
LIGHT_FRAMES = []
for (c, up, sz, rad) in LIGHTS:
    u = _n(np.cross(up, c)) if abs(np.dot(up, c)) < 0.99 else _n(np.cross([1, 0, 0], c))
    v = np.cross(c, u)
    LIGHT_FRAMES.append((c, u, v, sz, rad))


def env(d):
    y = d[:, 1]
    base = np.clip(0.012 + 0.02 * (1 - np.abs(y)) ** 6, 0, 1)[:, None] * np.array([0.6, 0.7, 1.0])
    out = base.copy()
    for (c, u, v, (su, sv), rad) in LIGHT_FRAMES:
        dc = d @ c
        ok = dc > 0.05
        a = (d @ u) / np.maximum(dc, 1e-3)
        b = (d @ v) / np.maximum(dc, 1e-3)
        ea = smoothstep(su, su * 0.92, np.abs(a))
        eb = smoothstep(sv, sv * 0.85, np.abs(b))
        # soft-box falloff toward the edges
        fall = 0.75 + 0.25 * (1 - (a / su) ** 2)
        out += (ok * ea * eb * fall)[:, None] * rad
    return out


def intersect(o, d, skip=None):
    n = o.shape[0]
    tbest = np.full(n, np.inf)
    ibest = np.full(n, -2, np.int64)  # -1 floor, -2 nothing
    for i in range(len(R)):
        oc = o - C[i]
        b = np.einsum('ij,ij->i', oc, d)
        c = np.einsum('ij,ij->i', oc, oc) - R[i] ** 2
        disc = b * b - c
        hit = disc > 0
        sq = np.sqrt(np.maximum(disc, 0))
        t0 = -b - sq
        t1 = -b + sq
        t = np.where(t0 > 1e-4, t0, np.where(t1 > 1e-4, t1, np.inf))
        t = np.where(hit, t, np.inf)
        better = t < tbest
        tbest = np.where(better, t, tbest)
        ibest = np.where(better, i, ibest)
    tf = np.where(d[:, 1] < -1e-6, -o[:, 1] / np.minimum(d[:, 1], -1e-6), np.inf)
    tf = np.where(tf > 1e-4, tf, np.inf)
    better = tf < tbest
    tbest = np.where(better, tf, tbest)
    ibest = np.where(better, -1, ibest)
    return tbest, ibest


def soft_shadow(p, L, ang):
    s = np.ones(p.shape[0])
    for i in range(len(R)):
        if MAT[i] == EMIT:
            continue
        oc = C[i] - p
        tca = oc @ L
        d2 = np.einsum('ij,ij->i', oc, oc) - tca ** 2
        dist = np.sqrt(np.maximum(d2, 0)) - R[i]
        w = np.maximum(tca * ang, 1e-3)
        occ = 1 - smoothstep(-w, w, dist)
        occ = np.where(tca > 0, occ, 0)
        if MAT[i] == GLASS:
            occ *= 0.55
        s *= 1 - occ
    return s


def contact_ao(p, n):
    ao = np.ones(p.shape[0])
    for i in range(len(R)):
        v = C[i] - p
        dd = np.sqrt(np.einsum('ij,ij->i', v, v))
        cosv = np.clip(np.einsum('ij,ij->i', v, n) / dd, 0, 1)
        ao *= 1 - np.clip((R[i] / dd) ** 2 * cosv, 0, 1) * 0.9
    return ao


def direct(p, n, alb):
    out = np.zeros_like(p)
    for (c, u, v, (su, sv), rad) in LIGHT_FRAMES:
        ndl = np.clip(n @ c, 0, 1)
        E = rad * (4 * su * sv) * 0.35
        sh = soft_shadow(p + n * 1e-3, c, (su + sv) * 0.5)
        out += (ndl * sh)[:, None] * E
    return alb * out


def schlick(cosi, f0):
    return f0 + (1 - f0) * (1 - cosi)[:, None] ** 5


def trace(o, d, depth=7):
    n = o.shape[0]
    col = np.zeros((n, 3))
    thr = np.ones((n, 3))
    idx = np.arange(n)
    for bounce in range(depth):
        if len(idx) == 0:
            break
        t, hid = intersect(o, d)
        miss = hid == -2
        col[idx[miss]] += thr[miss] * env(d[miss])
        keep = ~miss
        o, d, thr, idx, t, hid = o[keep], d[keep], thr[keep], idx[keep], t[keep], hid[keep]
        p = o + d * t[:, None]
        nrm = np.zeros_like(p)
        fl = hid == -1
        nrm[fl] = (0, 1, 0)
        sp = ~fl
        nrm[sp] = (p[sp] - C[hid[sp]]) / R[hid[sp]][:, None]
        mat = np.where(fl, -1, MAT[np.maximum(hid, 0)])
        new_o = p.copy()
        new_d = d.copy()
        cont = np.ones(len(idx), bool)
        cosi = -np.einsum('ij,ij->i', d, nrm)
        refl_d = d + 2 * cosi[:, None] * nrm

        # floor: polished dark tiles
        m = mat == -1
        if m.any():
            pp = p[m]
            tile = (np.floor(pp[:, 0] / 1.5) + np.floor(pp[:, 2] / 1.5)) % 2
            grout = np.minimum(np.abs(np.mod(pp[:, 0], 1.5) - 0.75), np.abs(np.mod(pp[:, 2], 1.5) - 0.75))
            alb = (0.025 + 0.012 * tile)[:, None] * np.array([1.0, 1.0, 1.05])
            alb *= (1 - 0.6 * (grout > 0.735))[:, None]
            ao = contact_ao(pp, nrm[m])
            fade = np.exp(-np.sqrt(pp[:, 0] ** 2 + pp[:, 2] ** 2) / 14)[:, None]
            col[idx[m]] += thr[m] * (direct(pp, nrm[m], alb) * ao[:, None] * fade + alb * 0.02)
            F = schlick(np.clip(cosi[m], 0, 1), 0.04) * 0.85 * fade * ao[:, None] ** 0.5
            thr[m] *= F
            new_d[m] = refl_d[m]
            new_o[m] = pp + nrm[m] * 1e-4
        for mm, f0col in ((CHROME, None), (GOLD, None)):
            m = mat == mm
            if m.any():
                f0 = COL[hid[m]]
                F = schlick(np.clip(cosi[m], 0, 1), f0)
                thr[m] *= F
                new_d[m] = refl_d[m]
                new_o[m] = p[m] + nrm[m] * 1e-4
        m = mat == LACQUER
        if m.any():
            alb = COL[hid[m]]
            ao = contact_ao(p[m] - nrm[m] * 0.0, nrm[m]) ** 0.3
            col[idx[m]] += thr[m] * (direct(p[m], nrm[m], alb) + alb * 0.03) * ao[:, None]
            F = schlick(np.clip(cosi[m], 0, 1), 0.05)
            thr[m] *= F
            new_d[m] = refl_d[m]
            new_o[m] = p[m] + nrm[m] * 1e-4
        m = mat == EMIT
        if m.any():
            e = COL[hid[m]]
            rim = (0.6 + 0.4 * np.clip(cosi[m], 0, 1))[:, None]
            col[idx[m]] += thr[m] * e * rim
            cont[m] = False
        m = mat == GLASS
        if m.any():
            dd = d[m]
            nn = nrm[m]
            ci = cosi[m]
            entering = ci > 0
            eta = np.where(entering, 1 / 1.5, 1.5)
            nn = np.where(entering[:, None], nn, -nn)
            ci = np.abs(ci)
            k = 1 - eta ** 2 * (1 - ci ** 2)
            tir = k < 0
            tdir = eta[:, None] * dd + (eta * ci - np.sqrt(np.maximum(k, 0)))[:, None] * nn
            F = schlick(ci, 0.04)[:, 0]
            F = np.where(tir, 1.0, F)
            # reflection approximated by an environment lookup
            rd = dd + 2 * ci[:, None] * nn
            col[idx[m]] += thr[m] * env(rd / np.linalg.norm(rd, axis=1, keepdims=True)) * (F * entering)[:, None]
            tint = np.where(entering[:, None], 1.0, COL[hid[m]])
            thr[m] *= np.where(tir[:, None], 1.0, (1 - F)[:, None]) * tint
            nd = np.where(tir[:, None], rd, tdir)
            new_d[m] = nd / np.linalg.norm(nd, axis=1, keepdims=True)
            new_o[m] = p[m] + new_d[m] * 2e-4
        alive = cont & (thr.max(1) > 0.004)
        o, d, thr, idx = new_o[alive], new_d[alive], thr[alive], idx[alive]
    return col


def spheres(W, H):
    S = H / 2160
    cam = np.array([0.0, 1.9, -8.2])
    look = np.array([0.0, 0.75, 0.0])
    fwd = _n(look - cam)
    right = _n(np.cross([0, 1, 0], fwd))
    up = np.cross(fwd, right)
    fov = math.tan(math.radians(19))
    rg = rng(351)
    img = np.zeros((H * W, 3))
    spp = 4 if S > 0.4 else 1
    jit = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)][:spp] if spp > 1 else [(0.5, 0.5)]
    pyy, pxx = np.divmod(np.arange(H * W), W)
    chunk = 700_000
    for (jx, jy) in jit:
        for s0 in range(0, H * W, chunk):
            s1 = min(s0 + chunk, H * W)
            x = (pxx[s0:s1] + jx + rg.uniform(-0.2, 0.2, s1 - s0) - W / 2) / (H / 2) * fov
            y = (H / 2 - pyy[s0:s1] - jy + rg.uniform(-0.2, 0.2, s1 - s0)) / (H / 2) * fov
            d = fwd[None, :] + x[:, None] * right[None, :] + y[:, None] * up[None, :]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            o = np.broadcast_to(cam, d.shape).copy()
            img[s0:s1] += trace(o, d)
    img = (img / len(jit)).reshape(H, W, 3).astype(F32)
    return finish(img, exposure=1.25, bloom_amt=0.35, bloom_thr=0.9, vig=0.3)
