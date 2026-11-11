"""Technology scenes."""
import math
import numpy as np
import cv2
from common import *
from landscape import ridge, silhouette_mask


# ----------------------------------------------------------------------------
# 18  Cyberpunk city at night
# ----------------------------------------------------------------------------
NEON = [(1.0, 0.1, 0.6), (0.1, 0.9, 1.0), (1.0, 0.75, 0.1), (0.6, 0.2, 1.0), (0.2, 1.0, 0.5)]


def cyber_city(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    rg = rng(301)
    t = np.clip(py / H, 0, 1)
    sky = palette(t, [(0, hexcol('#05030f')), (0.45, hexcol('#1a0b33')), (0.75, hexcol('#5a1a5a')), (1, hexcol('#ff4a8a'))]) * 0.6
    # moon
    d = np.sqrt((px - W * 0.78) ** 2 + (py - H * 0.18) ** 2) / H
    sky += (smoothstep(0.075, 0.072, d) * 1.2 + np.exp(-d * 12) * 0.3)[..., None] * np.array([1.0, 0.6, 0.8], F32)
    img = sky
    fogc = np.array([0.55, 0.12, 0.45], F32)
    n_layers = 6
    for L in range(n_layers):
        f = L / (n_layers - 1)
        haze = (1 - f) ** 1.5 * 0.85
        ground = H * (0.62 + 0.42 * f)
        layer = np.zeros((H, W, 3), F32)
        mask = np.zeros((H, W), F32)
        x = rg.uniform(-0.05, 0) * W
        while x < W:
            bw = rg.uniform(60, 220) * S * (0.5 + 1.2 * f)
            bh = rg.uniform(0.15, 0.6) * H * (0.7 + 0.6 * f) * (2.0 if rg.random() < 0.06 else 1.0)
            x0, x1 = max(int(x), 0), int(min(x + bw, W))
            y0 = int(max(ground - bh, 0))
            y1 = int(min(ground, H))
            if x1 > x0 and y1 > y0:
                body = np.array([0.012, 0.012, 0.025], F32) * (0.6 + 0.8 * rg.random())
                hh, ww = y1 - y0, x1 - x0
                layer[y0:y1, x0:x1] = body
                mask[y0:y1, x0:x1] = 1
                # windows
                cw = max(int(rg.uniform(6, 12) * S * (0.6 + f)), 2)
                ch = max(int(cw * rg.uniform(1.2, 2.0)), 3)
                ny, nx = hh // ch + 1, ww // cw + 1
                p_on = rg.uniform(0.04, 0.3)
                on = rg.random((ny, nx)) < p_on
                # occasional whole-floor stripes
                stripes = rg.random(ny) < 0.03
                on[stripes] = True
                pal = np.array([(1.0, 0.8, 0.55), (0.6, 0.85, 1.0), (1.0, 0.4, 0.8), (0.5, 1.0, 0.9)], F32)
                cidx = rg.integers(0, len(pal), (ny, nx)) * (rg.random((ny, nx)) < 0.4)
                bright = rg.uniform(0.3, 1.5, (ny, nx)).astype(F32)
                ly, lx = np.mgrid[0:hh, 0:ww]
                cy, cx = ly // ch, lx // cw
                fy, fx = (ly % ch) / ch, (lx % cw) / cw
                win = (fx > 0.2) & (fx < 0.85) & (fy > 0.25) & (fy < 0.85)
                lit = on[cy, cx] & win
                wc = pal[cidx[cy, cx]] * bright[cy, cx][..., None]
                layer[y0:y1, x0:x1] += lit[..., None] * wc * (0.15 + 0.35 * f)
                # rooftop details
                if rg.random() < 0.5:
                    ax = x0 + ww // 2
                    ah = int(rg.uniform(0.02, 0.1) * H)
                    cv2.line(layer, (ax, y0), (ax, max(y0 - ah, 0)), (0.02, 0.02, 0.03), max(int(3 * S), 1))
                    cv2.line(mask, (ax, y0), (ax, max(y0 - ah, 0)), 1.0, max(int(3 * S), 1))
                    cv2.circle(layer, (ax, max(y0 - ah, 0)), max(int(4 * S), 1), (3.0, 0.2, 0.2), -1, cv2.LINE_AA)
                # neon signs on nearer layers
                if f > 0.3 and rg.random() < 0.55:
                    col = np.array(NEON[rg.integers(0, len(NEON))], F32)
                    if rg.random() < 0.5:
                        sw_, sh_ = int(rg.uniform(12, 26) * S * (0.5 + f)), int(rg.uniform(0.08, 0.25) * H * f)
                        sx_ = x0 + int(rg.uniform(0.1, 0.7) * ww)
                        sy_ = y0 + int(rg.uniform(0.05, 0.4) * hh)
                    else:
                        sw_, sh_ = int(rg.uniform(0.4, 0.9) * ww), int(rg.uniform(10, 22) * S * (0.5 + f))
                        sx_ = x0 + int(rg.uniform(0.05, 0.3) * ww)
                        sy_ = y0 + int(rg.uniform(0.05, 0.6) * hh)
                    sy2, sx2 = min(sy_ + sh_, H), min(sx_ + sw_, W)
                    if sy2 > sy_ and sx2 > sx_:
                        # glyph-like blocks inside the sign
                        gh, gw = sy2 - sy_, sx2 - sx_
                        g = (rg.random((gh // max(int(4 * S), 1) + 1, gw // max(int(4 * S), 1) + 1)) < 0.55).astype(F32)
                        g = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_NEAREST)
                        layer[sy_:sy2, sx_:sx2] = col * (0.6 + 2.4 * g[..., None])
                        mask[sy_:sy2, sx_:sx2] = 1
            x += bw * rg.uniform(0.85, 1.05)
        layer = layer * (1 - haze) + fogc * haze * 0.16 * mask[..., None]
        img = img * (1 - mask[..., None]) + layer * mask[..., None]
        # fog bank lit by the city below each layer
        fog_n = fbm(W, H, 500 * S, 6, seed=310 + L)
        fogl = smoothstep(ground - 0.25 * H, ground, py) * (0.18 + 0.1 * fog_n) * (1 - 0.5 * f)
        img = img * (1 - fogl[..., None]) + fogc * 0.5 * fogl[..., None]
    # light glow from all emissive parts
    img += fast_blur(np.maximum(img - 0.4, 0), 25 * S) * 0.8
    # flying traffic light trails
    tr = np.zeros((H, W, 3), F32)
    for k in range(6):
        y0 = rg.uniform(0.3, 0.55) * H
        x0, x1 = -0.1 * W, 1.1 * W
        bend = rg.uniform(-0.08, 0.08) * H
        xs = np.linspace(x0, x1, 200)
        ys = y0 + bend * np.sin((xs - x0) / (x1 - x0) * np.pi) + rg.uniform(-0.03, 0.03) * H * (xs - x0) / (x1 - x0)
        col = (1.0, 0.25, 0.2) if k % 2 else (0.9, 0.95, 1.0)
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(tr, [pts], False, tuple(c * rg.uniform(0.2, 0.6) for c in col), max(int(1.5 * S), 1), cv2.LINE_AA)
    img += tr + blur(tr, 6 * S) * 2
    # rain
    rain = np.zeros((H, W), F32)
    for _ in range(int(4000 * W / 3840)):
        x0, y0 = rg.uniform(-0.1 * W, W), rg.uniform(0, H)
        ln = rg.uniform(25, 80) * S
        cv2.line(rain, (int(x0), int(y0)), (int(x0 + ln * 0.12), int(y0 + ln)), rg.uniform(0.03, 0.12), 1, cv2.LINE_AA)
    img += rain[..., None] * np.array([0.7, 0.6, 1.0], F32)
    return finish(img, exposure=1.3, bloom_amt=0.35, bloom_thr=0.6, vig=0.35, saturation=1.2)


# ----------------------------------------------------------------------------
# 19  Circuit board macro
# ----------------------------------------------------------------------------
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def put_text(img, text, org, font, scale, color, thick):
    """Anti-aliased text onto a float image (cv2.putText needs 8-bit)."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    pad = thick + 2
    x0, y0 = max(x - pad, 0), max(y - th - pad, 0)
    x1, y1 = min(x + tw + pad, img.shape[1]), min(y + base + pad, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.putText(m, text, (x - x0, y - y0), font, scale, 255, thick, cv2.LINE_AA)
    a = (m.astype(F32) / 255)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + np.array(color, F32) * a


def circuit(W, H):
    S = H / 2160
    rg = rng(321)
    # render on a larger canvas then tilt into perspective
    CW, CH = int(W * 1.3), int(H * 1.6)
    g = max(int(14 * S), 3)
    gw, gh = CW // g, CH // g
    occ = np.zeros((gh, gw), bool)
    chips = []
    big = (gw // 2 - 20, gh // 2 - 20, 40, 40)
    chips.append(big)
    for _ in range(40):
        w, h = rg.integers(5, 18), rg.integers(4, 14)
        x, y = rg.integers(2, gw - w - 2), rg.integers(2, gh - h - 2)
        if occ[y - 2:y + h + 2, x - 2:x + w + 2].any():
            continue
        chips.append((x, y, w, h))
        occ[y - 1:y + h + 1, x - 1:x + w + 1] = True
    bx, by, bw, bh = big
    occ[by - 1:by + bh + 1, bx - 1:bx + bw + 1] = True
    traces = []
    starts = []
    for (x, y, w, h) in chips:
        for i in range(0, w, 1):
            starts.append((x + i, y - 2, 6))
            starts.append((x + i, y + h + 1, 2))
        for j in range(0, h, 1):
            starts.append((x - 2, y + j, 4))
            starts.append((x + w + 1, y + j, 0))
    rg.shuffle(starts)
    for _ in range(1200):
        starts.append((rg.integers(0, gw), rg.integers(0, gh), rg.integers(0, 8)))
    for (x, y, d) in starts:
        if not (0 <= x < gw and 0 <= y < gh) or occ[y, x]:
            continue
        path = [(x, y)]
        occ[y, x] = True
        L = rg.integers(8, 90)
        for step in range(L):
            if rg.random() < 0.12:
                d = (d + rg.choice([-1, 1])) % 8
            nx, ny = x + DIRS[d][0], y + DIRS[d][1]
            if not (0 <= nx < gw and 0 <= ny < gh) or occ[ny, nx]:
                # try turning
                ok = False
                for dd in (1, -1, 2, -2):
                    d2 = (d + dd) % 8
                    nx, ny = x + DIRS[d2][0], y + DIRS[d2][1]
                    if 0 <= nx < gw and 0 <= ny < gh and not occ[ny, nx]:
                        d, ok = d2, True
                        break
                if not ok:
                    break
            # block diagonal crossing
            occ[ny, nx] = True
            x, y = nx, ny
            path.append((x, y))
        if len(path) > 3:
            traces.append(path)
    img = np.zeros((CH, CW, 3), F32)
    # substrate with fibre weave
    fib = fbm(CW, CH, 3 * S, 3, seed=322)
    weave = (np.sin(np.arange(CW)[None, :] / (5 * S)) * np.sin(np.arange(CH)[:, None] / (5 * S)))
    sub = np.array([0.006, 0.03, 0.025], F32) * (1 + 0.15 * fib[..., None] + 0.08 * weave[..., None].astype(F32))
    img[:] = sub
    tw = max(int(g * 0.38), 1)
    copper = np.zeros((CH, CW), np.uint8)
    pulses = np.zeros((CH, CW), F32)
    for p in traces:
        pts = (np.array(p) * g + g // 2).astype(np.int32)
        cv2.polylines(copper, [pts], False, 255, tw, cv2.LINE_AA)
        cv2.circle(copper, tuple(pts[-1]), int(tw * 1.4), 255, -1, cv2.LINE_AA)
        if rg.random() < 0.06:
            n = len(pts)
            a = rg.integers(0, max(n - 4, 1))
            b = min(a + rg.integers(3, 12), n)
            cv2.polylines(pulses, [pts[a:b]], False, 1.0, max(tw // 2, 1), cv2.LINE_AA)
            cv2.circle(pulses, tuple(pts[b - 1]), int(tw * 1.3), 1.0, -1, cv2.LINE_AA)
    cm = copper.astype(F32) / 255
    # trace bevel: highlight from blurred mask gradient
    bm = blur(cm, tw * 0.4)
    gy, gx = np.gradient(bm)
    hl = np.clip(-(gx + gy) * tw * 1.5, 0, 1)
    gold = np.array([0.42, 0.25, 0.07], F32)
    tcol = gold * (0.45 + 0.4 * fib[..., None] * 0.2) + hl[..., None] * np.array([1.0, 0.85, 0.5], F32) * 0.8
    img = img * (1 - cm[..., None]) + tcol * cm[..., None]
    # vias (holes) at trace ends
    for p in traces:
        if rg.random() < 0.6:
            c = tuple(int(v) for v in (np.array(p[-1]) * g + g // 2))
            cv2.circle(img, c, int(tw * 1.4), (0.6, 0.45, 0.18), -1, cv2.LINE_AA)
            cv2.circle(img, c, max(int(tw * 0.6), 1), (0.002, 0.002, 0.002), -1, cv2.LINE_AA)
    # chips
    for k, (x, y, w, h) in enumerate(chips):
        x0, y0, x1, y1 = x * g, y * g, (x + w) * g, (y + h) * g
        pin = max(int(g * 0.35), 1)
        for i in range(w):
            cx = x0 + i * g + g // 2
            cv2.rectangle(img, (cx - pin // 2, y0 - g // 2), (cx + pin // 2, y0), (0.7, 0.7, 0.72), -1)
            cv2.rectangle(img, (cx - pin // 2, y1), (cx + pin // 2, y1 + g // 2), (0.7, 0.7, 0.72), -1)
        for j in range(h):
            cy = y0 + j * g + g // 2
            cv2.rectangle(img, (x0 - g // 2, cy - pin // 2), (x0, cy + pin // 2), (0.7, 0.7, 0.72), -1)
            cv2.rectangle(img, (x1, cy - pin // 2), (x1 + g // 2, cy + pin // 2), (0.7, 0.7, 0.72), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0.012, 0.012, 0.014), -1)
        gl = np.linspace(0.02, 0.0, y1 - y0, dtype=F32)[:, None, None]
        img[y0:y1, x0:x1] += gl
        cv2.rectangle(img, (x0, y0), (x1, y1), (0.05, 0.05, 0.055), max(int(2 * S), 1), cv2.LINE_AA)
        cv2.circle(img, (x0 + g, y0 + g), max(g // 4, 1), (0.05, 0.05, 0.05), -1, cv2.LINE_AA)
        if w >= 6:
            label = ["NX-%d" % rg.integers(100, 999), "Q%dA" % rg.integers(10, 99), "CORE-%d" % k, "SYN %02d" % k][k % 4]
            fs = min(w, h * 3) * g / 160
            put_text(img, label, (x0 + g, y0 + (y1 - y0) // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0.18, 0.18, 0.2), max(int(fs * 2), 1))
    # central processor: heat-spreader with logo-like rings
    x0, y0, x1, y1 = bx * g, by * g, (bx + bw) * g, (by + bh) * g
    cv2.rectangle(img, (x0 + 2 * g, y0 + 2 * g), (x1 - 2 * g, y1 - 2 * g), (0.1, 0.105, 0.11), -1)
    brushed = fbm(x1 - x0 - 4 * g, y1 - y0 - 4 * g, 2 * S, 2, seed=323)
    img[y0 + 2 * g:y1 - 2 * g, x0 + 2 * g:x1 - 2 * g] *= (0.9 + 0.08 * brushed[..., None])
    c = ((x0 + x1) // 2, (y0 + y1) // 2)
    for r in range(3):
        cv2.circle(img, c, int((5 + r * 3) * g * 0.5), (0.1, 0.8, 1.0), max(int(3 * S), 1), cv2.LINE_AA)
    put_text(img, "QUANTA", (c[0] - int(4.2 * g), c[1] + int(9.5 * g)), cv2.FONT_HERSHEY_DUPLEX, g / 11, (0.12, 0.12, 0.13), max(int(g / 6), 1))
    # glowing data pulses
    pc = np.array([0.15, 0.9, 1.0], F32)
    img += pulses[..., None] * pc * 3 + blur(pulses, 4 * S)[..., None] * pc * 2 + fast_blur(pulses, 20 * S)[..., None] * pc * 1.5
    img += fast_blur(np.clip(img - 0.5, 0, None), 12 * S) * 0.3
    # perspective tilt
    src = np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]])
    dst = np.float32([[-W * 0.02, -H * 0.25], [W * 1.02, -H * 0.25], [W * 1.3, H * 1.1], [-W * 0.3, H * 1.1]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_AREA)
    # tilt-shift depth of field around the processor
    py = np.arange(H, dtype=F32)[:, None]
    focus = H * 0.55
    dof = np.clip(np.abs(py - focus) / (0.45 * H), 0, 1) ** 1.5
    b1 = blur(out, 4 * S)
    b2 = fast_blur(out, 14 * S)
    out = out * (1 - smoothstep(0, 0.5, dof))[..., None] + b1 * (smoothstep(0, 0.5, dof) - smoothstep(0.5, 1, dof))[..., None] + b2 * smoothstep(0.5, 1, dof)[..., None]
    return finish(out, exposure=1.4, bloom_amt=0.3, bloom_thr=0.7, vig=0.4, saturation=1.1)


# ----------------------------------------------------------------------------
# 20  Synthwave
# ----------------------------------------------------------------------------
def synthwave(W, H):
    S = H / 2160
    px, py = pixel_grid(W, H)
    hor = H * 0.56
    t = np.clip(py / hor, 0, 1)
    sky = palette(t, [(0, hexcol('#0a0220')), (0.45, hexcol('#3a0a5a')), (0.8, hexcol('#c0207a')), (1, hexcol('#ff7a3a'))]) * 0.8
    sky += starfield(W, H, int(8000 * W * H / (3840 * 2160)), seed=331, bright=0.5, spikes=False) * (1 - t)[..., None] ** 2
    # striped sun
    sx, sy, sr = W * 0.5, hor - 0.18 * H, 0.24 * H
    d = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
    disk = smoothstep(sr + 1.5, sr - 1.5, d)
    v = np.clip((py - (sy - sr)) / (2 * sr), 0, 1)
    scol = palette(v, [(0, hexcol('#fff07a')), (0.5, hexcol('#ff9a3a')), (1, hexcol('#ff2a7a'))]) * 3
    # horizontal cuts, thicker toward the bottom
    band = (py - sy) / sr
    period = 0.13
    ph = np.mod(band + 2.0, period) / period
    thick = np.clip((band - 0.05) * 0.55, 0, 0.6)
    cut = (ph < thick) & (band > 0.05)
    disk = disk * (~cut)
    img = sky * (1 - disk[..., None]) + scol * disk[..., None]
    img += (np.exp(-np.maximum(d - sr, 0) / (0.08 * H)) * (d > sr) * 0.5)[..., None] * np.array([1.0, 0.3, 0.5], F32)
    img[py >= hor] = 0
    # perspective grid floor
    floor = np.zeros((H, W, 3), F32)
    f = 1.4
    cam_h = 1.0
    Y = (hor - py) / (H / 2)
    X = (px - W / 2) / (H / 2)
    below = py > hor + 0.5
    z = np.where(below, cam_h * f / np.maximum(-Y, 1e-4), 1e6)
    x = X * z / f
    pix = z * 2 / H / f
    gz = np.abs(np.mod(z * 0.8 + 0.5, 1.0) - 0.5) / (0.8 * pix / np.maximum(-Y, 1e-3) * 0.25 + 1e-6)
    gx_ = np.abs(np.mod(x * 0.8 + 0.5, 1.0) - 0.5) / (0.8 * pix + 1e-6)
    line = np.clip(1.2 - np.minimum(gz, gx_ * 1.0) * 0.8, 0, 1)
    fade = np.exp(-z / 60)
    floor += (line * fade * below)[..., None] * np.array([1.0, 0.15, 0.85], F32) * 2.5
    floor += (below * np.exp(-(py - hor) / (0.05 * H)) * 0.4)[..., None] * np.array([1.0, 0.25, 0.5], F32)
    # sun reflection on the floor
    refl = np.exp(-np.abs(px - sx) / (0.12 * W)) * np.exp(-(py - hor) / (0.2 * H)) * below
    floor += refl[..., None] * np.array([1.0, 0.4, 0.3], F32) * 0.25
    img += floor
    # wireframe mountains left and right (painter's algorithm, back to front)
    rgm = rng(332)
    hn = tile_noise(512, 64, 6, 333)
    nz_, nx_ = 70, 120
    zs = np.geomspace(60, 2.0, nz_)
    xs = np.linspace(-40, 40, nx_)
    for zi, zz in enumerate(zs):
        ax = np.abs(xs)
        mh = np.clip((ax - 4.0) / 4.0, 0, 1) ** 1.4 * (2.2 + 1.8 * hn[(zi * 7) % 512, (np.arange(nx_) * 4) % 512]) * np.clip(zz / 8, 0.4, 1.5)
        mh = np.maximum(mh, 0)
        sxp = W / 2 + xs / zz * f * H / 2
        syp = hor + (cam_h - mh) / zz * f * H / 2
        poly = np.concatenate([np.stack([sxp, syp], 1), [[sxp[-1], H * 1.5], [sxp[0], H * 1.5]]]).astype(np.int32)
        pts = np.stack([sxp, syp], 1).astype(np.int32)
        m = mh > 0.02
        # only hide where the mountain is raised
        raised = np.zeros((H, W), np.uint8)
        for side in (xs < 0, xs > 0):
            idx = np.where(side & m)[0]
            if len(idx) < 2:
                continue
            seg = np.concatenate([pts[idx], [[pts[idx[-1]][0], int(hor + cam_h / zz * f * H / 2)], [pts[idx[0]][0], int(hor + cam_h / zz * f * H / 2)]]])
            cv2.fillPoly(raised, [seg.astype(np.int32)], 255)
        cv2.copyTo(np.full_like(img, 0.0) + np.array([0.01, 0.0, 0.03], F32), raised, img)
        alpha = float(np.exp(-zz / 45))
        c = (0.1 * alpha + 0.02, 0.9 * alpha + 0.05, 1.0 * alpha + 0.1)
        for side in (xs < 0, xs > 0):
            idx = np.where(side & (mh > 0.02))[0]
            if len(idx) >= 2:
                cv2.polylines(img, [pts[idx]], False, c, max(int(2 * S), 1), cv2.LINE_AA)
    # longitudinal mountain lines (connect rows)
    # (approximate by drawing ridgelines of fixed x across depth)
    for xi in range(0, nx_, 3):
        xx = xs[xi]
        if abs(xx) < 4.5:
            continue
        pts = []
        for zi, zz in enumerate(zs):
            mh = max(np.clip((abs(xx) - 4.0) / 4.0, 0, 1) ** 1.4 * (2.2 + 1.8 * hn[(zi * 7) % 512, (xi * 4) % 512]) * np.clip(zz / 8, 0.4, 1.5), 0)
            pts.append((W / 2 + xx / zz * f * H / 2, hor + (cam_h - mh) / zz * f * H / 2))
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], False, (0.05, 0.35, 0.45), max(int(1 * S), 1), cv2.LINE_AA)
    img += fast_blur(np.maximum(img - 0.3, 0), 8 * S) * 1.0
    # chromatic aberration + scanlines
    out = img.copy()
    sh = max(int(2 * S), 1)
    out[..., 0] = np.roll(img[..., 0], sh, 1)
    out[..., 2] = np.roll(img[..., 2], -sh, 1)
    scan = 1 - 0.06 * (np.mod(py, max(int(4 * S), 2)) < max(int(1.5 * S), 1))
    out *= scan[..., None]
    return finish(out, exposure=1.2, bloom_amt=0.35, bloom_thr=0.6, vig=0.35, saturation=1.15)


# ----------------------------------------------------------------------------
# 22  Neural network in 3D
# ----------------------------------------------------------------------------
def neural(W, H):
    S = H / 2160
    rg = rng(341)
    X, Y = coords(W, H)
    px, py = pixel_grid(W, H)
    bg = palette(np.clip(np.sqrt((X * 0.6) ** 2 + Y ** 2) / 1.4, 0, 1), [(0, hexcol('#0c1a3a')), (0.6, hexcol('#050a1a')), (1, hexcol('#010208'))])
    img = bg.copy()
    layers = [24, 48, 64, 64, 48, 32, 12]
    nodes = []
    for li, n in enumerate(layers):
        x = (li - (len(layers) - 1) / 2) * 1.1
        for k in range(n):
            a = rg.uniform(0, 2 * np.pi)
            r = math.sqrt(rg.uniform(0, 1)) * (0.4 + 0.35 * math.sqrt(n / 64)) * 1.6
            nodes.append((x + rg.normal(0, 0.04), r * math.cos(a), r * math.sin(a), li))
    nodes = np.array(nodes)
    # rotate the whole network for a 3D view
    ay, ax_ = math.radians(-32), math.radians(14)
    R1 = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    R2 = np.array([[1, 0, 0], [0, math.cos(ax_), -math.sin(ax_)], [0, math.sin(ax_), math.cos(ax_)]])
    P = nodes[:, :3] @ R1.T @ R2.T
    cz = 6.0
    z = P[:, 2] + cz
    f = 3.2
    sx = W / 2 + P[:, 0] / z * f * H / 2 * 0.72
    sy = H / 2 - P[:, 1] / z * f * H / 2 * 0.72
    lay = nodes[:, 3].astype(int)
    lcol = palette(lay / (len(layers) - 1), [(0, (0.1, 0.8, 1.0)), (0.5, (0.6, 0.3, 1.0)), (1, (1.0, 0.25, 0.6))])
    # depth bins for depth of field
    zmin, zmax = z.min(), z.max()
    nb = 6
    bins = [np.zeros((H, W, 3), F32) for _ in range(nb)]
    def zbin(zz):
        return int(np.clip((zz - zmin) / (zmax - zmin + 1e-6) * nb, 0, nb - 1))
    starts = np.cumsum([0] + layers)
    pulses = []
    for li in range(len(layers) - 1):
        a_idx = np.arange(starts[li], starts[li + 1])
        b_idx = np.arange(starts[li + 1], starts[li + 2])
        for a in a_idx:
            k = rg.integers(3, 9)
            for b in rg.choice(b_idx, size=min(k, len(b_idx)), replace=False):
                wgt = rg.uniform(0.02, 0.12) * (3 if rg.random() < 0.05 else 1)
                c = tuple(float(v) * wgt for v in (lcol[a] + lcol[b]) / 2)
                zb = zbin((z[a] + z[b]) / 2)
                cv2.line(bins[zb], (int(sx[a]), int(sy[a])), (int(sx[b]), int(sy[b])), c, max(int(1.2 * S), 1), cv2.LINE_AA)
                if rg.random() < 0.04:
                    pulses.append((a, b, rg.uniform(0.1, 0.9)))
    for i in range(len(nodes)):
        r = max(int(7 * S * 6 / z[i]), 1)
        c = tuple(float(v) for v in lcol[i] * 1.6)
        zb = zbin(z[i])
        cv2.circle(bins[zb], (int(sx[i]), int(sy[i])), r, c, -1, cv2.LINE_AA)
        cv2.circle(bins[zb], (int(sx[i]), int(sy[i])), max(r // 2, 1), (3.0, 3.0, 3.0), -1, cv2.LINE_AA)
    for (a, b, tt) in pulses:
        x_ = sx[a] + (sx[b] - sx[a]) * tt
        y_ = sy[a] + (sy[b] - sy[a]) * tt
        zb = zbin(z[a] + (z[b] - z[a]) * tt)
        cv2.circle(bins[zb], (int(x_), int(y_)), max(int(4 * S), 1), (4.0, 4.0, 4.0), -1, cv2.LINE_AA)
    focus = zbin((zmin + zmax) / 2)
    for k in range(nb):
        sig = abs(k - focus) * 3.5 * S
        layer = bins[k] if sig < 0.5 else fast_blur(bins[k], sig) if sig >= 6 else blur(bins[k], sig)
        img += layer
    img += fast_blur(sum(bins), 20 * S) * 0.8
    # floating particles
    n = int(3000 * W * H / (3840 * 2160))
    buf = np.zeros((H, W, 3), F32)
    splat(buf, rg.uniform(0, W, n), rg.uniform(0, H, n), (np.array([[0.3, 0.6, 1.0]], F32) * rg.uniform(0.05, 0.5, (n, 1))).astype(F32))
    img += blur(buf, 1.5 * S) * 3 + fast_blur(buf, 10 * S) * 10
    return finish(img, exposure=1.3, bloom_amt=0.3, bloom_thr=0.7, vig=0.35, saturation=1.15)
