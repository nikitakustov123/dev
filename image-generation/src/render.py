"""Render the gallery.

usage: python render.py [--scale 1.0] [--only 1,5,12] [--out ../images]
"""
import argparse
import os
import sys
import time
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save  # noqa: E402

SCENES = [
    # id, file name, module, function
    (1, "nebula", "space", "nebula"),
    (2, "spiral-galaxy", "space", "galaxy"),
    (3, "ringed-giant", "space", "ringed_planet"),
    (4, "black-hole", "blackhole", "black_hole"),
    (5, "milky-way-lake", "space", "milky_way"),
    (6, "alien-world", "space", "alien_world"),
    (7, "solar-eclipse", "space", "eclipse"),
    (8, "aurora", "nature", "aurora"),
    (9, "misty-mountains", "nature", "misty_mountains"),
    (10, "ocean-sunset", "ocean", "ocean_sunset"),
    (11, "forest-light", "nature", "forest"),
    (12, "aerial-terrain", "nature", "aerial_terrain"),
    (13, "underwater", "nature", "underwater"),
    (14, "desert-dunes", "dunes", "dunes"),
    (15, "lightning-storm", "nature", "lightning"),
    (16, "sakura-tree", "nature", "sakura"),
    (17, "snowflake", "nature", "snowflake"),
    (18, "cyberpunk-city", "tech", "cyber_city"),
    (19, "circuit-board", "tech", "circuit"),
    (20, "synthwave", "tech", "synthwave"),
    (21, "chrome-spheres", "raytrace", "spheres"),
    (22, "neural-network", "tech", "neural"),
    (23, "clifford-attractor", "abstract", "attractor"),
    (24, "light-painting", "abstract", "harmonograph"),
    (25, "mandelbrot", "abstract", "mandelbrot"),
    (26, "julia-set", "abstract", "julia"),
    (27, "reaction-diffusion", "abstract", "reaction_diffusion"),
    (28, "gold-marble", "abstract", "marble"),
    (29, "geode", "abstract", "geode"),
    (30, "hyperspace", "abstract", "hyperspace"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "images"))
    a = ap.parse_args()
    W, H = int(round(3840 * a.scale)), int(round(2160 * a.scale))
    only = {int(x) for x in a.only.split(",") if x.strip()} if a.only else None
    os.makedirs(a.out, exist_ok=True)
    for sid, name, mod, fn in SCENES:
        if only and sid not in only:
            continue
        t0 = time.time()
        f = getattr(importlib.import_module(mod), fn)
        img = f(W, H)
        path = os.path.join(a.out, f"{sid:02d}-{name}.jpg")
        save(img, path)
        print(f"[{sid:02d}] {name:22s} {W}x{H}  {time.time() - t0:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
