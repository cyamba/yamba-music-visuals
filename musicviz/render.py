"""Audio-reactive frame renderer (the look made for "Forgotten Mysteries").

Layers (back to front):
  feedback buffer  – previous frame zoomed/rotated & decayed  -> tunnel trails
  lace field       – kaleidoscopic domain-warped interference ridges (mid / bass driven)
  starfield        – warp-speed streaks (loudness + section drive speed)
  rune rings       – 3 counter-rotating tick rings, radius pulses on bass
  spectrum halo    – 128 mirrored radial bars from the log spectrum
  shockwaves       – expanding rings spawned on detected kicks
  orb + bloom      – glowing core, global bloom, filmic tonemap, vignette

Everything is a pure function of the frame index plus the feedback buffer, so the
track can be rendered in parallel chunks with a short warm-up per chunk.

usage: python -m musicviz.render features.npz out.mp4 --w 1280 --h 720 [--start S --end S]
(normally driven by `python -m musicviz`, see pipeline.py)
"""
import argparse, subprocess, math
import numpy as np
import cv2


def hexrgb(h):
    h = h.lstrip('#'); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def make_lut(stops):
    xs = [s[0] for s in stops]; cs = np.array([hexrgb(s[1]) for s in stops])
    t = np.linspace(0, 1, 256)
    rgb = np.stack([np.interp(t, xs, cs[:, c]) for c in range(3)], 1)
    return rgb[:, ::-1].astype(np.float32)  # BGR


PALETTES = {
    "mystery": (  # calm: indigo -> teal -> pale gold ; intense: violet -> magenta -> amber
        [(0, '#04030c'), (.25, '#1d1150'), (.5, '#0e6a8e'), (.75, '#5fe0cf'), (1, '#f6f1d6')],
        [(0, '#060312'), (.25, '#26105e'), (.5, '#5a3fd0'), (.75, '#b38cff'), (1, '#f4eeff')],
        [(0, '#0a0210'), (.25, '#4b0f5e'), (.5, '#c4307a'), (.75, '#f3a33a'), (1, '#fff6d8')]),
    "ember": (
        [(0, '#060304'), (.25, '#3a0d12'), (.5, '#a3361c'), (.75, '#f0a040'), (1, '#fff3d0')],
        [(0, '#070206'), (.25, '#4a0c2a'), (.5, '#c42a4a'), (.75, '#ff8a5a'), (1, '#fff0e0')],
        [(0, '#08020a'), (.25, '#52105a'), (.5, '#e0354f'), (.75, '#ffc04a'), (1, '#ffffff')]),
    "abyss": (
        [(0, '#010409'), (.25, '#062a3f'), (.5, '#0b7a8a'), (.75, '#7ff0e0'), (1, '#e8fffb')],
        [(0, '#02040c'), (.25, '#0b2457'), (.5, '#1f6fd0'), (.75, '#86c8ff'), (1, '#f0f8ff')],
        [(0, '#02030a'), (.25, '#132b6e'), (.5, '#3b6fe0'), (.75, '#a6d4ff'), (1, '#ffffff')]),
}


class Viz:
    def __init__(self, F, W, H, palette="mystery", seed=7, symmetry=6):
        self.F, self.W, self.H = F, W, H
        self.fps = int(F["fps"]); self.n = int(F["n_frames"])
        self.S = min(W, H)
        self.k = symmetry
        rng = np.random.default_rng(seed)
        fp = 1 / self.fps

        # cumulative "phases" so motion speeds up with energy but never jumps
        self.P_bass = np.cumsum(0.4 + 1.8 * F["bass"]) * fp
        self.P_mid = np.cumsum(0.15 + 0.9 * F["mid"]) * fp
        self.D_star = np.cumsum(0.12 + 1.1 * F["rms"] ** 1.5 + 0.7 * F["section"]) * fp

        # kick events (rising edges) -> shockwaves
        k = F["kick"] > 0.45
        self.kicks = np.flatnonzero(k[1:] & ~k[:-1]) + 1

        # pixel grid in units of S, centred
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        self.x = (xs - W / 2) / self.S; self.y = (ys - H / 2) / self.S
        self.r = np.sqrt(self.x ** 2 + self.y ** 2)
        self.th = np.arctan2(self.y, self.x)
        self.vign = np.clip(1.15 - 0.55 * (self.r / (0.5 * max(W, H) / self.S)) ** 2, 0, 1)[..., None]

        self.luts = [make_lut(p) for p in PALETTES[palette]]  # calm -> mid -> intense

        # stars
        N = 420
        self.sx = rng.uniform(-1, 1, N).astype(np.float32)
        self.sy = rng.uniform(-1, 1, N).astype(np.float32)
        self.sz0 = rng.uniform(0, 1, N).astype(np.float32)

        # rune rings: radius (×S), tick count, direction, per-tick length pattern
        self.rings = []
        for j, (rad, cnt) in enumerate([(0.17, 36), (0.24, 60), (0.31, 90)]):
            pat = rng.choice([0, 0.35, 0.6, 1.0], cnt, p=[.25, .35, .25, .15])
            self.rings.append((rad, cnt, (-1) ** j * (0.25 + 0.15 * j), pat))
        self.buf = np.zeros((H, W, 3), np.float32)

    def lut(self, i):
        m = float(self.F["section"][i]) ** 1.2 * (len(self.luts) - 1)
        j = min(int(m), len(self.luts) - 2); f = m - j
        return self.luts[j] * (1 - f) + self.luts[j + 1] * f

    # ---- layers -------------------------------------------------------------
    def lace(self, i, lut):
        F = self.F; t = i / self.fps
        mid, bass, sec = F["mid"][i], F["bass"][i], F["section"][i]
        seg = 2 * np.pi / self.k
        a = np.mod(self.th + 0.6 * self.P_mid[i], seg)
        a = np.abs(a - seg / 2)
        x = self.r * np.cos(a); y = self.r * np.sin(a)
        x2 = x + 0.28 * np.sin(3.1 * y + 0.40 * t)
        y2 = y + 0.28 * np.sin(2.7 * x - 0.31 * t)
        f = (np.sin(6.0 * x2 + 0.5 * t) + np.sin(7.0 * y2 - 0.37 * t + 2.0 * mid)
             + np.sin(11.0 * self.r - 2.4 * self.P_bass[i]))
        ridge = (1 - np.abs(np.sin(1.6 * f))) ** (7 - 3 * sec)       # thin glowing filaments
        v = np.clip(0.5 + f / 6, 0, 1)
        idx = (90 + 160 * v).astype(np.int32)
        col = lut[idx]
        amp = (0.10 + 0.55 * mid + 0.25 * sec) * np.exp(-1.3 * self.r)
        return col * (ridge * amp)[..., None]

    def geometry(self, i, lut):
        F = self.F; S = self.S; cx, cy = self.W / 2, self.H / 2
        g = np.zeros((self.H, self.W, 3), np.uint8)
        bass, mid, high, rms = (float(F[k][i]) for k in ("bass", "mid", "high", "rms"))
        AA = cv2.LINE_AA
        c8 = lambda c, b: tuple(float(min(255, v * 255 * b)) for v in c)

        # starfield streaks
        def proj(D):
            z = np.mod(self.sz0 - 0.25 * D, 1.0) + 0.02
            return self.sx / z * 0.35 * S + cx, self.sy / z * 0.35 * S + cy, z
        x1, y1, z1 = proj(self.D_star[i]); x0, y0, z0 = proj(self.D_star[max(i - 2, 0)])
        ok = (z0 > z1) & (np.abs(x1 - cx) < self.W) & (np.abs(y1 - cy) < self.H)
        sc = lut[215]
        for a, b, c, d, z in zip(x0[ok], y0[ok], x1[ok], y1[ok], z1[ok]):
            br = min(1.0, (1.05 - z) ** 2 * (0.5 + rms))
            cv2.line(g, (int(a), int(b)), (int(c), int(d)), c8(sc, br), 1, AA)

        # rune rings
        pulse = 1 + 0.07 * bass + 0.05 * float(F["kick"][i])
        for j, (rad, cnt, spd, pat) in enumerate(self.rings):
            R = rad * S * pulse
            rot = spd * self.P_mid[i]
            br = 0.35 + 0.65 * (mid if j == 1 else high if j == 2 else bass)
            col = c8(lut[175 + 18 * j], br)
            cv2.circle(g, (int(cx), int(cy)), int(R), c8(lut[150], 0.5 * br), 1, AA)
            ang = rot + np.arange(cnt) * 2 * np.pi / cnt
            for ag, L in zip(ang, pat):
                if L == 0: continue
                l = L * 0.035 * S
                p0 = (int(cx + R * math.cos(ag)), int(cy + R * math.sin(ag)))
                p1 = (int(cx + (R + l) * math.cos(ag)), int(cy + (R + l) * math.sin(ag)))
                cv2.line(g, p0, p1, col, 1 + (L == 1.0), AA)

        # spectrum halo (mirrored, starting at the top)
        spec = F["spec"][i]; nb = len(spec)
        R0 = 0.36 * S * pulse
        for b in range(nb):
            v = spec[b] ** 1.4
            if v < 0.03: continue
            L = v * 0.22 * S
            col = c8(lut[100 + int(130 * b / nb)], 0.4 + 0.6 * v)
            for sgn in (1, -1):
                ag = -np.pi / 2 + sgn * (b + 0.5) * np.pi / nb
                ca, sa = math.cos(ag), math.sin(ag)
                cv2.line(g, (int(cx + R0 * ca), int(cy + R0 * sa)),
                         (int(cx + (R0 + L) * ca), int(cy + (R0 + L) * sa)), col, 2, AA)

        # shockwaves from recent kicks
        life = 1.3 * self.fps
        for kf in self.kicks[(self.kicks <= i) & (self.kicks > i - life)]:
            age = (i - kf) / life
            R = (0.12 + 0.9 * age ** 0.7) * S
            cv2.circle(g, (int(cx), int(cy)), int(R), c8(lut[225], (1 - age) ** 2 * F["kick"][kf]), 2, AA)
        return g.astype(np.float32) / 255

    # ---- compose ------------------------------------------------------------
    def frame(self, i):
        F = self.F; lut = self.lut(i)
        bass, mid, sec = float(F["bass"][i]), float(F["mid"][i]), float(F["section"][i])
        M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), 0.15 + 0.5 * mid * (1 if (i // (self.fps * 16)) % 2 else -1),
                                    1.004 + 0.022 * bass + 0.01 * sec)
        decay = 0.84 + 0.10 * sec
        self.buf = cv2.warpAffine(self.buf, M, (self.W, self.H), flags=cv2.INTER_LINEAR) * decay
        lace = self.lace(i, lut); geo = self.geometry(i, lut)
        self.buf += (1 - decay) * 3.0 * (0.4 * lace + 0.6 * geo)   # steady-state gain ~3x, not 1/(1-decay)
        img = self.buf + 0.55 * lace + 0.65 * geo

        # core orb
        k = float(F["kick"][i])
        rad = 0.05 * (1 + 0.6 * bass + 0.6 * k)
        orb = np.exp(-(self.r / rad) ** 2)[..., None] * (lut[250] * (0.5 + 0.9 * k + 0.4 * bass))
        img += orb

        # bloom
        sm = cv2.resize(img, (self.W // 4, self.H // 4), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 5)
        img += cv2.resize(sm, (self.W, self.H), interpolation=cv2.INTER_LINEAR) * (0.45 + 0.45 * sec)

        img *= 1 + 0.25 * float(F["flux"][i])
        img = 1 - np.exp(-img * 1.25)                    # filmic tonemap
        lum = img.mean(2, keepdims=True)
        img = lum + (img - lum) * 1.35                   # restore saturation
        img *= self.vign
        return np.clip(img * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features"); ap.add_argument("out")
    ap.add_argument("--w", type=int, default=1280); ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--start", type=float, default=0); ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--palette", default="mystery"); ap.add_argument("--warmup", type=float, default=4)
    ap.add_argument("--crf", type=int, default=14); ap.add_argument("--preset", default="veryfast")
    a = ap.parse_args()
    F = dict(np.load(a.features))
    v = Viz(F, a.w, a.h, a.palette)
    s = int(a.start * v.fps); e = v.n if a.end is None else min(v.n, int(a.end * v.fps))
    for i in range(max(0, s - int(a.warmup * v.fps)), s):   # warm the feedback buffer
        v.frame(i)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{a.w}x{a.h}", "-r", str(v.fps), "-i", "-", "-c:v", "libx264",
                            "-preset", a.preset, "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.out],
                           stdin=subprocess.PIPE)
    import time; t0 = time.time()
    for i in range(s, e):
        enc.stdin.write(v.frame(i).tobytes())
        if (i - s) % 300 == 0:
            el = time.time() - t0
            print(f"{a.out}: {i - s}/{e - s}  {el / max(1, i - s) * 1000:.0f} ms/frame", flush=True)
    enc.stdin.close(); enc.wait()
    print("done", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
