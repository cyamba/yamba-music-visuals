"""Audio feature extraction for music visuals.

Produces one feature row per video frame (fps-aligned), saved as .npz:
  spec   (n_frames, n_bands)  log-band spectrum, normalised 0..1, smoothed
  rms    (n_frames,)          loudness envelope 0..1
  bass / mid / high           band energies 0..1 (attack/release smoothed)
  kick                        bass onset impulses 0..1 (fast decay)
  flux                        broadband onset strength 0..1
  section                     slow "intensity" curve 0..1 (for scene/palette changes)
"""
import subprocess, sys
import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d

SR = 44100


def load_mono(path, sr=SR):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def norm01(x, lo=2, hi=98):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def env(x, attack, release):
    """One-pole attack/release follower (per-frame coefficients 0..1)."""
    y = np.empty_like(x); s = 0.0
    for i, v in enumerate(x):
        k = attack if v > s else release
        s += (v - s) * k
        y[i] = s
    return y


def analyze(path, fps=30, n_bands=64):
    y = load_mono(path)
    hop = SR // fps if SR % fps == 0 else SR / fps
    n_fft = 4096
    n_frames = int(len(y) / SR * fps)
    win = np.hanning(n_fft).astype(np.float32)
    pad = np.pad(y, (n_fft // 2, n_fft // 2))
    idx = (np.arange(n_frames) * hop).astype(int)
    frames = np.stack([pad[i:i + n_fft] for i in idx]) * win
    mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)

    # log-spaced bands 30 Hz .. 16 kHz
    edges = np.geomspace(30, 16000, n_bands + 1)
    bi = np.searchsorted(freqs, edges)
    bands = np.stack([mag[:, bi[i]:max(bi[i + 1], bi[i] + 1)].mean(1) for i in range(n_bands)], 1)
    lb = np.log1p(bands * 10)
    lb = (lb - np.percentile(lb, 5, axis=0)) / (np.percentile(lb, 99, axis=0) - np.percentile(lb, 5, axis=0) + 1e-9)
    spec = np.clip(lb, 0, 1)
    # gentle temporal smoothing per band (fast attack, slower release)
    spec = np.stack([env(spec[:, b], 0.7, 0.25) for b in range(n_bands)], 1)

    def adaptive(x, secs=15):
        """Blend global and local (rolling-window) normalisation so quiet
        passages still move while loud ones still hit harder."""
        w = int(fps * secs)
        lo = uniform_filter1d(x, w)  # local mean
        hi = maximum_filter1d(x, w)
        hi = uniform_filter1d(hi, w)
        local = np.clip((x - lo * 0.9) / (hi - lo * 0.9 + 1e-9), 0, 1)
        return 0.6 * local + 0.4 * norm01(x)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return adaptive(np.log1p(mag[:, m].mean(1) * 100))

    bass = env(band(30, 160), 0.8, 0.18)
    mid = env(band(160, 2500), 0.6, 0.15)
    high = env(band(2500, 14000), 0.6, 0.2)
    rms = env(norm01(np.sqrt((frames ** 2).mean(1))), 0.5, 0.08)

    # onsets: positive spectral flux
    lmag = np.log1p(mag)
    d = np.maximum(np.diff(lmag, axis=0, prepend=lmag[:1]), 0)
    lowm = freqs < 150
    kflux = d[:, lowm].sum(1); bflux = d.sum(1)

    def peaks(f, thr_win=fps // 2, k=1.4):
        loc = uniform_filter1d(f, thr_win * 2) * k
        is_pk = (f > loc) & (f == maximum_filter1d(f, 5))
        out = np.zeros_like(f)
        out[is_pk] = norm01(f)[is_pk]
        return env(out, 1.0, 0.25)  # instant hit, ~4 frame decay

    kick = peaks(kflux)
    flux = peaks(bflux)
    section = norm01(uniform_filter1d(rms, fps * 8))  # ~8 s intensity curve
    return dict(fps=fps, n_frames=n_frames, spec=spec.astype(np.float32), rms=rms, bass=bass,
                mid=mid, high=high, kick=kick, flux=flux, section=section)


if __name__ == "__main__":
    # usage: python -m musicviz.analyze song.mp3 features.npz   (prints a coarse structure map)
    src, dst = sys.argv[1], sys.argv[2]
    f = analyze(src)
    np.savez_compressed(dst, **f)
    n = f["n_frames"]; fps = f["fps"]
    print(f"frames={n} ({n/fps:.1f}s)  kicks>0.5: {(np.diff((f['kick']>0.5).astype(int))==1).sum()}")
    # coarse structure: intensity per 10 s
    for s in range(0, n, fps * 10):
        sl = slice(s, s + fps * 10)
        print(f"{s/fps:5.0f}s  rms {f['rms'][sl].mean():.2f}  bass {f['bass'][sl].mean():.2f}  "
              f"high {f['high'][sl].mean():.2f}  sect {f['section'][sl].mean():.2f}  "
              + "#" * int(f['rms'][sl].mean() * 40))
