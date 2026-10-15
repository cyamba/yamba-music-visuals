"""One command: song in, finished music video out.

    python -m musicviz song.mp3 -o video.mp4 --title "Forgotten Mysteries" \
        --artist "Christopher Yamba" --year 2007

Steps: analyse audio -> render frames in parallel chunks -> join -> title cards,
fades and audio -> web-ready H.264/AAC MP4 (faststart).
"""
import argparse, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np

from . import analyze, titles
from .render import PALETTES


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="musicviz", description="Audio-reactive music video generator")
    ap.add_argument("audio", help="input audio file (mp3, wav, flac, ...)")
    ap.add_argument("-o", "--out", default="video.mp4", help="output MP4 (default: video.mp4)")
    ap.add_argument("--title", default="", help="track title for the title cards")
    ap.add_argument("--artist", default="")
    ap.add_argument("--year", default="")
    ap.add_argument("--tagline", default="", help="extra line on the outro card")
    ap.add_argument("--url", default="", help="small URL line on the outro card")
    ap.add_argument("--no-titles", action="store_true", help="skip intro/outro cards")
    ap.add_argument("--size", default="1280x720", help="WxH, e.g. 1920x1080 or 1080x1920 (vertical)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--palette", default="mystery", choices=sorted(PALETTES))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2, help="parallel render processes")
    ap.add_argument("--preview", metavar="START:END", help="render only this window in seconds, e.g. 130:150")
    ap.add_argument("--crf", type=int, default=23, help="x264 quality (lower = better/bigger)")
    ap.add_argument("--maxrate", default="1600k", help="x264 bitrate cap; 'none' to disable")
    ap.add_argument("--work", help="keep intermediate files in this folder")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    W, H = map(int, a.size.lower().split("x"))
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="musicviz_"))
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. analysis
    print("· analysing audio", flush=True)
    feats = analyze.analyze(a.audio, fps=a.fps)
    fpath = work / "features.npz"
    np.savez_compressed(fpath, **feats)
    n, fps = feats["n_frames"], a.fps

    start, end = 0.0, n / fps
    if a.preview:
        s, e = a.preview.split(":")
        start, end = float(s), min(float(e), n / fps)

    # 2. parallel render (each chunk warms its feedback buffer before writing)
    span = (end - start) / a.jobs
    chunks = []
    procs = []
    for j in range(a.jobs):
        cs, ce = start + j * span, (start + (j + 1) * span) if j < a.jobs - 1 else end
        out = work / f"chunk_{j:02d}.mp4"
        chunks.append(out)
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "musicviz.render", str(fpath), str(out), "--w", str(W), "--h", str(H),
             "--start", f"{cs:.4f}", "--end", f"{ce:.4f}", "--palette", a.palette],
            stdout=open(work / f"chunk_{j:02d}.log", "w"), stderr=subprocess.STDOUT))
    print(f"· rendering {W}x{H} @ {fps} fps in {a.jobs} parallel chunks …", flush=True)
    for j, p in enumerate(procs):
        if p.wait() != 0:
            sys.exit(f"render chunk {j} failed, see {work / f'chunk_{j:02d}.log'}")
    (work / "chunks.txt").write_text("".join(f"file '{c.name}'\n" for c in chunks))
    raw = work / "raw.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(work / "chunks.txt"),
         "-c", "copy", str(raw)])

    # 3. titles, fades, audio
    print("· compositing titles + audio", flush=True)
    dur = end - start
    inputs = ["-i", str(raw), "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", a.audio]
    chain, last = ["[0:v]fade=t=in:st=0:d=2[v0]"], "v0"
    if not a.no_titles and (a.title or a.artist):
        titles.make(W, H, str(work / "t"), a.title, a.artist, a.year, a.tagline, a.url)
        inputs += ["-loop", "1", "-framerate", str(fps), "-i", str(work / "t_intro.png"),
                   "-loop", "1", "-framerate", str(fps), "-i", str(work / "t_outro.png")]
        out_st = max(0.0, dur - 11)
        chain += [
            "[2:v]format=rgba,fade=t=in:st=1.2:d=1.8:alpha=1,fade=t=out:st=8:d=2.2:alpha=1,"
            "trim=0:10.5,setpts=PTS-STARTPTS[ti]",
            f"[3:v]format=rgba,trim=0:12,setpts=PTS-STARTPTS,fade=t=in:st=0:d=2:alpha=1,"
            f"fade=t=out:st=9.8:d=2:alpha=1,setpts=PTS+{out_st:.3f}/TB[to]",
            "[v0][ti]overlay=eof_action=pass[v1]", "[v1][to]overlay=eof_action=pass[v2]"]
        last = "v2"
    chain.append(f"[{last}]fade=t=out:st={max(0, dur - 2.5):.3f}:d=2.4,format=yuv420p[v]")
    rate = [] if a.maxrate == "none" else ["-maxrate", a.maxrate, "-bufsize", str(int(a.maxrate.rstrip("k")) * 2) + "k"]
    meta = []
    for k, v in (("title", a.title), ("artist", a.artist), ("composer", a.artist), ("date", a.year)):
        if v: meta += ["-metadata", f"{k}={v}"]
    run(["ffmpeg", "-y", "-v", "error", "-stats", *inputs, "-filter_complex", ";".join(chain),
         "-map", "[v]", "-map", "1:a", "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), *rate, "-profile:v", "high",
         "-c:a", "aac", "-b:a", "192k", "-ac", "2", *meta, "-movflags", "+faststart", a.out])

    if not a.work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"✓ {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
