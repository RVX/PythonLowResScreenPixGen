"""
MONOLITH CEILING – Generative Video Suite
Art Basel 2026 / Julian Charrière

Fixture grid:
  9 rows (Y/depth)  ×  24 bars (X/horizontal)  ×  20 LEDs per bar
  → canvas: 480 px wide  ×  9 px tall
  Each pixel = one physical LED on GLP Impression X4 Bar 20 (CW+WW)
  B&W intensity (0–255) maps directly to DMX via grandMA3 Bitmap

All videos saved to  Monolith_video_folder/  sequentially named:

  ENERGETIC / STRUCTURED
  01_scan_bounce_gol.webm       row scan → bounce → GoL explosion
  02_rain.webm                  per-column drops falling Y (top→bottom)
  03_wave_ripple.webm           3 interfering sine waves along X
  04_lightning.webm             stochastic horizontal discharge bolts
  05_meteor_shower.webm         diagonal streaks with comet tails
  06_breathing_noise.webm       12-layer harmonic noise with breath envelope
  07_cellular_automata.webm     Rule-30 CA, one per row, fast
  08_rain_horizontal.webm       comet streaks along X axis (depth Y fixed)

  DELICATE / ORGANIC / SLOW
  09_slow_aurora.webm           soft drifting bands, Gaussian Y falloff
  10_deep_pulse.webm            front-to-back breathing pulse, 12s period
  11_fireflies.webm             sparse single-pixel sparks, black base
  12_slow_cellular_drift.webm   Rule-110 CA at 1 step/8 frames, long decay
  13_tide.webm                  slow horizontal luminous wash, ~35% peak
  14_stellar_parallax.webm      3-layer drifting stars with parallax depth
  15_heat_diffusion.webm        heat-equation blooms spreading from seeds
  16_column_choir.webm          close harmonics per column, shimmering moiré
  17_depth_ripple.webm          expanding rings from random origin points

Requirements:  pip install numpy opencv-python
"""

import os
import math
import subprocess
import numpy as np
import cv2

# ── Grid ──────────────────────────────────────────────────────────────────────
GW  = 480    # 24 bars × 20 LEDs
GH  = 9      # 9 rows

# ── Render config ─────────────────────────────────────────────────────────────
# grandMA3 Bitmap maps the source video to the canvas pixel-for-pixel when
# Width/Height in Bitmap Configuration match the fixture grid.
# We upscale only for visual preview; the Bitmap reads SCALE_W × SCALE_H internally.
# Set PREVIEW_SCALE = 1 if you want the raw 480×9 file (smallest, most direct).
PREVIEW_SCALE = 4    # 1920 × 36 – grandMA3 max width, renders ~9× faster than ×12
VW = GW * PREVIEW_SCALE
VH = GH * PREVIEW_SCALE

FPS  = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), "Monolith_video_folder")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(2026)


# ── FFmpeg-based video writer ─────────────────────────────────────────────────
class FfmpegWriter:
    """
    Drop-in replacement for cv2.VideoWriter that pipes raw grayscale frames
    to a standalone ffmpeg process.

    WHY: OpenCV's VP8 encoder converts BGR→YUV internally.  The lossy YUV
    4:2:0 quantisation introduces ±1 Cb/Cr rounding errors that show up as
    a greenish tint in near-black fade tails.  By sending a single-channel
    (gray) stream to ffmpeg we guarantee Cb=Cr=128 for every pixel — no
    chroma channel, no chroma artifact.
    """
    def __init__(self, path: str, fps: int, width: int, height: int):
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "gray",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libvpx",
            "-crf", "10", "-b:v", "0",
            "-auto-alt-ref", "0",
            "-pix_fmt", "yuv420p",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    def write(self, gray: np.ndarray) -> None:
        """Accept a single-channel uint8 (H×W) frame."""
        self._proc.stdin.write(gray.tobytes())

    def release(self) -> None:
        self._proc.stdin.close()
        _, stderr = self._proc.communicate()
        if self._proc.returncode != 0 and stderr:
            print(f"  [ffmpeg] {stderr.decode(errors='replace').strip()}")

    def isOpened(self) -> bool:
        return self._proc.poll() is None


# ── Helpers ───────────────────────────────────────────────────────────────────

def ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3

def ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

def open_writer(filename: str) -> FfmpegWriter:
    path = os.path.join(OUT_DIR, filename)
    return FfmpegWriter(path, FPS, VW, VH)

def write_frame(writer: FfmpegWriter, canvas: np.ndarray) -> None:
    """
    canvas: float32 (GH × GW), values 0–255
    Upscale with nearest-neighbour (no blending between pixels —
    each physical LED is one discrete unit) then write as raw gray to ffmpeg.
    """
    gray = np.clip(canvas, 0, 255).astype(np.uint8)
    big  = cv2.resize(gray, (VW, VH), interpolation=cv2.INTER_NEAREST)
    writer.write(big)

def close(writer: cv2.VideoWriter, path_hint: str = "") -> None:
    writer.release()
    print(f"  saved -> {path_hint}")


# ══════════════════════════════════════════════════════════════════════════════
# 01 – SCAN / BOUNCE / GAME-OF-LIFE  (adapted to 480×9 grid)
# ══════════════════════════════════════════════════════════════════════════════

def make_01_scan_gol() -> None:
    name = "01_scan_bounce_gol.webm"
    print(f"[01] {name} …")
    w    = open_writer(name)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    # Phase A – vertical scan across rows (top→bottom) with horizontal cursor
    SCAN_N = 90
    for i in range(SCAN_N):
        buf *= 0.88
        t   = i / (SCAN_N - 1)
        y   = int(t * (GH - 1))
        x   = int(t * (GW - 1))
        buf[y, :] = np.maximum(buf[y, :], 220)          # full row lights up
        buf[y, x] = 255                                  # leading cursor
        write_frame(w, buf)

    # Phase B – bounce row top↔bottom (4 half-sweeps)
    BOUNCE_N = 45
    cur_y    = GH - 1
    for b in range(4):
        end_y = 0 if b % 2 == 0 else GH - 1
        start = cur_y
        for i in range(BOUNCE_N):
            buf  *= 0.85
            t     = ease_in_out(i / (BOUNCE_N - 1))
            cur_y = int(start + (end_y - start) * t)
            buf[cur_y, :] = np.maximum(buf[cur_y, :], 200)
            write_frame(w, buf)

    # Phase C – converge to middle row, scan inward from both edges
    CONV_N  = 60
    start   = cur_y
    mid_y   = GH // 2
    for i in range(CONV_N):
        buf *= 0.85
        t    = ease_out(i / (CONV_N - 1))
        y    = int(start + (mid_y - start) * t)
        # Converge from both X ends toward centre
        edge = int((1.0 - t) * (GW // 2))
        buf[y, :GW//2 - edge] = np.maximum(buf[y, :GW//2 - edge], 180)
        buf[y, GW//2 + edge:] = np.maximum(buf[y, GW//2 + edge:], 180)
        write_frame(w, buf)

    # Phase D – explosion flash
    for i in range(20):
        v = 255 * ((1 - i / 19) ** 2)
        write_frame(w, np.full((GH, GW), v, dtype=np.float32))

    # Phase E – Game of Life on 480×9 grid
    # Seed: full row of live cells in the centre row + random noise
    gol = np.zeros((GH, GW), dtype=bool)
    gol[mid_y, :] = True                                  # horizontal line seed
    gol |= RNG.random((GH, GW)) < 0.15                    # random sparks

    def gol_step(g: np.ndarray) -> np.ndarray:
        gi = g.astype(np.uint8)
        n  = sum(np.roll(np.roll(gi, dy, 0), dx, 1)
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                 if dy != 0 or dx != 0)
        return (n == 3) | (g & (n == 2))

    GOL_N = 600
    trail = np.zeros((GH, GW), dtype=np.float32)
    for i in range(GOL_N):
        trail *= 0.96
        np.maximum(trail, gol.astype(np.float32) * 255, out=trail)
        write_frame(w, trail)
        gol = gol_step(gol)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 02 – RAIN
# Bright drops fall from top row down across 9 rows, leaving a fading trail.
# Each column is an independent raindrop with random speed and brightness.
# ══════════════════════════════════════════════════════════════════════════════

def make_02_rain() -> None:
    name = "02_rain.webm"
    print(f"[02] {name} …")
    w    = open_writer(name)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    # Each column: fractional Y position, speed, brightness, active flag
    col_y     = RNG.uniform(-GH, 0,   GW).astype(np.float32)
    col_speed = RNG.uniform(0.08, 0.4, GW).astype(np.float32)
    col_bright= RNG.uniform(160, 255,  GW).astype(np.float32)

    TOTAL = FPS * 40   # 40 s

    for _ in range(TOTAL):
        buf *= 0.75                                       # trail decay
        col_y += col_speed                               # advance drops

        # reset columns that fell off bottom
        done = col_y >= GH
        col_y[done]     = RNG.uniform(-GH * 0.5, 0, done.sum())
        col_speed[done] = RNG.uniform(0.08, 0.4,   done.sum())
        col_bright[done]= RNG.uniform(160, 255,     done.sum())

        # paint active pixels
        iy = col_y.astype(int)
        mask = (iy >= 0) & (iy < GH)
        for x in np.where(mask)[0]:
            buf[iy[x], x] = max(buf[iy[x], x], col_bright[x])

        write_frame(w, buf)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 03 – WAVE RIPPLE
# A sinusoidal wave travels horizontally across the 480-pixel span.
# Multiple overlapping waves create organic interference patterns.
# ══════════════════════════════════════════════════════════════════════════════

def make_03_wave_ripple() -> None:
    name = "03_wave_ripple.webm"
    print(f"[03] {name} …")
    w    = open_writer(name)

    TOTAL = FPS * 45
    xs    = np.arange(GW, dtype=np.float32)
    ys    = np.arange(GH, dtype=np.float32)

    # 3 travelling waves with different freq / speed / amplitude
    waves = [
        {"freq": 2 * math.pi / 60,  "speed": 0.04, "amp": 1.0,  "phase": 0.0},
        {"freq": 2 * math.pi / 120, "speed": 0.025,"amp": 0.6,  "phase": 1.2},
        {"freq": 2 * math.pi / 40,  "speed": 0.06, "amp": 0.4,  "phase": 2.5},
    ]
    # Y modulation: each wave has a vertical envelope (sine over 9 rows)
    y_env = [
        np.sin(np.pi * ys / (GH - 1)),
        np.abs(np.sin(2 * np.pi * ys / GH)),
        np.ones(GH),
    ]

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for wi, wave in enumerate(waves):
            wave["phase"] += wave["speed"]
            h_wave = np.sin(xs * wave["freq"] - wave["phase"])  # (GW,)
            # Outer product with vertical envelope → (GH, GW)
            combined = np.outer(y_env[wi], h_wave) * wave["amp"]
            canvas  += combined

        # Normalise to 0–255, add slight noise for organic feel
        canvas  = (canvas + 3.0) / 6.0          # roughly 0–1
        canvas += RNG.random((GH, GW)).astype(np.float32) * 0.04
        canvas  = np.clip(canvas * 255, 0, 255)
        write_frame(w, canvas)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 04 – LIGHTNING
# Sudden bright horizontal discharges propagate across the ceiling,
# leaving phosphor afterglow that slowly decays.
# ══════════════════════════════════════════════════════════════════════════════

def make_04_lightning() -> None:
    name = "04_lightning.webm"
    print(f"[04] {name} …")
    w    = open_writer(name)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    TOTAL      = FPS * 40
    next_bolt  = 0

    for f in range(TOTAL):
        buf *= 0.90

        if f >= next_bolt:
            # Pick a random row and a random starting X
            row   = RNG.integers(0, GH)
            start = RNG.integers(0, GW)
            # Lightning spreads left and right from origin as a bright impulse
            length = RNG.integers(GW // 6, GW // 2)
            end_L  = max(0,    start - length)
            end_R  = min(GW-1, start + length)

            # Multi-frame propagation over ~8 frames
            for spread in range(1, 12):
                l  = max(0,    start - int(spread * length / 11))
                r  = min(GW-1, start + int(spread * length / 11))
                buf[row, l:r+1] = 255
                # Feather adjacent rows
                if row > 0:
                    buf[row-1, l:r+1] = np.maximum(buf[row-1, l:r+1], 160)
                if row < GH-1:
                    buf[row+1, l:r+1] = np.maximum(buf[row+1, l:r+1], 160)
                write_frame(w, buf)
                buf[row, l:r+1] *= 0.70

            # Quiet gap: 1–4 s between bolts
            next_bolt = f + RNG.integers(FPS * 1, FPS * 4)
        else:
            write_frame(w, buf)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 05 – METEOR SHOWER
# Bright streaks travel diagonally across the ceiling grid.
# On a 480×9 grid, "diagonal" = small Y offset over many X steps.
# ══════════════════════════════════════════════════════════════════════════════

def make_05_meteor_shower() -> None:
    name = "05_meteor_shower.webm"
    print(f"[05] {name} …")
    w    = open_writer(name)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    # Active meteors: [x (float), y (float), vx, vy, length, brightness]
    MAX_METEORS = 12
    meteors     = []

    def spawn():
        return {
            "x":  float(RNG.integers(-GW//4, GW)),
            "y":  float(RNG.integers(0, GH)),
            "vx": RNG.uniform(3.0, 8.0),           # pixels/frame horizontally
            "vy": RNG.uniform(-0.15, 0.15),         # very slow vertical drift
            "tail": RNG.integers(8, 30),            # tail length in pixels
            "bright": RNG.uniform(180, 255),
        }

    for _ in range(MAX_METEORS):
        meteors.append(spawn())

    TOTAL = FPS * 40

    for _ in range(TOTAL):
        buf *= 0.82

        for m in meteors:
            m["x"] += m["vx"]
            m["y"] += m["vy"]
            m["y"]  = float(np.clip(m["y"], 0, GH - 1))

            # Draw tail (fade with distance)
            for t in range(m["tail"]):
                tx = int(m["x"]) - t
                ty = int(round(m["y"] - t * m["vy"] / max(m["vx"], 0.01)))
                if 0 <= tx < GW and 0 <= ty < GH:
                    fade       = (1.0 - t / m["tail"]) ** 2
                    buf[ty, tx] = max(buf[ty, tx], m["bright"] * fade)

            # Respawn when off-screen
            if m["x"] > GW + m["tail"]:
                m.update(spawn())

        write_frame(w, buf)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 06 – BREATHING NOISE
# Slow, organic perlin-like noise field that breathes in and out.
# Uses layered sine functions to approximate smooth spatial noise.
# Very quiet, meditative — good as a base state.
# ══════════════════════════════════════════════════════════════════════════════

def make_06_breathing_noise() -> None:
    name = "06_breathing_noise.webm"
    print(f"[06] {name} …")
    w    = open_writer(name)

    TOTAL  = FPS * 50
    xs     = np.linspace(0, 2 * math.pi, GW, dtype=np.float32)
    ys     = np.linspace(0, 2 * math.pi, GH, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    # Random phase offsets and frequencies for ~12 harmonic layers
    N_LAYERS = 12
    fx  = RNG.uniform(0.5, 4.0, N_LAYERS).astype(np.float32)
    fy  = RNG.uniform(0.5, 3.0, N_LAYERS).astype(np.float32)
    fpx = RNG.uniform(0, 2*math.pi, N_LAYERS).astype(np.float32)  # x phase speed
    fpy = RNG.uniform(0, 2*math.pi, N_LAYERS).astype(np.float32)  # y phase speed
    amp = RNG.uniform(0.5, 1.0, N_LAYERS).astype(np.float32)
    amp /= amp.sum()  # normalise

    phase_x = np.zeros(N_LAYERS, dtype=np.float32)
    phase_y = np.zeros(N_LAYERS, dtype=np.float32)

    # Global breath envelope (slow 0→1→0 cycles)
    breath_period = FPS * 8   # 8 s per breath

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)

        for li in range(N_LAYERS):
            layer   = np.sin(XX * fx[li] + phase_x[li]) * \
                      np.sin(YY * fy[li] + phase_y[li])
            canvas += layer * amp[li]
            phase_x[li] += fpx[li] * 0.003
            phase_y[li] += fpy[li] * 0.003

        # Map -1…1 → 0…1
        canvas = (canvas + 1.0) * 0.5

        # Breathing envelope
        breath = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(2 * math.pi * f / breath_period))
        canvas *= breath

        canvas = np.clip(canvas * 255, 0, 255)
        write_frame(w, canvas)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 07 – CELLULAR AUTOMATA (Rule 30 / Rule 110 hybrid)
# 1-D elementary cellular automaton run row-by-row across time.
# Rule 30 is famous for producing unpredictable, natural-looking patterns.
# On the ceiling: each column evolves its own 1-row CA state over time.
# ══════════════════════════════════════════════════════════════════════════════

def make_07_cellular_automata() -> None:
    name = "07_cellular_automata.webm"
    print(f"[07] {name} …")
    w    = open_writer(name)

    # Run 9 independent 1-D CAs (one per row), each with Rule 30
    RULE = 30
    rule_map = np.array([(RULE >> i) & 1 for i in range(8)], dtype=bool)

    # Random initial state (sparse)
    states = (RNG.random((GH, GW)) < 0.08)

    def ca_step(row: np.ndarray) -> np.ndarray:
        l = np.roll(row, -1)
        c = row
        r = np.roll(row,  1)
        idx = (l.astype(int) * 4 + c.astype(int) * 2 + r.astype(int))
        return rule_map[idx]

    TOTAL = FPS * 50
    trail = np.zeros((GH, GW), dtype=np.float32)

    for _ in range(TOTAL):
        trail *= 0.94
        for row in range(GH):
            states[row] = ca_step(states[row])
        np.maximum(trail, states.astype(np.float32) * 255, out=trail)
        write_frame(w, trail)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 08 – HORIZONTAL RAIN
# Streaks travel LEFT → RIGHT along the 480-px X axis (depth Y is fixed per streak).
# Each streak is a single row that lights up as a fast horizontal pulse,
# leaving a decaying comet tail behind it.
# Dark base — only a few active streaks at a time, sparse and directional.
# ══════════════════════════════════════════════════════════════════════════════

def make_08_rain_horizontal() -> None:
    name = "08_rain_horizontal.webm"
    print(f"[08] {name} …")
    w    = open_writer(name)
    buf  = np.zeros((GH, GW), dtype=np.float32)
    rng2 = np.random.default_rng(808)

    MAX_STREAKS = 5   # sparse — rarely more than 3–4 visible at once

    def new_streak():
        return {
            "x":     float(rng2.integers(-40, 0)),        # starts left of frame
            "y":     int(rng2.integers(0, GH)),            # fixed row
            "vx":    rng2.uniform(1.5, 5.0),               # slow–medium speed px/frame
            "tail":  rng2.integers(12, 50),                # tail length in pixels
            "bright": rng2.uniform(120, 220),              # NOT always full white
        }

    streaks = [new_streak() for _ in range(MAX_STREAKS)]
    # Stagger initial positions so they don't all start at once
    for i, s in enumerate(streaks):
        s["x"] = float(rng2.integers(-GW, 0))

    TOTAL = FPS * 55   # 55 s

    for _ in range(TOTAL):
        buf *= 0.82      # slower decay than vertical rain → longer glowing trails

        for s in streaks:
            s["x"] += s["vx"]
            hx = int(s["x"])
            hy = s["y"]
            # Draw tail (brighter at head, fades to black)
            for t in range(s["tail"]):
                tx = hx - t
                if 0 <= tx < GW:
                    fade         = ((s["tail"] - t) / s["tail"]) ** 2
                    buf[hy, tx]  = max(buf[hy, tx], s["bright"] * fade)
                    # Very faint bleed into adjacent depth rows
                    if hy > 0:
                        buf[hy-1, tx] = max(buf[hy-1, tx], s["bright"] * fade * 0.15)
                    if hy < GH-1:
                        buf[hy+1, tx] = max(buf[hy+1, tx], s["bright"] * fade * 0.15)

            # Respawn when fully off right edge
            if s["x"] > GW + s["tail"]:
                s.update(new_streak())

        write_frame(w, buf)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 09 – SLOW AURORA
# Soft, wide luminous bands drift slowly along X.
# Very dark base (~5 % ambient), peaks reach ~60–80 % brightness only at crests.
# The Y axis (depth) gets a gentle fade so front/back rows are dimmer.
# ══════════════════════════════════════════════════════════════════════════════

def make_09_slow_aurora() -> None:
    name = "09_slow_aurora.webm"
    print(f"[09] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(909)

    TOTAL = FPS * 60   # 60 s

    xs = np.linspace(0, 2 * math.pi * 3, GW, dtype=np.float32)  # 3 full cycles across 480px

    # 4 slow bands with independent phase speeds and Y concentrations
    bands = [
        {"phase": rng2.uniform(0, 2*math.pi),
         "speed": rng2.uniform(0.004, 0.010),
         "freq":  rng2.uniform(0.8, 1.5),
         "y_ctr": rng2.uniform(1.5, GH - 1.5),   # vertical centre of band
         "y_sig": rng2.uniform(1.0, 3.0),         # vertical spread (std dev in rows)
         "peak":  rng2.uniform(140, 210),          # max brightness
        }
        for _ in range(4)
    ]

    ys = np.arange(GH, dtype=np.float32)

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)

        for b in bands:
            b["phase"] += b["speed"]
            # Horizontal wave shape (0–1 range)
            h_shape = 0.5 + 0.5 * np.sin(xs * b["freq"] - b["phase"])  # (GW,)
            # Vertical Gaussian envelope centred on b["y_ctr"]
            v_env   = np.exp(-0.5 * ((ys - b["y_ctr"]) / b["y_sig"]) ** 2)  # (GH,)
            # Outer product → (GH, GW), scaled to peak brightness
            canvas += np.outer(v_env, h_shape) * b["peak"]

            # Slowly drift the band's vertical centre
            b["y_ctr"] += rng2.uniform(-0.003, 0.003)
            b["y_ctr"]  = float(np.clip(b["y_ctr"], 0.5, GH - 1.5))

        # Hard ceiling so we don't clip to white; minimum stays near black
        canvas = np.clip(canvas, 0, 200)
        write_frame(w, canvas)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 10 – DEEP PULSE
# The whole ceiling inhales and exhales in slow, long pulses.
# Each depth row (Y) has a slightly different phase offset so the pulse
# appears to travel from front to back (or back to front) across the room.
# Peak brightness ~40 %, black floor, very long period (~12 s per breath).
# ══════════════════════════════════════════════════════════════════════════════

def make_10_deep_pulse() -> None:
    name = "10_deep_pulse.webm"
    print(f"[10] {name} …")
    w    = open_writer(name)

    TOTAL      = FPS * 70   # 70 s
    PERIOD     = FPS * 12   # 12 s per breath cycle
    PEAK       = 100.0      # max brightness (< 50 %)
    # Phase offset per row: wave travels front→back over ~2 s
    ROW_OFFSET = (FPS * 2) / GH

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for y in range(GH):
            phase    = 2 * math.pi * (f - y * ROW_OFFSET) / PERIOD
            envelope = (0.5 + 0.5 * math.sin(phase)) ** 2.5  # slow ease, mostly dark
            # Slight X variation: cosine ripple with very low amplitude
            x_ripple = 1.0 + 0.06 * np.cos(
                np.linspace(0, 2 * math.pi * 3, GW, dtype=np.float32) + f * 0.008
            )
            canvas[y, :] = envelope * PEAK * x_ripple
        write_frame(w, canvas)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 11 – FIREFLIES
# Sparse single-pixel sparks appear randomly, glow to a peak, then fade.
# On a 480×9 grid this reads as rare, isolated points of light against black.
# Organic, quiet — good for long stretches between other content.
# ══════════════════════════════════════════════════════════════════════════════

def make_11_fireflies() -> None:
    name = "11_fireflies.webm"
    print(f"[11] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1111)

    TOTAL      = FPS * 60
    MAX_FIRES  = 18     # max simultaneous fireflies across 480×9 = 4320 pixels
    SPAWN_RATE = 0.03   # probability per frame of spawning a new one

    # Each firefly: x, y, life (0→peak→0), peak_bright, life_span
    fires: list[dict] = []

    def spawn_fire():
        life_span = rng2.integers(FPS * 2, FPS * 8)   # 2–8 s lifetime
        return {
            "x":     int(rng2.integers(0, GW)),
            "y":     int(rng2.integers(0, GH)),
            "t":     0,
            "span":  life_span,
            "peak":  rng2.uniform(60, 200),            # rarely reach full white
        }

    buf = np.zeros((GH, GW), dtype=np.float32)

    for f in range(TOTAL):
        buf *= 0.96   # very slow ambient decay

        # Spawn
        if len(fires) < MAX_FIRES and rng2.random() < SPAWN_RATE:
            fires.append(spawn_fire())

        # Update and draw
        alive = []
        for fi in fires:
            fi["t"] += 1
            t_norm = fi["t"] / fi["span"]
            # Triangle envelope: ramp up first half, fade second half
            env    = 1.0 - abs(2 * t_norm - 1.0)
            bright = env * fi["peak"]
            buf[fi["y"], fi["x"]] = max(buf[fi["y"], fi["x"]], bright)
            if fi["t"] < fi["span"]:
                alive.append(fi)
        fires = alive

        write_frame(w, buf)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 12 – SLOW CELLULAR DRIFT
# Rule-110 automaton at very low update rate (1 step every 8 frames),
# with a dim ambient noise floor and extremely slow trail decay.
# Mostly dark — alive cells glow briefly then fade, creating drifting
# constellations that feel geological and patient.
# ══════════════════════════════════════════════════════════════════════════════

def make_12_slow_cellular_drift() -> None:
    name = "12_slow_cellular_drift.webm"
    print(f"[12] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1212)

    # Rule 110
    RULE     = 110
    rule_map = np.array([(RULE >> i) & 1 for i in range(8)], dtype=bool)

    def ca_step(row: np.ndarray) -> np.ndarray:
        l   = np.roll(row, -1)
        c   = row
        r   = np.roll(row,  1)
        idx = l.astype(int) * 4 + c.astype(int) * 2 + r.astype(int)
        return rule_map[idx]

    # Very sparse seed — only ~4 % of cells alive
    states = rng2.random((GH, GW)) < 0.04
    trail  = np.zeros((GH, GW), dtype=np.float32)

    TOTAL      = FPS * 70
    STEP_EVERY = 8    # advance CA only every N frames

    for f in range(TOTAL):
        trail *= 0.985   # very slow phosphor decay
        if f % STEP_EVERY == 0:
            for row in range(GH):
                states[row] = ca_step(states[row])
            # New live cells ignite at moderate brightness
            np.maximum(trail, states.astype(np.float32) * 160, out=trail)
        write_frame(w, trail)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 13 – TIDE
# A luminous wash moves slowly left ↔ right like a tide across the ceiling.
# Asymmetric: sharper leading edge, long dim trailing haze.
# Peak ~35 % brightness. Y is brightest in the middle rows.
# ══════════════════════════════════════════════════════════════════════════════

def make_13_tide() -> None:
    name = "13_tide.webm"
    print(f"[13] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1313)

    TOTAL  = FPS * 75
    PERIOD = FPS * 20   # 20 s per full sweep cycle
    xs     = np.arange(GW, dtype=np.float32)
    ys_arr = np.arange(GH, dtype=np.float32)
    v_env  = np.exp(-0.5 * ((ys_arr - (GH - 1) / 2) / 2.2) ** 2)   # (GH,)

    for f in range(TOTAL):
        t_phase  = math.sin(2 * math.pi * f / PERIOD)
        move_dir = math.cos(2 * math.pi * f / PERIOD)
        centre_x = (GW / 2) + t_phase * (GW * 0.45)
        d        = (xs - centre_x) * (1 if move_dir >= 0 else -1)
        crest    = np.exp(-0.5 * (d / 18.0) ** 2) * 90.0
        trail    = np.exp(-np.maximum(d, 0) / 60.0) * 30.0
        h_shape  = np.maximum(crest, trail)
        canvas   = np.outer(v_env, h_shape)
        canvas  += rng2.random((GH, GW)).astype(np.float32) * 3.0
        write_frame(w, np.clip(canvas, 0, 100))

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 14 – STELLAR PARALLAX
# Three depth layers of sparse stars drift at different speeds (parallax).
# Front rows drift fast, back rows drift slow — illusion of depth.
# Occasional twinkle. Base 12–40/255. Mostly black.
# ══════════════════════════════════════════════════════════════════════════════

def make_14_stellar_parallax() -> None:
    name = "14_stellar_parallax.webm"
    print(f"[14] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1414)

    TOTAL  = FPS * 70
    layers = [
        {"rows": [0, 1],          "speed": 0.06,  "count": 8,  "base": 40},
        {"rows": [2, 3, 4, 5, 6], "speed": 0.022, "count": 12, "base": 22},
        {"rows": [7, 8],          "speed": 0.007, "count": 5,  "base": 12},
    ]
    for layer in layers:
        layer["stars"] = [
            {"x": rng2.uniform(0, GW), "y": int(rng2.choice(layer["rows"])),
             "bright": rng2.uniform(layer["base"] * 0.7, layer["base"] * 1.5),
             "twinkle": 0}
            for _ in range(layer["count"])
        ]

    buf = np.zeros((GH, GW), dtype=np.float32)

    for _ in range(TOTAL):
        buf *= 0.93
        for layer in layers:
            for s in layer["stars"]:
                s["x"] = (s["x"] + layer["speed"]) % GW
                if s["twinkle"] > 0:
                    s["twinkle"] -= 1
                    bright = s["bright"] * (1.0 + 1.5 * s["twinkle"] / 15)
                elif rng2.random() < 0.0015:
                    s["twinkle"] = 15
                    bright = s["bright"] * 3.0
                else:
                    bright = s["bright"]
                x = int(s["x"]) % GW
                buf[s["y"], x] = max(buf[s["y"], x], min(bright, 200))
        write_frame(w, buf)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 15 – HEAT DIFFUSION
# The 480×9 grid acts as a heat plate. Periodic seeds inject warmth;
# heat spreads via the 2D diffusion equation and slowly cools.
# Multiple simultaneous blooms create organic cloud-like pools of light.
# ══════════════════════════════════════════════════════════════════════════════

def make_15_heat_diffusion() -> None:
    name = "15_heat_diffusion.webm"
    print(f"[15] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1515)

    TOTAL      = FPS * 65
    INJECT_INT = FPS * 5     # new heat sources every ~5 s
    K          = 0.16        # diffusion rate (stable ≤ 0.25)
    DECAY      = 0.993       # cooling per frame

    field = np.zeros((GH, GW), dtype=np.float32)

    for f in range(TOTAL):
        lap   = (np.roll(field, -1, 0) + np.roll(field, 1, 0) +
                 np.roll(field, -1, 1) + np.roll(field, 1, 1) - 4 * field)
        field = np.clip(field + K * lap, 0, 255) * DECAY
        if f % INJECT_INT == 0:
            for _ in range(rng2.integers(1, 3)):
                hx = rng2.integers(0, GW)
                hy = rng2.integers(0, GH)
                field[hy, hx] = max(field[hy, hx], rng2.uniform(130, 210))
        write_frame(w, field)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 16 – COLUMN CHOIR
# Each of the 480 columns oscillates at its own frequency — all close to
# a 10-second fundamental with tiny deviations. Slow beating creates a
# shimmering moiré that drifts without repeating. Each column has its own
# Y Gaussian envelope. Brightness cap ~25 %. Quiet and textile-like.
# ══════════════════════════════════════════════════════════════════════════════

def make_16_column_choir() -> None:
    name = "16_column_choir.webm"
    print(f"[16] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1616)

    TOTAL     = FPS * 80
    base_freq = 2 * math.pi / (FPS * 10)
    freqs     = (base_freq + rng2.uniform(-base_freq * 0.3,
                                          base_freq * 0.3, GW)).astype(np.float32)
    phases    = rng2.uniform(0, 2 * math.pi, GW).astype(np.float32)
    y_centres = rng2.uniform(0, GH - 1, GW).astype(np.float32)
    y_spreads = rng2.uniform(0.6, 2.8,  GW).astype(np.float32)
    ys        = np.arange(GH, dtype=np.float32)
    PEAK      = 62.0

    for f in range(TOTAL):
        col_amp = PEAK * (0.5 + 0.5 * np.sin(freqs * f + phases))
        v_env   = np.exp(-0.5 * ((ys[:, None] - y_centres[None, :]) /
                                  y_spreads[None, :]) ** 2)
        write_frame(w, v_env * col_amp[None, :])

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# 17 – DEPTH RIPPLE
# Circular ripples expand from random (X, Y) origins. Because Y spans
# only 9 rows (but 7.88 m each), the ring is physically scaled correctly —
# a ripple takes much longer to cross Y than X. Dark base. Peak ~50 %.
# ══════════════════════════════════════════════════════════════════════════════

def make_17_depth_ripple() -> None:
    name = "17_depth_ripple.webm"
    print(f"[17] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1717)

    TOTAL      = FPS * 65
    SPAWN_INT  = FPS * 6
    MAX_RIPPLES = 5
    xs      = np.arange(GW, dtype=np.float32)
    ys      = np.arange(GH, dtype=np.float32)
    Y_SCALE = 7.88   # each Y step ≈ 7.88 m; X step ≈ 1.0 m

    ripples: list[dict] = []

    def new_ripple():
        return {"ox": rng2.uniform(0, GW), "oy": rng2.uniform(0, GH - 1),
                "r": 0.0, "speed": rng2.uniform(0.4, 1.0),
                "peak": rng2.uniform(50, 120), "width": rng2.uniform(1.5, 3.5)}

    for f in range(TOTAL):
        if f % SPAWN_INT == 0 and len(ripples) < MAX_RIPPLES:
            ripples.append(new_ripple())

        canvas = np.zeros((GH, GW), dtype=np.float32)
        alive  = []
        max_r  = math.hypot(GW, GH * Y_SCALE) + 12

        for rp in ripples:
            rp["r"] += rp["speed"]
            dx   = xs - rp["ox"]
            dy   = (ys - rp["oy"]) * Y_SCALE
            dist = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
            np.maximum(canvas,
                       np.exp(-0.5 * ((dist - rp["r"]) / rp["width"]) ** 2) * rp["peak"],
                       out=canvas)
            if rp["r"] < max_r:
                alive.append(rp)
        ripples = alive

        write_frame(w, canvas)

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-PIXEL MOVEMENT SERIES  (18 – 32)
# Each scene treats every LED as a discrete point of light.
# No continuous fields — only individual pixel events.
# ══════════════════════════════════════════════════════════════════════════════

# 18 – WANDERER
# One bright pixel takes a slow drunk-walk across the grid,
# pausing occasionally, leaving a long phosphor tail. Nothing else.

def make_18_wanderer() -> None:
    name = "18_wanderer.webm"
    print(f"[18] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1800)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    px, py  = GW // 2, GH // 2
    TOTAL   = FPS * 70
    PAUSE   = 0   # remaining pause frames

    for _ in range(TOTAL):
        buf *= 0.97
        buf[py, px] = 220

        if PAUSE > 0:
            PAUSE -= 1
        elif rng2.random() < 0.04:
            PAUSE = rng2.integers(FPS, FPS * 4)
        else:
            step = rng2.integers(0, 4)
            if step == 0: px = (px + 1) % GW
            elif step == 1: px = (px - 1) % GW
            elif step == 2: py = min(py + 1, GH - 1)
            else:           py = max(py - 1, 0)

        write_frame(w, buf)
    close(w, name)


# 19 – PIXEL MIGRATION
# ~20 pixels all drift slowly in the same direction (rightward),
# each at a slightly different speed. When one leaves the right edge
# it reappears at a random left position. Faint trails.

def make_19_pixel_migration() -> None:
    name = "19_pixel_migration.webm"
    print(f"[19] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(1900)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N = 22
    xs = rng2.uniform(0, GW, N)
    ys = rng2.integers(0, GH, N).astype(float)
    vx = rng2.uniform(0.18, 0.55, N)
    vy = rng2.uniform(-0.008, 0.008, N)
    br = rng2.uniform(100, 200, N)
    TOTAL = FPS * 65

    for _ in range(TOTAL):
        buf *= 0.91
        xs  += vx
        ys  += vy
        ys   = np.clip(ys, 0, GH - 1)
        wrap = xs >= GW
        xs[wrap] = rng2.uniform(-10, 0, wrap.sum())
        ys[wrap] = rng2.integers(0, GH, wrap.sum()).astype(float)
        for i in range(N):
            buf[int(ys[i]), int(xs[i]) % GW] = max(
                buf[int(ys[i]), int(xs[i]) % GW], br[i])
        write_frame(w, buf)
    close(w, name)


# 20 – COLD SPARKS
# Single pixels flash on for exactly 1 frame then vanish.
# Rate: 1–3 sparks per frame on average. Everything else is black.
# Feels like interference / cosmic rays.

def make_20_cold_sparks() -> None:
    name = "20_cold_sparks.webm"
    print(f"[20] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2000)
    TOTAL = FPS * 55

    for _ in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        n = rng2.poisson(1.8)
        for _ in range(n):
            canvas[rng2.integers(0, GH), rng2.integers(0, GW)] = rng2.uniform(160, 255)
        write_frame(w, canvas)
    close(w, name)


# 21 – PIXEL TELEGRAPH
# Random pixels light up in short dot/dash patterns (1 or 3 frames on,
# gap of 2–8 frames). Multiple independent "stations" transmit
# simultaneously from fixed positions. Black background.

def make_21_pixel_telegraph() -> None:
    name = "21_pixel_telegraph.webm"
    print(f"[21] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2100)

    N_STATIONS = 30
    sx  = rng2.integers(0, GW, N_STATIONS)
    sy  = rng2.integers(0, GH, N_STATIONS)
    sbr = rng2.uniform(120, 220, N_STATIONS)

    # Each station has a state: "on" counter, "off" counter
    on_left  = np.zeros(N_STATIONS, dtype=int)
    off_left = rng2.integers(4, 20, N_STATIONS)

    TOTAL = FPS * 60

    for _ in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for i in range(N_STATIONS):
            if on_left[i] > 0:
                canvas[sy[i], sx[i]] = sbr[i]
                on_left[i] -= 1
                if on_left[i] == 0:
                    off_left[i] = rng2.integers(3, 18)
            elif off_left[i] > 0:
                off_left[i] -= 1
                if off_left[i] == 0:
                    on_left[i] = 1 if rng2.random() < 0.6 else 3  # dot or dash
        write_frame(w, canvas)
    close(w, name)


# 22 – RANDOM WALK SWARM
# 12 pixels each do an independent random walk.
# Each leaves a decaying trail proportional to its own brightness.
# Pixels occasionally "leap" to a random new position.

def make_22_random_walk_swarm() -> None:
    name = "22_random_walk_swarm.webm"
    print(f"[22] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2200)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N  = 12
    px = rng2.integers(0, GW, N)
    py = rng2.integers(0, GH, N)
    br = rng2.uniform(80, 200, N)
    TOTAL = FPS * 65

    DIRS = [( 1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    for _ in range(TOTAL):
        buf *= 0.94
        for i in range(N):
            buf[py[i], px[i]] = max(buf[py[i], px[i]], br[i])
            if rng2.random() < 0.005:           # rare teleport
                px[i] = rng2.integers(0, GW)
                py[i] = rng2.integers(0, GH)
            else:
                dx, dy = DIRS[rng2.integers(0, len(DIRS))]
                px[i] = int(np.clip(px[i] + dx, 0, GW - 1))
                py[i] = int(np.clip(py[i] + dy, 0, GH - 1))
        write_frame(w, buf)
    close(w, name)


# 23 – PIXEL HEARTBEAT
# 6 pixels pulse together with a cardiac double-beat rhythm (lub-dub).
# Between beats: complete darkness. Each pixel is at a fixed position
# in the grid spread evenly along X at different depth rows.

def make_23_pixel_heartbeat() -> None:
    name = "23_pixel_heartbeat.webm"
    print(f"[23] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2300)

    N   = 8
    pxs = [int(i * GW / N) + GW // (N * 2) for i in range(N)]
    pys = rng2.integers(0, GH, N)
    br  = rng2.uniform(150, 240, N)

    # Cardiac template at 30 fps: lub (4f on), gap (6f), dub (2f on), rest (38f)
    BEAT = ([1]*4 + [0]*6 + [1]*2 + [0]*38)
    beat_len = len(BEAT)
    TOTAL = FPS * 60

    buf = np.zeros((GH, GW), dtype=np.float32)

    for f in range(TOTAL):
        buf *= 0.88
        if BEAT[f % beat_len]:
            for i in range(N):
                buf[pys[i], pxs[i]] = br[i]
        write_frame(w, buf)
    close(w, name)


# 24 – PIXEL ORBITS
# 5 pixels each orbit a fixed centre point at different radii and speeds.
# Because GH=9 is tiny, orbits are elliptical (X radius >> Y radius).
# Phosphor trail. Black background.

def make_24_pixel_orbits() -> None:
    name = "24_pixel_orbits.webm"
    print(f"[24] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2400)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    cx, cy = GW / 2, (GH - 1) / 2
    N  = 5
    rx = rng2.uniform(30, 200, N)          # X radii in pixels
    ry = rng2.uniform(0.8, 3.5, N)         # Y radii in rows
    sp = rng2.uniform(0.008, 0.03, N)      # rad/frame
    ph = rng2.uniform(0, 2*math.pi, N)     # initial phase
    br = rng2.uniform(100, 210, N)
    TOTAL = FPS * 70

    for f in range(TOTAL):
        buf *= 0.95
        for i in range(N):
            angle = ph[i] + sp[i] * f
            x = int(np.clip(cx + rx[i] * math.cos(angle), 0, GW - 1))
            y = int(np.clip(cy + ry[i] * math.sin(angle), 0, GH - 1))
            buf[y, x] = max(buf[y, x], br[i])
        write_frame(w, buf)
    close(w, name)


# 25 – TRAILING CURSOR
# A single pixel moves slowly across the full 480-pixel span
# like a pointer scanning left→right, then jumps to a new row
# and scans again. Speed varies. Long glowing comet trail.

def make_25_trailing_cursor() -> None:
    name = "25_trailing_cursor.webm"
    print(f"[25] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2500)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    cx, cy  = 0.0, float(rng2.integers(0, GH))
    speed   = rng2.uniform(0.8, 2.5)
    TOTAL   = FPS * 70

    for _ in range(TOTAL):
        buf *= 0.96
        buf[int(cy), int(cx) % GW] = 200
        cx += speed
        if cx >= GW:
            cx  = 0.0
            cy  = float(rng2.integers(0, GH))
            speed = rng2.uniform(0.8, 2.5)
        write_frame(w, buf)
    close(w, name)


# 26 – SPARSE STATIC
# ~8–12 random single pixels per frame light up independently,
# each at a random low-to-mid brightness. Between flashes: pure black.
# Feels like distant, slow cosmic noise.

def make_26_sparse_static() -> None:
    name = "26_sparse_static.webm"
    print(f"[26] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2600)
    buf  = np.zeros((GH, GW), dtype=np.float32)
    TOTAL = FPS * 55

    for _ in range(TOTAL):
        buf *= 0.78    # fast fade — each spark only lives 2–3 frames
        n = rng2.integers(2, 7)
        for _ in range(n):
            buf[rng2.integers(0, GH), rng2.integers(0, GW)] = rng2.uniform(40, 140)
        write_frame(w, buf)
    close(w, name)


# 27 – GRAVITY WELL
# 2 invisible gravity centres slowly drift across X.
# ~30 free pixels are attracted toward the nearest centre,
# moving 1 pixel per step. When they reach it they respawn far away.
# Trails show the paths converging.

def make_27_gravity_well() -> None:
    name = "27_gravity_well.webm"
    print(f"[27] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2700)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N   = 28
    px  = rng2.integers(0, GW, N).astype(float)
    py  = rng2.integers(0, GH, N).astype(float)
    br  = rng2.uniform(80, 180, N)

    # Two wells drifting slowly
    wx  = np.array([GW * 0.3, GW * 0.7], dtype=float)
    wy  = np.array([2.0, 6.0], dtype=float)
    wvx = np.array([0.05, -0.04])

    TOTAL = FPS * 65

    for _ in range(TOTAL):
        buf  *= 0.93
        wx   += wvx
        wx    = np.clip(wx, 20, GW - 20)
        wvx  *= np.where(np.abs(wx - wx.mean()) > GW * 0.35, -1, 1)

        for i in range(N):
            # Choose nearest well
            dists = [math.hypot(px[i] - wx[j], (py[i] - wy[j]) * 7.88)
                     for j in range(2)]
            j = int(np.argmin(dists))
            if dists[j] < 1.5:
                px[i] = rng2.uniform(0, GW)
                py[i] = rng2.uniform(0, GH - 1)
            else:
                # Move one step toward well
                ang  = math.atan2((wy[j] - py[i]) * 7.88, wx[j] - px[i])
                px[i] = np.clip(px[i] + math.cos(ang), 0, GW - 1)
                py[i] = np.clip(py[i] + math.sin(ang) / 7.88, 0, GH - 1)

            buf[int(py[i]), int(px[i])] = max(buf[int(py[i]), int(px[i])], br[i])
        write_frame(w, buf)
    close(w, name)


# 28 – PIXEL EROSION
# Start with a fully lit centre row (all 480 pixels on at 180).
# Each frame a random subset of pixels dims by a random amount.
# Some pixels re-ignite briefly. Row slowly erodes to black.
# Repeats with the next row.

def make_28_pixel_erosion() -> None:
    name = "28_pixel_erosion.webm"
    print(f"[28] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2800)

    TOTAL     = FPS * 65
    row_cycle = FPS * 22    # new row every ~22 s
    buf  = np.zeros((GH, GW), dtype=np.float32)
    cur_row   = GH // 2
    buf[cur_row, :] = 180

    for f in range(TOTAL):
        # Random erosion
        mask = rng2.random(GW) < 0.06
        buf[cur_row, mask] *= rng2.uniform(0.5, 0.9, mask.sum())
        # Rare re-ignition
        relight = rng2.random(GW) < 0.004
        buf[cur_row, relight] = rng2.uniform(80, 160, relight.sum())

        if f % row_cycle == 0 and f > 0:
            cur_row = rng2.integers(0, GH)
            buf[cur_row, :] = np.maximum(buf[cur_row, :], 180)

        write_frame(w, buf * 0.999)
    close(w, name)


# 29 – PIXEL REPULSION
# Pairs of pixels that strongly repel each other — they flee to opposite
# ends of the X axis, then slowly drift back toward centre and repel again.
# 4 repulsion pairs, each on a different depth row. Long fading trails.

def make_29_pixel_repulsion() -> None:
    name = "29_pixel_repulsion.webm"
    print(f"[29] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(2900)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    PAIRS = 4
    ax   = np.full(PAIRS, GW / 2 - 10, dtype=float)
    bx   = np.full(PAIRS, GW / 2 + 10, dtype=float)
    rows = rng2.choice(GH, PAIRS, replace=False)
    avx  = np.zeros(PAIRS)
    bvx  = np.zeros(PAIRS)
    br   = rng2.uniform(130, 210, PAIRS)
    TOTAL = FPS * 65

    for _ in range(TOTAL):
        buf *= 0.95
        for i in range(PAIRS):
            sep  = bx[i] - ax[i]
            # Repulsion force (inverse-square along X)
            f_rep = 800 / max(abs(sep), 1) ** 1.5
            # Gentle return-to-centre spring
            cx_a  = GW / 2 - 5
            cx_b  = GW / 2 + 5
            avx[i] += -f_rep - 0.002 * (ax[i] - cx_a)
            bvx[i] +=  f_rep - 0.002 * (bx[i] - cx_b)
            avx[i] *= 0.92   # drag
            bvx[i] *= 0.92
            ax[i]   = float(np.clip(ax[i] + avx[i], 0, GW - 1))
            bx[i]   = float(np.clip(bx[i] + bvx[i], 0, GW - 1))
            ry = rows[i]
            buf[ry, int(ax[i])] = max(buf[ry, int(ax[i])], br[i])
            buf[ry, int(bx[i])] = max(buf[ry, int(bx[i])], br[i])
        write_frame(w, buf)
    close(w, name)


# 30 – QUANTUM DOTS
# An extremely dim field (~3–8/255) of single pixels flicker on/off
# each frame completely independently with low probability.
# The effect is a barely-perceptible probabilistic shimmer — the ceiling
# seems almost off but never quite is.

def make_30_quantum_dots() -> None:
    name = "30_quantum_dots.webm"
    print(f"[30] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3000)
    TOTAL = FPS * 60
    P_ON  = 0.012     # ~5760 * 0.012 ≈ 69 pixels lit per frame

    for _ in range(TOTAL):
        canvas = (rng2.random((GH, GW)) < P_ON).astype(np.float32)
        canvas *= rng2.uniform(4, 22, (GH, GW)).astype(np.float32)
        write_frame(w, canvas)
    close(w, name)


# 31 – PIXEL DRIFT FIELD
# A slowly rotating 2D vector field guides 40 pixels.
# Each pixel moves 1 step per frame according to the local field direction.
# The field rotates completely every ~30 s. Fading trails.

def make_31_pixel_drift_field() -> None:
    name = "31_pixel_drift_field.webm"
    print(f"[31] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3100)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N   = 40
    px  = rng2.uniform(0, GW, N)
    py  = rng2.uniform(0, GH - 1, N)
    br  = rng2.uniform(80, 180, N)
    TOTAL       = FPS * 70
    FIELD_SPEED = 2 * math.pi / (FPS * 30)

    for f in range(TOTAL):
        buf *= 0.94
        theta = f * FIELD_SPEED   # global field rotation angle
        for i in range(N):
            # Local field: spiral + rotation
            dx_f = math.cos(theta + py[i] * 0.3)
            dy_f = math.sin(theta + px[i] * 0.015) * 0.25
            px[i] = (px[i] + dx_f) % GW
            py[i] = float(np.clip(py[i] + dy_f, 0, GH - 1))
            buf[int(py[i]), int(px[i])] = max(buf[int(py[i]), int(px[i])], br[i])
        write_frame(w, buf)
    close(w, name)


# 32 – PIXEL CONSTELLATIONS
# 6 fixed "star" pixels form a static dim pattern.
# Every 8–15 s a brief bright "edge" pulses between two neighbours
# (a straight line of pixels lights up for ~10 frames, then fades).
# Long dark gaps between connections. Feels like thinking.

def make_32_pixel_constellations() -> None:
    name = "32_pixel_constellations.webm"
    print(f"[32] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3200)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    STARS = 7
    sx = rng2.integers(20, GW - 20, STARS)
    sy = rng2.integers(0, GH, STARS)
    br = rng2.uniform(30, 60, STARS)   # dim resting glow

    TOTAL    = FPS * 75
    next_con = FPS * rng2.integers(5, 12)
    flash_px: list[tuple[int, int]] = []
    flash_t  = 0

    for f in range(TOTAL):
        buf *= 0.97
        # Resting stars always glow faintly
        for i in range(STARS):
            buf[sy[i], sx[i]] = max(buf[sy[i], sx[i]], br[i])

        # Flash connection between two random stars
        if f >= next_con:
            i, j = rng2.choice(STARS, 2, replace=False)
            x0, y0, x1, y1 = sx[i], sy[i], sx[j], sy[j]
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            flash_px = [(int(x0 + (x1-x0)*t/steps),
                         int(y0 + (y1-y0)*t/steps))
                        for t in range(steps + 1)]
            flash_t  = 14
            next_con = f + rng2.integers(FPS * 6, FPS * 14)

        if flash_t > 0:
            intensity = 180 * (flash_t / 14) ** 2
            for fx, fy in flash_px:
                if 0 <= fx < GW:
                    buf[fy, fx] = max(buf[fy, fx], intensity)
            flash_t -= 1

        write_frame(w, buf)
    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# Y-AXIS / DEPTH MOVEMENT SERIES  (33 – 42)
# All primary motion travels along Y (the 9-row depth axis).
# Y=0 = back of room / far wall.  Y=8 = front / audience edge.
# ══════════════════════════════════════════════════════════════════════════════

# 33 – DEPTH SCAN
# A full-width bright row sweeps from front (Y=8) to back (Y=0),
# bounces, converges to centre — mirroring scene 01 but on the Y axis.
# Because Y only has 9 steps the sweep is very deliberate and slow.

def make_33_depth_scan() -> None:
    name = "33_depth_scan.webm"
    print(f"[33] {name} …")
    w   = open_writer(name)
    buf = np.zeros((GH, GW), dtype=np.float32)

    SWEEP   = FPS * 6          # 6 s per half-sweep (slow across only 9 rows)
    BOUNCES = 6
    cur_y   = GH - 1

    for b in range(BOUNCES):
        end_y = 0 if b % 2 == 0 else GH - 1
        start = cur_y
        for i in range(SWEEP):
            buf *= 0.88
            t     = ease_in_out(i / (SWEEP - 1))
            cur_y = int(round(start + (end_y - start) * t))
            buf[cur_y, :] = np.maximum(buf[cur_y, :], 200)
            write_frame(w, buf)

    close(w, name)


# 34 – DEPTH RAIN  (drops travel front → back)
# Each column has an independent drop that starts at Y=8 (front)
# and travels toward Y=0 (back), leaving a fading trail.
# Physically: light appears near the audience and retreats into the room.

def make_34_depth_rain() -> None:
    name = "34_depth_rain.webm"
    print(f"[34] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3400)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    # For Y=9 rows, "speed" is in rows/frame — must be small
    col_y     = rng2.uniform(0, GH,      GW).astype(np.float32)
    col_speed = rng2.uniform(0.01, 0.06, GW).astype(np.float32)  # slow!
    col_bright= rng2.uniform(120, 220,   GW).astype(np.float32)
    TOTAL     = FPS * 60

    for _ in range(TOTAL):
        buf *= 0.80
        col_y -= col_speed   # move front→back (decreasing Y index)

        done = col_y < 0
        col_y[done]      = rng2.uniform(GH * 0.5, GH - 1, done.sum())
        col_speed[done]  = rng2.uniform(0.01, 0.06, done.sum())
        col_bright[done] = rng2.uniform(120, 220,   done.sum())

        iy   = np.clip(col_y.astype(int), 0, GH - 1)
        mask = col_y >= 0
        for x in np.where(mask)[0]:
            buf[iy[x], x] = max(buf[iy[x], x], col_bright[x])
        write_frame(w, buf)

    close(w, name)


# 35 – DEPTH CASCADE
# Rows activate one by one from back (Y=0) to front (Y=8),
# each staying on for a moment then fading. Repeats continuously.
# Variable intensity per row — feels like successive waves of light
# rolling toward the audience.

def make_35_depth_cascade() -> None:
    name = "35_depth_cascade.webm"
    print(f"[35] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3500)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    ROW_HOLD  = FPS * 2      # each row stays bright for ~2 s
    TOTAL     = FPS * 70
    seq_timer = 0
    cur_row   = 0            # start from back (Y=0)
    row_bright= rng2.uniform(100, 200, GH)

    for f in range(TOTAL):
        buf *= 0.94
        buf[cur_row, :] = np.maximum(buf[cur_row, :], row_bright[cur_row])
        seq_timer += 1
        if seq_timer >= ROW_HOLD:
            seq_timer = 0
            cur_row   = (cur_row + 1) % GH   # advance back→front
            row_bright[cur_row] = rng2.uniform(100, 200)
        write_frame(w, buf)

    close(w, name)


# 36 – DEPTH CURTAIN
# The whole ceiling is dark. A bright "curtain" rolls from back (Y=0)
# toward the front (Y=8), each newly-revealed row glowing then dimming
# as the curtain passes. Then it resets and rolls again.

def make_36_depth_curtain() -> None:
    name = "36_depth_curtain.webm"
    print(f"[36] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3600)

    TRAVEL   = FPS * 18    # 18 s to travel across 9 rows
    PAUSE_N  = FPS * 4     # dark pause between passes
    TOTAL    = FPS * 70

    f = 0
    while f < TOTAL:
        # Roll front → back
        for step in range(GH):
            for _ in range(TRAVEL // GH):
                if f >= TOTAL: break
                buf = np.zeros((GH, GW), dtype=np.float32)
                # All revealed rows glow, brightest at the edge
                for r in range(step + 1):
                    age   = step - r
                    glow  = 180 * math.exp(-age * 0.45)
                    buf[r, :] = glow
                write_frame(w, buf)
                f += 1
        # Dark pause
        for _ in range(PAUSE_N):
            if f >= TOTAL: break
            write_frame(w, np.zeros((GH, GW), dtype=np.float32))
            f += 1
        # Roll back → front (reverse)
        for step in range(GH - 1, -1, -1):
            for _ in range(TRAVEL // GH):
                if f >= TOTAL: break
                buf = np.zeros((GH, GW), dtype=np.float32)
                for r in range(step, GH):
                    age   = r - step
                    glow  = 180 * math.exp(-age * 0.45)
                    buf[r, :] = glow
                write_frame(w, buf)
                f += 1
        for _ in range(PAUSE_N):
            if f >= TOTAL: break
            write_frame(w, np.zeros((GH, GW), dtype=np.float32))
            f += 1

    close(w, name)


# 37 – Y-STRIPES DRIFT
# Each row has a different constant brightness that slowly cycles.
# Row intensities drift up and down at independent rates (all slow),
# so the ceiling reads as shifting horizontal bands of depth.
# Very quiet — feels like light through venetian blinds.

def make_37_y_stripes_drift() -> None:
    name = "37_y_stripes_drift.webm"
    print(f"[37] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3700)

    TOTAL   = FPS * 65
    periods = rng2.uniform(FPS * 8, FPS * 25, GH).astype(np.float32)
    phases  = rng2.uniform(0, 2 * math.pi, GH).astype(np.float32)
    peaks   = rng2.uniform(40, 140, GH).astype(np.float32)

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for y in range(GH):
            v = peaks[y] * (0.5 + 0.5 * math.sin(2 * math.pi * f / periods[y] + phases[y]))
            canvas[y, :] = v
        write_frame(w, canvas)

    close(w, name)


# 38 – DEPTH METEOR  (streaks travel along Y)
# Short bright streaks move along the Y axis (front→back or back→front),
# each streak spanning the full width at a single advancing row position.
# The streak "band" moves across the depth axis, leaving a trail.
# With Y=9, each streak traverses the full depth in ~60–120 frames.

def make_38_depth_meteor() -> None:
    name = "38_depth_meteor.webm"
    print(f"[38] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3800)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    MAX_STREAKS = 4

    def new_streak():
        direction = 1 if rng2.random() < 0.5 else -1
        return {
            "y":      float(GH - 1 if direction == -1 else 0),
            "vy":     direction * rng2.uniform(0.04, 0.12),   # rows/frame
            "bright": rng2.uniform(130, 220),
            "tail":   rng2.integers(2, 5),        # tail in rows
            "xmask":  rng2.random(GW) > rng2.uniform(0.1, 0.4),  # sparse X
        }

    streaks = [new_streak() for _ in range(MAX_STREAKS)]
    for i, s in enumerate(streaks):
        s["y"] = float(rng2.uniform(0, GH - 1))   # stagger starts

    TOTAL = FPS * 65

    for _ in range(TOTAL):
        buf *= 0.85
        for s in streaks:
            s["y"] += s["vy"]
            iy = int(round(s["y"]))
            for t in range(s["tail"]):
                ry = iy - t * int(np.sign(s["vy"]))
                if 0 <= ry < GH:
                    fade = ((s["tail"] - t) / s["tail"]) ** 2
                    row_val = s["bright"] * fade * s["xmask"].astype(np.float32)
                    np.maximum(buf[ry, :], row_val, out=buf[ry, :])

            off = s["y"] < -1 or s["y"] > GH
            if off:
                s.update(new_streak())
        write_frame(w, buf)

    close(w, name)


# 39 – ROW SEQUENCER
# Each of the 9 rows activates in a programmable sequence with variable
# timing and brightness — like a step sequencer playing a rhythm on depth.
# A random 9-step pattern is generated, then played in a loop.
# Each "beat" lights the row for 1–4 frames. Gaps between beats.

def make_39_row_sequencer() -> None:
    name = "39_row_sequencer.webm"
    print(f"[39] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(3900)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    # Random sequence: for each step, which row, how long on, how long off
    SEQ_LEN = 12
    seq_row    = rng2.integers(0, GH,         SEQ_LEN)
    seq_on     = rng2.integers(2,  8,          SEQ_LEN)
    seq_off    = rng2.integers(4, 22,          SEQ_LEN)
    seq_bright = rng2.uniform(100, 220,        SEQ_LEN)

    TOTAL = FPS * 65
    step  = 0
    timer = 0
    phase = "on"   # or "off"

    for _ in range(TOTAL):
        buf *= 0.88
        if phase == "on":
            buf[seq_row[step], :] = np.maximum(
                buf[seq_row[step], :], seq_bright[step])
            timer += 1
            if timer >= seq_on[step]:
                timer = 0
                phase = "off"
        else:
            timer += 1
            if timer >= seq_off[step]:
                timer = 0
                step  = (step + 1) % SEQ_LEN
                phase = "on"
        write_frame(w, buf)

    close(w, name)


# 40 – DEPTH WANDER
# Like scene 18 (single pixel wanderer) but biased strongly toward Y movement.
# 3 independent pixels each drift mostly up/down (Y axis) with rare X steps.
# On the 480×9 grid this creates vertical lines of phosphor that drift
# slowly front-to-back, occasionally shifting to a new X column.

def make_40_depth_wander() -> None:
    name = "40_depth_wander.webm"
    print(f"[40] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4000)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N  = 3
    px = rng2.integers(60, GW - 60, N)
    py = rng2.integers(0, GH, N)
    br = rng2.uniform(100, 190, N)
    TOTAL = FPS * 70

    for _ in range(TOTAL):
        buf *= 0.96
        for i in range(N):
            buf[py[i], px[i]] = max(buf[py[i], px[i]], br[i])
            r = rng2.random()
            if r < 0.45:        py[i] = min(py[i] + 1, GH - 1)  # forward
            elif r < 0.90:      py[i] = max(py[i] - 1, 0)        # back
            elif r < 0.95:      px[i] = min(px[i] + 1, GW - 1)   # rare X+
            else:               px[i] = max(px[i] - 1, 0)        # rare X-
        write_frame(w, buf)

    close(w, name)


# 41 – FRONT GLOW PULSE
# The front row (Y=8, closest to audience) pulses with a slow sine.
# Each subsequent row picks up the pulse delayed by ~1.5 s per row,
# creating a wave that appears to emanate from the audience edge
# and propagate toward the back wall.

def make_41_front_glow_pulse() -> None:
    name = "41_front_glow_pulse.webm"
    print(f"[41] {name} …")
    w    = open_writer(name)

    TOTAL   = FPS * 65
    PERIOD  = FPS * 9        # 9 s per pulse cycle
    DELAY   = FPS * 1.5      # 1.5 s delay per row
    PEAK    = 160.0

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for y in range(GH):
            front_row = GH - 1 - y          # Y=8 is front, maps to y=0 here
            delay     = (GH - 1 - front_row) * DELAY
            phase     = 2 * math.pi * (f - delay) / PERIOD
            val       = PEAK * max(0.0, math.sin(phase)) ** 2
            canvas[front_row, :] = val
        write_frame(w, canvas)

    close(w, name)


# 42 – Y-AXIS RIPPLE CHAIN
# Ripples originate at random X positions on the FRONT row (Y=8)
# and propagate only along Y (toward the back). Each ripple is a
# Gaussian brightness band that expands outward in Y over time.
# Gives the sensation that something is emitting from the audience edge.

def make_42_y_axis_ripple_chain() -> None:
    name = "42_y_axis_ripple_chain.webm"
    print(f"[42] {name} …")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4200)

    TOTAL      = FPS * 65
    SPAWN_INT  = FPS * 3
    MAX_R      = 5
    ripples: list[dict] = []
    xs = np.arange(GW, dtype=np.float32)
    ys = np.arange(GH, dtype=np.float32)

    def new_rip():
        return {
            "ox":    rng2.uniform(0, GW),    # X origin
            "r":     0.0,                    # current Y radius
            "speed": rng2.uniform(0.05, 0.12),
            "peak":  rng2.uniform(60, 150),
            "xsig":  rng2.uniform(8, 30),    # X spread of the source
        }

    for f in range(TOTAL):
        if f % SPAWN_INT == 0 and len(ripples) < MAX_R:
            ripples.append(new_rip())

        canvas = np.zeros((GH, GW), dtype=np.float32)
        alive  = []
        for rp in ripples:
            rp["r"] += rp["speed"]
            # X envelope: Gaussian around origin X
            x_env = np.exp(-0.5 * ((xs - rp["ox"]) / rp["xsig"]) ** 2)
            # Y ring: front row = GH-1, so ring centre is GH-1, expands toward 0
            y_pos  = (GH - 1) - rp["r"]
            y_ring = np.exp(-0.5 * ((ys - y_pos) / 0.7) ** 2)
            canvas += np.outer(y_ring, x_env) * rp["peak"]
            if y_pos > -2:
                alive.append(rp)
        ripples = alive

        write_frame(w, np.clip(canvas, 0, 255))

    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# NATURAL PATTERNS – WAVES / FLOWS / SWARMS / PARTICLES  (43 – 57)
# ══════════════════════════════════════════════════════════════════════════════

# 43 – MURMURATION
# Boids flocking algorithm: ~80 particles with cohesion (move toward flock
# centroid), separation (avoid crowding) and alignment (match neighbour
# velocity).  Wraps in X, bounces in Y.  Long phosphor trails.

def make_43_murmuration() -> None:
    name = "43_murmuration.webm"
    print(f"[43] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4300)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N  = 80
    px = rng2.uniform(GW * 0.2, GW * 0.8, N)
    py = rng2.uniform(0.5, GH - 1.5, N)
    vx = rng2.uniform(-2.0, 2.0, N)
    vy = rng2.uniform(-0.12, 0.12, N)
    br = rng2.uniform(100, 200, N)

    TOTAL = FPS * 70
    VR    = 40.0    # visual range (px)
    SR    = 12.0    # separation range
    MSPD  = 2.5     # max speed

    for _ in range(TOTAL):
        buf *= 0.93
        iy = np.clip(py.astype(int), 0, GH - 1)
        ix = px.astype(int) % GW
        for i in range(N):
            buf[iy[i], ix[i]] = max(buf[iy[i], ix[i]], float(br[i]))

        # Pairwise distances (X wrapped, Y physically scaled)
        dx = px[:, None] - px[None, :]
        dx = (dx + GW // 2) % GW - GW // 2        # shortest wrap
        dy = (py[:, None] - py[None, :]) * 7.88    # row ≈ 7.88 m
        dist = np.sqrt(dx ** 2 + dy ** 2) + np.eye(N) * 9999.0

        vis  = dist < VR
        sep  = dist < SR
        vc   = vis.sum(axis=1).astype(float)
        safe = np.where(vc > 0, vc, 1.0)

        # Cohesion
        cx = (vis * px[None, :]).sum(axis=1) / safe
        cy = (vis * py[None, :]).sum(axis=1) / safe
        vx += (cx - px) * 0.0005 * (vc > 0)
        vy += (cy - py) * 0.004  * (vc > 0)

        # Separation (push away from near neighbours)
        vx += (sep * dx).sum(axis=1) * 0.012
        vy += (sep * (py[:, None] - py[None, :])).sum(axis=1) * 0.08

        # Alignment
        avg_vx = (vis * vx[None, :]).sum(axis=1) / safe
        avg_vy = (vis * vy[None, :]).sum(axis=1) / safe
        vx += (avg_vx - vx) * 0.05 * (vc > 0)
        vy += (avg_vy - vy) * 0.05 * (vc > 0)

        # Speed limit
        spd = np.sqrt(vx ** 2 + (vy * 7.88) ** 2) + 1e-9
        clip = spd > MSPD
        vx[clip] *= MSPD / spd[clip]
        vy[clip] *= MSPD / spd[clip]

        px = (px + vx) % GW
        py = np.clip(py + vy, 0.0, float(GH - 1))
        wall = (py <= 0) | (py >= GH - 1)
        vy[wall] *= -1

        write_frame(w, buf)
    close(w, name)


# 44 – KELP SWAY
# Each of the 480 columns has a "kelp tip" that oscillates in Y.
# A phase gradient across X (like a slow current) makes the whole
# ceiling sway in waves.  A soft Gaussian glow decays below each tip.

def make_44_kelp_sway() -> None:
    name = "44_kelp_sway.webm"
    print(f"[44] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4400)

    TOTAL    = FPS * 65
    rest_y   = rng2.uniform(2.5, 6.0, GW).astype(np.float32)
    amp      = rng2.uniform(1.2, 3.2, GW).astype(np.float32)
    freq     = rng2.uniform(0.012, 0.035, GW).astype(np.float32)
    phase_x  = np.linspace(0, 6 * math.pi, GW, dtype=np.float32)
    br       = rng2.uniform(100, 200, GW).astype(np.float32)
    ys_arr   = np.arange(GH, dtype=np.float32)[:, None]   # (GH, 1)

    drift = 0.0
    for f in range(TOTAL):
        drift     += 0.008
        y_tips     = rest_y + amp * np.sin(freq * f + phase_x + drift)
        y_tips     = np.clip(y_tips, 0.0, GH - 1.0)           # (GW,)
        # Gaussian glow centred on tip position
        canvas = br[None, :] * np.exp(-0.5 * ((ys_arr - y_tips[None, :]) / 1.1) ** 2)
        write_frame(w, np.clip(canvas, 0, 255))
    close(w, name)


# 45 – WATER CAUSTICS
# N plane waves at random angles interfere; squaring the sum produces
# the bright-node / dark-groove pattern of underwater caustic light.
# Nodes drift as each wave has a different angular speed.

def make_45_caustics() -> None:
    name = "45_caustics.webm"
    print(f"[45] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4500)

    TOTAL  = FPS * 65
    xs = np.linspace(0, 2 * math.pi * 4, GW, dtype=np.float32)
    ys = np.linspace(0, 2 * math.pi * 4 * GH / GW, GH, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    N_WAVES = 6
    angles = rng2.uniform(0, 2 * math.pi, N_WAVES)
    k_mag  = rng2.uniform(0.8, 2.0, N_WAVES).astype(np.float32)
    kx     = (k_mag * np.cos(angles)).astype(np.float32)
    ky     = (k_mag * np.sin(angles)).astype(np.float32)
    omega  = rng2.uniform(0.018, 0.055, N_WAVES).astype(np.float32)
    phi    = rng2.uniform(0, 2 * math.pi, N_WAVES).astype(np.float32)

    for f in range(TOTAL):
        field = sum(np.sin(kx[i] * XX + ky[i] * YY + omega[i] * f + phi[i])
                    for i in range(N_WAVES))
        canvas = field ** 2
        mx = canvas.max()
        if mx > 0:
            canvas = canvas / mx * 170.0
        write_frame(w, canvas.astype(np.float32))
    close(w, name)


# 46 – SMOKE PLUME
# An emitter on the bottom row releases particles that drift upward
# (decreasing Y) with turbulent sinusoidal horizontal displacement.
# Phosphor buffer gives slow trailing wisps.

def make_46_smoke_plume() -> None:
    name = "46_smoke_plume.webm"
    print(f"[46] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4600)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N        = 55
    emitter_x = float(GW // 2)
    px  = rng2.uniform(emitter_x - 6, emitter_x + 6, N)
    py  = np.full(N, float(GH - 1))
    age = rng2.uniform(0, 45, N)
    br  = rng2.uniform(70, 160, N)
    drift_ph = rng2.uniform(0, 2 * math.pi, N)

    TOTAL = FPS * 65

    for f in range(TOTAL):
        buf *= 0.90
        for i in range(N):
            age[i] += 1
            py[i]  -= 0.07
            px[i]  += 0.35 * math.sin(age[i] * 0.11 + drift_ph[i]) + \
                       rng2.uniform(-0.15, 0.15)
            if py[i] < 0 or not (0 <= px[i] < GW):
                px[i]     = rng2.uniform(emitter_x - 8, emitter_x + 8)
                py[i]     = float(GH - 1)
                age[i]    = 0
                drift_ph[i] = rng2.uniform(0, 2 * math.pi)
            ix = int(np.clip(px[i], 0, GW - 1))
            iy = int(np.clip(py[i], 0, GH - 1))
            buf[iy, ix] = max(buf[iy, ix], br[i])
        write_frame(w, buf)
    close(w, name)


# 47 – LAVA LAMP
# 7 soft Gaussian blobs oscillate slowly up and down (Y axis) with
# independent periods.  X positions drift lazily.  Dark base.

def make_47_lava_lamp() -> None:
    name = "47_lava_lamp.webm"
    print(f"[47] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4700)

    TOTAL = FPS * 70
    N = 7
    blob_x     = rng2.uniform(30, GW - 30, N).astype(float)
    blob_vx    = rng2.uniform(-0.04, 0.04, N)
    blob_sig_x = rng2.uniform(18, 45, N).astype(np.float32)
    blob_sig_y = rng2.uniform(0.9, 2.2, N).astype(np.float32)
    blob_period= rng2.uniform(FPS * 9, FPS * 18, N)
    blob_phase = rng2.uniform(0, 2 * math.pi, N)
    blob_peak  = rng2.uniform(70, 150, N).astype(np.float32)

    xs = np.arange(GW, dtype=np.float32)
    ys = np.arange(GH, dtype=np.float32)

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for i in range(N):
            blob_y_i = (GH - 1) / 2 + ((GH - 1) / 2 - 0.5) * math.sin(
                2 * math.pi * f / blob_period[i] + blob_phase[i])
            blob_x[i] += blob_vx[i]
            blob_x[i]  = float(np.clip(blob_x[i], 25, GW - 25))
            if blob_x[i] <= 25 or blob_x[i] >= GW - 25:
                blob_vx[i] *= -1
            x_env = np.exp(-0.5 * ((xs - blob_x[i]) / blob_sig_x[i]) ** 2)
            y_env = np.exp(-0.5 * ((ys - blob_y_i) / blob_sig_y[i]) ** 2)
            canvas += np.outer(y_env, x_env) * blob_peak[i]
        write_frame(w, np.clip(canvas, 0, 200))
    close(w, name)


# 48 – BIOLUMINESCENCE
# A trigger wave sweeps left→right across 480 columns over ~6 s.
# Each column it crosses lights up (bright flash, slow phosphor decay).
# Multiple waves repeat with dark gaps — like a bow wave through
# a bay of dinoflagellates.

def make_48_bioluminescence() -> None:
    name = "48_bioluminescence.webm"
    print(f"[48] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4800)

    TOTAL       = FPS * 70
    WAVE_PERIOD = FPS * 22
    wave_spd    = GW / (FPS * 5)    # cross full width in 5 s

    col_bright  = np.zeros(GW, dtype=np.float32)
    xs          = np.arange(GW, dtype=np.float32)
    wave_x      = -30.0
    next_wave   = 0

    for f in range(TOTAL):
        if f >= next_wave:
            wave_x += wave_spd
            if wave_x > GW + 30:
                wave_x    = -30.0
                next_wave = f + WAVE_PERIOD
            triggered = np.abs(xs - wave_x) < wave_spd * 2.5
            col_bright[triggered] = rng2.uniform(170, 240, triggered.sum())

        col_bright *= 0.93
        canvas = np.outer(np.ones(GH, dtype=np.float32), col_bright)
        write_frame(w, canvas)
    close(w, name)


# 49 – SHOAL SCATTER
# 40 "fish" pixels swim as a tight school, slowly circling.
# Every 10–16 s a predator pulse strikes at a random position —
# all fish scatter radially, then slowly regroup.  Fading trails.

def make_49_shoal_scatter() -> None:
    name = "49_shoal_scatter.webm"
    print(f"[49] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(4900)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N = 40
    cx_tgt, cy_tgt = float(GW // 2), float(GH // 2)
    px = rng2.uniform(cx_tgt - 20, cx_tgt + 20, N)
    py = np.clip(rng2.uniform(cy_tgt - 1.5, cy_tgt + 1.5, N), 0, GH - 1)
    vx = rng2.uniform(-0.4, 0.4, N)
    vy = rng2.uniform(-0.03, 0.03, N)
    br = rng2.uniform(120, 200, N)

    TOTAL       = FPS * 70
    scatter     = False
    scatter_t   = 0
    next_event  = rng2.integers(FPS * 8, FPS * 14)
    pred_x = pred_y = 0.0

    for f in range(TOTAL):
        buf *= 0.91

        if f >= next_event and not scatter:
            scatter    = True
            scatter_t  = 0
            pred_x     = rng2.uniform(GW * 0.2, GW * 0.8)
            pred_y     = rng2.uniform(1, GH - 2)
            next_event = f + rng2.integers(FPS * 10, FPS * 18)

        if scatter:
            scatter_t += 1
            for i in range(N):
                ddx = px[i] - pred_x
                ddy = (py[i] - pred_y) * 7.88
                d   = math.sqrt(ddx ** 2 + ddy ** 2) + 1e-3
                vx[i] += ddx / d * 2.5
                vy[i] += (py[i] - pred_y) / d * 0.5
            if scatter_t > FPS * 3:
                scatter = False
        else:
            # Slowly circle a drifting centre
            cx_tgt = GW * (0.3 + 0.4 * math.sin(f * 0.003))
            for i in range(N):
                vx[i] += (cx_tgt - px[i]) * 0.002
                vy[i] += (cy_tgt - py[i]) * 0.01

        vx *= 0.92
        vy *= 0.92
        px = np.clip(px + vx, 0, GW - 1)
        py = np.clip(py + vy, 0, GH - 1)

        iy = py.astype(int)
        ix = px.astype(int)
        for i in range(N):
            buf[iy[i], ix[i]] = max(buf[iy[i], ix[i]], br[i])
        write_frame(w, buf)
    close(w, name)


# 50 – THERMAL CONVECTION
# Each column has an oscillating "temperature" out of phase with its
# neighbours — hot columns form bright rising plumes (concentrated near
# Y=0), cool columns sink (bright near Y=8).  Creates rolling convection cells.

def make_50_thermal_convection() -> None:
    name = "50_thermal_convection.webm"
    print(f"[50] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5000)

    TOTAL   = FPS * 65
    periods = rng2.uniform(FPS * 6, FPS * 16, GW).astype(np.float32)
    phases  = np.linspace(0, 6 * math.pi, GW, dtype=np.float32) + \
              rng2.uniform(-0.5, 0.5, GW).astype(np.float32)
    ys_arr  = np.arange(GH, dtype=np.float32)[:, None]

    for f in range(TOTAL):
        # Temperature 0–1 per column
        temp = 0.5 + 0.5 * np.sin(2 * math.pi * f / periods + phases)  # (GW,)
        # Hot → plume peaks near Y=0; cool → sinks to Y=GH-1
        y_peak = (1.0 - temp) * (GH - 1)       # (GW,)
        sigma  = 1.2 + temp * 1.5              # (GW,) wider when cold
        canvas = np.exp(-0.5 * ((ys_arr - y_peak[None, :]) / sigma[None, :]) ** 2)
        canvas *= (50 + temp[None, :] * 140)
        write_frame(w, np.clip(canvas.astype(np.float32), 0, 220))
    close(w, name)


# 51 – CHLADNI FIGURES
# Two superimposed standing-wave modes with slowly evolving frequency
# ratios.  Bright regions = antinodes; dark grooves = nodal lines.
# The ratio shifts every ~30 s, reorganising the pattern.

def make_51_chladni() -> None:
    name = "51_chladni.webm"
    print(f"[51] {name} ...")
    w    = open_writer(name)

    TOTAL  = FPS * 70
    xs = np.linspace(0, 2 * math.pi, GW, dtype=np.float32)
    ys = np.linspace(0, 2 * math.pi, GH, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    base = 2 * math.pi / (FPS * 30)

    for f in range(TOTAL):
        f1x = 1.5 + 0.6 * math.sin(base * f)
        f1y = 2.0 + 0.4 * math.cos(base * f * 0.71)
        f2x = 3.0 + 0.5 * math.sin(base * f * 1.3)
        f2y = 1.0 + 0.7 * math.cos(base * f * 0.43)
        field  = np.sin(f1x * XX) * np.sin(f1y * YY) + \
                 np.sin(f2x * XX) * np.sin(f2y * YY)
        # Sand gathers at NODES (zero crossings) → bright at |field| ≈ 0
        canvas = np.maximum(0.0, 1.0 - np.abs(field) * 2.2) ** 2 * 190.0
        write_frame(w, canvas.astype(np.float32))
    close(w, name)


# 52 – RIVER FLOW
# A slowly-evolving layered-sine flow field guides 55 particles.
# The field has curl, producing visible eddies and meanders.
# Particles leave long phosphor trails.

def make_52_river_flow() -> None:
    name = "52_river_flow.webm"
    print(f"[52] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5200)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N  = 55
    px = rng2.uniform(0, GW, N)
    py = rng2.uniform(0, GH - 1, N)
    br = rng2.uniform(80, 180, N)
    ft = 0.0

    TOTAL = FPS * 70

    def uv(x, y, t):
        ux = (0.9 + 0.4 * math.sin(y * 0.55 + t * 0.013)
              + 0.25 * math.sin(x * 0.018 + y * 0.8 + t * 0.009))
        uy = (0.12 * math.sin(x * 0.028 - t * 0.011)
              + 0.08 * math.cos(y * 1.1 + t * 0.017))
        return ux, uy

    for _ in range(TOTAL):
        buf *= 0.93
        ft  += 1.0
        for i in range(N):
            iy = int(np.clip(py[i], 0, GH - 1))
            ix = int(px[i]) % GW
            buf[iy, ix] = max(buf[iy, ix], br[i])
            ux, uy = uv(px[i], py[i], ft)
            px[i]  = (px[i] + ux) % GW
            py[i]  = float(np.clip(py[i] + uy * 0.22, 0, GH - 1))
            if py[i] <= 0 or py[i] >= GH - 1:
                vy_bounce = -uy * 0.22
                py[i] = float(np.clip(py[i] + vy_bounce, 0, GH - 1))
        write_frame(w, buf)
    close(w, name)


# 53 – MYCELIUM GROWTH
# Branching fungal tendrils grow from 3 seed points.
# Tips step mostly along X with tiny Y deviation; branch randomly.
# Growth is slow — older segments fade, new growth glows brightly.

def make_53_mycelium() -> None:
    name = "53_mycelium.webm"
    print(f"[53] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5300)
    canvas = np.zeros((GH, GW), dtype=np.float32)

    TOTAL = FPS * 70
    tips: list[dict] = []

    def spawn(x, y, angle, energy=1.0):
        tips.append({"x": float(x), "y": float(y),
                     "angle": angle, "energy": energy})

    for _ in range(3):
        sx = int(rng2.integers(GW // 5, 4 * GW // 5))
        sy = int(rng2.integers(1, GH - 1))
        for _ in range(4):
            spawn(sx, sy, rng2.uniform(0, 2 * math.pi))

    for f in range(TOTAL):
        canvas *= 0.998
        new_tips = []
        for tip in tips:
            tip["energy"] *= 0.986
            if tip["energy"] < 0.07:
                continue
            tip["angle"] += rng2.uniform(-0.28, 0.28)
            tip["x"] = (tip["x"] + math.cos(tip["angle"]) * 1.0) % GW
            tip["y"] = float(np.clip(tip["y"] + math.sin(tip["angle"]) * 0.11,
                                     0, GH - 1))
            ix, iy = int(tip["x"]) % GW, int(tip["y"])
            canvas[iy, ix] = min(canvas[iy, ix] + tip["energy"] * 55, 155)
            new_tips.append(tip)
            if rng2.random() < 0.0025 and tip["energy"] > 0.4:
                spawn(tip["x"], tip["y"],
                      tip["angle"] + rng2.choice([-1, 1]) * rng2.uniform(0.4, 1.1),
                      tip["energy"] * 0.65)
        tips = new_tips
        if f % (FPS * 18) == 0:
            sx = int(rng2.integers(0, GW))
            sy = int(rng2.integers(1, GH - 1))
            for _ in range(3):
                spawn(sx, sy, rng2.uniform(0, 2 * math.pi))
        write_frame(w, canvas)
    close(w, name)


# 54 – PHEROMONE TRAIL
# 30 ants walk between two food sources at opposite X ends (X≈15 and
# X≈465).  Each ant deposits pheromone brightness along its path.
# Pheromone evaporates slowly; converges to 1–2 bright horizontal paths.

def make_54_pheromone_trail() -> None:
    name = "54_pheromone_trail.webm"
    print(f"[54] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5400)

    TOTAL     = FPS * 70
    EVAP      = 0.993
    pheromone = np.zeros((GH, GW), dtype=np.float32)

    N      = 30
    px     = rng2.uniform(0, GW, N)
    py     = rng2.uniform(1, GH - 2, N)
    vx     = rng2.uniform(-1, 1, N)
    vy     = rng2.uniform(-0.05, 0.05, N)
    target = (rng2.random(N) > 0.5).astype(int)
    food_x = [15.0, float(GW - 15)]
    food_y = [4.0, 4.0]

    for _ in range(TOTAL):
        pheromone *= EVAP

        for i in range(N):
            ix = int(np.clip(px[i], 0, GW - 1))
            iy = int(np.clip(py[i], 0, GH - 1))
            pheromone[iy, ix] = min(pheromone[iy, ix] + 14.0, 190.0)

            tx, ty = food_x[target[i]], food_y[target[i]]
            d = math.hypot(px[i] - tx, (py[i] - ty) * 7.88)
            if d < 3.0:
                target[i] = 1 - target[i]
            else:
                vx[i] += (tx - px[i]) * 0.006
                vy[i] += (ty - py[i]) * 0.02
                vx[i] += rng2.uniform(-0.25, 0.25)
                vy[i] += rng2.uniform(-0.018, 0.018)

            vx[i] *= 0.84
            vy[i] *= 0.84
            spd = math.sqrt(vx[i] ** 2 + (vy[i] * 7.88) ** 2) + 1e-9
            if spd > 2.0:
                vx[i] *= 2.0 / spd
                vy[i] *= 2.0 / spd
            px[i] = float(np.clip(px[i] + vx[i], 0, GW - 1))
            py[i] = float(np.clip(py[i] + vy[i], 0, GH - 1))

        canvas = pheromone.copy()
        for i in range(N):
            canvas[int(np.clip(py[i], 0, GH - 1)),
                   int(np.clip(px[i], 0, GW - 1))] = 240.0
        write_frame(w, np.clip(canvas, 0, 255))
    close(w, name)


# 55 – CORAL PULSE
# 8 fixed "polyp" positions along the ceiling each pulse radially at
# their own period (3–8 s).  Rings expand outward and fade quickly.
# Rhythmic, organic, and quiet.

def make_55_coral_pulse() -> None:
    name = "55_coral_pulse.webm"
    print(f"[55] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5500)

    TOTAL    = FPS * 65
    N        = 8
    polyp_x  = rng2.uniform(25, GW - 25, N)
    polyp_y  = rng2.uniform(1.5, GH - 2.5, N)
    period   = rng2.uniform(FPS * 3, FPS * 8, N)
    phase    = rng2.uniform(0, 2 * math.pi, N)
    peak     = rng2.uniform(100, 200, N)
    xs = np.arange(GW, dtype=np.float32)
    ys = np.arange(GH, dtype=np.float32)

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for i in range(N):
            t = 0.5 + 0.5 * math.sin(2 * math.pi * f / period[i] + phase[i])
            r = t * 14.0
            dx   = xs - polyp_x[i]
            dy   = (ys - polyp_y[i]) * 3.5
            dist = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
            ring = np.exp(-0.5 * ((dist - r) / 1.6) ** 2) * peak[i] * t
            canvas = np.maximum(canvas, ring)
        write_frame(w, np.clip(canvas, 0, 230))
    close(w, name)


# 56 – DEEP CURRENT  (laminar / Poiseuille shear)
# Each of the 9 rows flows at a different speed — fastest in the middle
# rows, slowest at the edges — like viscous laminar flow.  50 particles
# drift rightward, shear makes adjacent rows visibly separate over time.

def make_56_deep_current() -> None:
    name = "56_deep_current.webm"
    print(f"[56] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5600)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N  = 50
    px = rng2.uniform(0, GW, N)
    py = rng2.uniform(0, GH - 1, N)
    br = rng2.uniform(80, 170, N)

    mid = (GH - 1) / 2.0
    row_speed = np.array([
        max(0.06, 0.7 * (1.0 - ((y - mid) / (mid + 0.01)) ** 2))
        for y in range(GH)
    ], dtype=np.float32)

    TOTAL = FPS * 65
    for _ in range(TOTAL):
        buf *= 0.93
        for i in range(N):
            row = int(np.clip(py[i], 0, GH - 1))
            buf[row, int(px[i]) % GW] = max(buf[row, int(px[i]) % GW], br[i])
            px[i] = (px[i] + row_speed[row]) % GW
            py[i] = float(np.clip(py[i] + rng2.uniform(-0.018, 0.018), 0, GH - 1))
        write_frame(w, buf)
    close(w, name)


# 57 – PHOSPHENE PULSE
# Bright concentric rings pulse outward from a fixed centre point,
# faster and more rhythmic than the depth ripples.  Each ring spawns
# every 5 s and fades as it expands.  Multiple simultaneous rings.
# Like the rings you see when pressing on closed eyes.

def make_57_phosphene() -> None:
    name = "57_phosphene.webm"
    print(f"[57] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5700)

    TOTAL      = FPS * 65
    SPAWN_INT  = FPS * 5
    pulses: list[dict] = []
    xs = np.arange(GW, dtype=np.float32)
    ys = np.arange(GH, dtype=np.float32)

    def new_pulse():
        return {
            "ox":    rng2.uniform(GW * 0.2, GW * 0.8),
            "oy":    rng2.uniform(1.0, GH - 2.0),
            "r":     0.0,
            "speed": rng2.uniform(1.8, 3.8),
            "peak":  rng2.uniform(120, 230),
            "width": rng2.uniform(1.0, 2.5),
        }

    for f in range(TOTAL):
        if f % SPAWN_INT == 0 and len(pulses) < 6:
            pulses.append(new_pulse())
        canvas = np.zeros((GH, GW), dtype=np.float32)
        alive  = []
        for p in pulses:
            p["r"] += p["speed"]
            dx   = xs - p["ox"]
            dy   = (ys - p["oy"]) * 3.8
            dist = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
            fade = max(0.0, 1.0 - p["r"] / 90.0)
            ring = np.exp(-0.5 * ((dist - p["r"]) / p["width"]) ** 2) * p["peak"] * fade
            canvas = np.maximum(canvas, ring)
            if p["r"] < 100:
                alive.append(p)
        pulses = alive
        write_frame(w, np.clip(canvas, 0, 255))
    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# ORGANIC Y-AXIS / DEPTH MOVEMENTS  (58 – 65)
# All motion is primarily along Y (the 9-row depth axis).
# Y=0 = back wall.  Y=8 = audience edge (front).
# ══════════════════════════════════════════════════════════════════════════════

# 58 – BREATH OF THE DEEP
# The whole ceiling inhales slowly from the back (Y=0) brightening toward
# the front (Y=8), holds, then exhales back.  Brightness curves are
# sigmoid-shaped so the transition zone is soft and biological.
# A faint low-frequency X ripple modulates the envelope to avoid
# the effect reading as a flat wash.

def make_58_breath_of_deep() -> None:
    name = "58_breath_of_deep.webm"
    print(f"[58] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5800)

    TOTAL  = FPS * 70
    PERIOD = FPS * 16      # 16 s per full breath cycle
    xs     = np.arange(GW, dtype=np.float32)
    ys     = np.arange(GH, dtype=np.float32)

    for f in range(TOTAL):
        phase = 2 * math.pi * f / PERIOD
        # Breath envelope: 0 (exhale) → 1 (inhale) slow sinusoid
        breath = 0.5 + 0.5 * math.sin(phase)
        # Sigmoid front-to-back reveal: front rows light up first when inhaling
        # front_edge moves from Y=8 (front) to Y=0 (back) as breath increases
        edge   = (1.0 - breath) * (GH - 1)      # 0=full, GH-1=none
        y_sig  = np.clip(ys - edge, 0, GH - 1) / max(GH - 1 - edge, 0.01)
        y_env  = 1.0 / (1.0 + np.exp(-6.0 * (y_sig - 0.5)))  # sigmoid

        # Subtle X ripple — organic texture, not geometric
        x_ripple = 1.0 + 0.07 * np.sin(xs * 0.042 + f * 0.019) + \
                   0.04 * np.sin(xs * 0.018 - f * 0.007)

        canvas = np.outer(y_env, x_ripple) * breath * 180.0
        write_frame(w, np.clip(canvas, 0, 180))
    close(w, name)


# 59 – SILT SETTLING
# Bright particles are born at a random Y row (like disturbed sediment)
# and slowly sink — move from whatever row toward Y=8 (front/floor).
# They accumulate and pool in the front rows, then slowly evaporate.
# New disturbance events fire every ~10 s.

def make_59_silt_settling() -> None:
    name = "59_silt_settling.webm"
    print(f"[59] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(5900)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N         = 60
    px        = rng2.uniform(0, GW, N)
    py        = rng2.uniform(0, GH - 1, N).astype(float)
    vy        = rng2.uniform(0.008, 0.040, N)   # sink speed rows/frame
    settled   = np.zeros(N, dtype=bool)
    br        = rng2.uniform(80, 170, N)
    TOTAL     = FPS * 75
    next_dist = FPS * 10

    for f in range(TOTAL):
        buf *= 0.985   # very slow evaporation / pooling glow

        for i in range(N):
            if not settled[i]:
                py[i] = min(py[i] + vy[i], float(GH - 1))
                if py[i] >= GH - 1:
                    settled[i] = True
            iy = int(np.clip(py[i], 0, GH - 1))
            ix = int(np.clip(px[i], 0, GW - 1))
            buf[iy, ix] = max(buf[iy, ix], br[i])

        if f == next_dist:
            row_disturb = rng2.integers(0, GH)
            idxs = rng2.choice(N, N // 3, replace=False)
            for i in idxs:
                px[i]      = rng2.uniform(0, GW)
                py[i]      = float(row_disturb)
                vy[i]      = rng2.uniform(0.008, 0.04)
                settled[i] = False
                br[i]      = rng2.uniform(80, 170)
            next_dist = f + rng2.integers(FPS * 8, FPS * 14)

        write_frame(w, buf)
    close(w, name)


# 60 – PRESSURE WAVE
# A single over-pressure front propagates from Y=0 (back) to Y=8 (front)
# — wide Gaussian brightness band that expands and fades as it travels,
# then bounces softly off the front edge and returns.
# Period ~20 s.  Like a pressure wave in a fluid-filled room.

def make_60_pressure_wave() -> None:
    name = "60_pressure_wave.webm"
    print(f"[60] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6000)

    TOTAL   = FPS * 70
    xs      = np.arange(GW, dtype=np.float32)
    ys      = np.arange(GH, dtype=np.float32)

    # Two damped waves travelling in opposite directions
    waves = [
        {"pos": 0.0, "vel": (GH - 1) / (FPS * 9), "amp": 160.0, "sig": 1.4},
        {"pos": float(GH - 1), "vel": -(GH - 1) / (FPS * 11), "amp": 100.0, "sig": 1.0},
    ]

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for wave in waves:
            wave["pos"] += wave["vel"]
            if wave["pos"] > GH - 1:
                wave["pos"] = GH - 1
                wave["vel"] *= -0.72
                wave["amp"] *= 0.82
            if wave["pos"] < 0:
                wave["pos"] = 0
                wave["vel"] *= -0.72
                wave["amp"] *= 0.82
            if wave["amp"] < 4:
                wave["pos"]  = 0.0
                wave["vel"]  = (GH - 1) / (FPS * 9)
                wave["amp"]  = rng2.uniform(120, 170)

            y_env = np.exp(-0.5 * ((ys - wave["pos"]) / wave["sig"]) ** 2)
            x_env = 1.0 + 0.05 * np.sin(xs * 0.031 + f * 0.021)
            canvas += np.outer(y_env, x_env) * wave["amp"]

        write_frame(w, np.clip(canvas, 0, 220))
    close(w, name)


# 61 – DEPTH WEED
# Each X column has a vertical "stem" that sways gently in an
# independently-phased sinusoidal current.  The stem is a dim
# continuous glow from Y=8 (root) up to a drifting bright tip.
# The weed colony breathes: all tips surge brighter on slow 18 s cycle.

def make_61_depth_weed() -> None:
    name = "61_depth_weed.webm"
    print(f"[61] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6100)

    TOTAL      = FPS * 70
    root_y     = GH - 1    # all roots at front (audience)
    tip_rest   = rng2.uniform(2.0, 5.5, GW).astype(np.float32)
    tip_amp    = rng2.uniform(0.5, 2.0, GW).astype(np.float32)
    tip_freq   = rng2.uniform(0.008, 0.022, GW).astype(np.float32)
    tip_phase  = rng2.uniform(0, 2 * math.pi, GW).astype(np.float32)
    stem_bright= rng2.uniform(30, 70, GW).astype(np.float32)
    tip_bright = rng2.uniform(100, 180, GW).astype(np.float32)
    BREATH     = FPS * 18

    ys = np.arange(GH, dtype=np.float32)

    for f in range(TOTAL):
        tip_y = tip_rest + tip_amp * np.sin(tip_freq * f + tip_phase)
        tip_y = np.clip(tip_y, 0.5, GH - 1.5)             # (GW,)

        breath = 0.7 + 0.3 * math.sin(2 * math.pi * f / BREATH)

        # Vectorised: ys (GH,1) broadcast against tip_y (1,GW)
        stem_centres = (tip_y[None, :] + root_y) / 2.0     # (1,GW)
        stem = stem_bright[None, :] * np.exp(
            -0.5 * ((ys[:, None] - stem_centres) / 2.0) ** 2)
        tips = tip_bright[None, :] * breath * np.exp(
            -0.5 * ((ys[:, None] - tip_y[None, :]) / 0.65) ** 2)
        canvas = np.maximum(stem, tips).astype(np.float32)
        write_frame(w, np.clip(canvas, 0, 200))
    close(w, name)


# 62 – UPWELLING
# Cold, dark deep water (back rows) periodically rises to the surface
# (front row) in slow columns of brightening light — like a deep-sea
# upwelling current bringing nutrients to the surface.
# Random column groups activate; bright front row pulse dies slowly.

def make_62_upwelling() -> None:
    name = "62_upwelling.webm"
    print(f"[62] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6200)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    TOTAL      = FPS * 70
    UPSURGE    = FPS * 7     # new upwelling pulse every ~7 s
    next_surge = 0
    # Brightness envelope per column: 0 = settled, >0 = rising
    col_phase  = np.zeros(GW, dtype=np.float32)

    for f in range(TOTAL):
        buf *= 0.91

        if f >= next_surge:
            # Activate a random band of ~30–80 contiguous columns
            cx         = rng2.integers(20, GW - 20)
            bw         = rng2.integers(20, 80)
            x0         = max(0, cx - bw // 2)
            x1         = min(GW, cx + bw // 2)
            col_phase[x0:x1] = rng2.uniform(0, 2 * math.pi, x1 - x0).astype(np.float32)
            next_surge = f + rng2.integers(FPS * 5, FPS * 10)

        # Advance phase; bright front row (Y=8) when phase sin > 0
        col_phase += 0.04
        surf_bright = np.maximum(0, np.sin(col_phase)) * rng2.uniform(100, 180, GW)
        buf[GH - 1, :] = np.maximum(buf[GH - 1, :], surf_bright.astype(np.float32))

        # Faint mid-water glow proportional to surface activity
        for y in range(GH - 2, -1, -1):
            fade = ((GH - 1 - y) / (GH - 1)) ** 1.8
            row_glow = surf_bright * (1.0 - fade) * 0.4
            buf[y, :] = np.maximum(buf[y, :], row_glow.astype(np.float32))

        write_frame(w, buf)
    close(w, name)


# 63 – PERISTALSIS
# Periodic contractions travel along Y like muscular peristalsis —
# a bright ring of illumination that squeezes forward
# (Y=0→8) then fades, followed by another.
# X is full-width but modulated by a slow ripple so it doesn't read flat.

def make_63_peristalsis() -> None:
    name = "63_peristalsis.webm"
    print(f"[63] {name} ...")
    w    = open_writer(name)

    TOTAL     = FPS * 70
    PERIOD    = FPS * 11    # one contraction every 11 s
    PEAK      = 200.0
    xs        = np.arange(GW, dtype=np.float32)
    ys        = np.arange(GH, dtype=np.float32)

    for f in range(TOTAL):
        phase = (f % PERIOD) / PERIOD    # 0→1 per cycle
        # Contraction front moves from Y=0 to Y=8
        centre = phase * (GH - 1)
        # Sharp Gaussian — narrow ring
        y_env  = np.exp(-0.5 * ((ys - centre) / 0.75) ** 2)
        # Trailing dim halo
        tail   = np.exp(-np.maximum(ys - centre, 0) * 1.2) * 0.35
        y_tot  = np.maximum(y_env, tail)
        # Organic X texture — low-amplitude spatially random ripple
        x_env  = 1.0 + 0.08 * np.sin(xs * 0.055 + f * 0.012) + \
                 0.05 * np.sin(xs * 0.025 - f * 0.008)
        canvas = np.outer(y_tot, x_env) * PEAK
        write_frame(w, np.clip(canvas, 0, PEAK))
    close(w, name)


# 64 – TIDAL BORE
# A single very bright thin wavefront (tidal bore) rushes from the back
# (Y=0) to the front (Y=8) in ~3 s, then a long slow turbulent aftermath
# gradually dims over ~14 s.  Repeats with a dark pause of ~10 s.
# The bore X profile is jagged/noisy rather than smooth.

def make_64_tidal_bore() -> None:
    name = "64_tidal_bore.webm"
    print(f"[64] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6400)

    TOTAL     = FPS * 75
    ys        = np.arange(GH, dtype=np.float32)
    xs        = np.arange(GW, dtype=np.float32)

    bore_speed  = (GH - 1) / (FPS * 2.8)     # rows/frame — crosses in ~2.8 s
    BORE_BRIGHT = 230.0
    TURB_DECAY  = 0.96

    bore_y    = -1.0
    turbulence = np.zeros((GH, GW), dtype=np.float32)
    next_bore  = 0

    for f in range(TOTAL):
        turbulence *= TURB_DECAY

        if f >= next_bore:
            bore_y += bore_speed
            if bore_y > GH - 1:
                bore_y    = -1.0
                next_bore = f + rng2.integers(FPS * 8, FPS * 14)
            else:
                # Bore: noisy bright row  + rows behind glowing hot
                noise = rng2.uniform(0.6, 1.0, GW).astype(np.float32)
                row   = int(np.clip(bore_y, 0, GH - 1))
                turbulence[row, :] = np.maximum(turbulence[row, :],
                                                noise * BORE_BRIGHT)
                # Deposit turbulent energy in rows behind the front
                for behind in range(1, row + 1):
                    decay = math.exp(-behind * 0.55)
                    turbulence[row - behind, :] = np.maximum(
                        turbulence[row - behind, :],
                        noise * BORE_BRIGHT * decay * 0.4)

        write_frame(w, turbulence)
    close(w, name)


# 65 – NEURAL PROPAGATION
# Y-axis signal propagation inspired by action potentials:
# A stimulus fires at a random column and travels both forward (toward Y=8)
# AND backward (toward Y=0) simultaneously — a depolarisation wave.
# After it passes, a brief refractory shadow (dim) lingers before the
# column recovers.  Multiple independent propagations overlap.

def make_65_neural_propagation() -> None:
    name = "65_neural_propagation.webm"
    print(f"[65] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6500)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    TOTAL      = FPS * 70
    SPAWN_INT  = FPS * 5
    pulses: list[dict] = []
    xs_a = np.arange(GW, dtype=np.float32)
    ys_a = np.arange(GH, dtype=np.float32)

    def new_pulse():
        return {
            "ox":    rng2.uniform(GW * 0.1, GW * 0.9),   # X centre of stimulus
            "oy":    rng2.uniform(0.5, GH - 1.5),         # Y origin
            "r":     0.0,
            "speed": rng2.uniform(0.07, 0.18),            # rows/frame
            "peak":  rng2.uniform(120, 220),
            "xsig":  rng2.uniform(15, 55),                # X spread of the column
        }

    for f in range(TOTAL):
        buf *= 0.88

        if f % SPAWN_INT == 0 and len(pulses) < 5:
            pulses.append(new_pulse())

        canvas = np.zeros((GH, GW), dtype=np.float32)
        alive  = []
        for p in pulses:
            p["r"] += p["speed"]
            x_env = np.exp(-0.5 * ((xs_a - p["ox"]) / p["xsig"]) ** 2)
            # Forward ring (toward Y=8)
            yf = p["oy"] + p["r"]
            rf = np.exp(-0.5 * ((ys_a - yf) / 0.6) ** 2)
            # Backward ring (toward Y=0)
            yb = p["oy"] - p["r"]
            rb = np.exp(-0.5 * ((ys_a - yb) / 0.6) ** 2)
            # Refractory zone between the two fronts (dim)
            refractory = np.exp(-0.5 * ((ys_a - p["oy"]) / (p["r"] * 0.5 + 0.5)) ** 2) * 0.3
            ring = (np.maximum(rf, rb) - refractory).clip(0)
            canvas += np.outer(ring, x_env) * p["peak"]
            if p["r"] < GH + 2:
                alive.append(p)
        pulses = alive

        np.maximum(buf, canvas, out=buf)
        write_frame(w, np.clip(buf, 0, 255))
    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# MORE NATURAL PATTERNS  (66 – 72)
# ══════════════════════════════════════════════════════════════════════════════

# 66 – SLIME MOLD
# Physarum-inspired network: particles oscillate along tube-like paths
# that reinforce themselves.  Nutrient nodes (fixed bright points) anchor
# the network.  Over time tubes converge to a sparse bright lattice.

def make_66_slime_mold() -> None:
    name = "66_slime_mold.webm"
    print(f"[66] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6600)
    trail = np.zeros((GH, GW), dtype=np.float32)

    N_AGENTS  = 120
    N_NODES   = 6
    # Nutrient nodes
    node_x = rng2.integers(40, GW - 40, N_NODES)
    node_y = rng2.integers(1, GH - 1, N_NODES)

    px     = rng2.uniform(0, GW, N_AGENTS)
    py     = rng2.uniform(0, GH - 1, N_AGENTS)
    angle  = rng2.uniform(0, 2 * math.pi, N_AGENTS)
    SENSOR = 4.0     # sensor look-ahead distance
    TURN   = 0.35    # turn rate toward chemo

    TOTAL = FPS * 75

    for f in range(TOTAL):
        trail *= 0.988

        for i in range(N_AGENTS):
            # Sense ahead, left, right
            sa  = angle[i]
            for da, idx in [(0, 0), (-TURN, 1), (TURN, 2)]:
                sx  = px[i] + SENSOR * math.cos(sa + da)
                sy  = py[i] + SENSOR * math.sin(sa + da) / 3.5
                isy = int(np.clip(sy, 0, GH - 1))
                isx = int(sx) % GW
                samples = [trail[isy, isx]]
                # Add attraction toward nutrient nodes
                for ni in range(N_NODES):
                    d = math.hypot(sx - node_x[ni], (sy - node_y[ni]) * 3.5)
                    samples.append(30.0 / max(d, 1.0))
                if idx == 0:
                    fwd = sum(samples)
                elif idx == 1:
                    lft = sum(samples)
                else:
                    rgt = sum(samples)

            if fwd > lft and fwd > rgt:
                pass
            elif lft > rgt:
                angle[i] -= TURN
            elif rgt > lft:
                angle[i] += TURN
            else:
                angle[i] += rng2.uniform(-TURN, TURN)

            angle[i] += rng2.uniform(-0.08, 0.08)
            px[i]     = (px[i] + math.cos(angle[i]) * 0.8) % GW
            py[i]     = float(np.clip(py[i] + math.sin(angle[i]) * 0.08, 0, GH - 1))

            iy = int(np.clip(py[i], 0, GH - 1))
            ix = int(px[i]) % GW
            trail[iy, ix] = min(trail[iy, ix] + 4.0, 180.0)

        # Nutrient nodes always glow
        for ni in range(N_NODES):
            trail[node_y[ni], node_x[ni]] = min(trail[node_y[ni], node_x[ni]] + 8, 220)

        write_frame(w, trail)
    close(w, name)


# 67 – SAND RIPPLE
# The classic aeolian sand ripple: a spatial sinusoid whose wavelength
# slowly drifts and whose amplitude breathes.  Two ripple systems with
# slightly different wavelengths interfere to create a slowly-evolving
# moiré that looks unmistakably like raked sand or a shallow seabed.

def make_67_sand_ripple() -> None:
    name = "67_sand_ripple.webm"
    print(f"[67] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6700)

    TOTAL = FPS * 70
    xs    = np.arange(GW, dtype=np.float32)
    ys    = np.arange(GH, dtype=np.float32)

    # Two ripple systems
    sys1  = {"lam": 35.0, "dlam": 0.004, "phase": 0.0, "spd": 0.015,
             "amp": 110.0, "y_shear": 0.18}
    sys2  = {"lam": 52.0, "dlam": -0.003, "phase": 1.8, "spd": -0.009,
             "amp": 60.0,  "y_shear": -0.12}

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for s in (sys1, sys2):
            s["lam"]   = max(15.0, s["lam"] + s["dlam"])
            s["phase"] += s["spd"]
            k          = 2 * math.pi / s["lam"]
            for yi, y in enumerate(ys):
                phase_y = s["phase"] + y * s["y_shear"]
                row     = (0.5 + 0.5 * np.sin(k * xs + phase_y)) * s["amp"]
                canvas[yi, :] += row

        # Soft Y envelope — brightest in mid-depth rows
        y_env = np.exp(-0.5 * ((ys - (GH - 1) / 2) / 2.8) ** 2)
        canvas = canvas * y_env[:, None]
        write_frame(w, np.clip(canvas, 0, 210))
    close(w, name)


# 68 – CILIA SWEEP
# Thousands of microscopic hair-like cilia beat in a coordinated
# metachronal wave — each X column beats slightly later than the one
# to its left.  Each column is a dim "hair" that flicks between
# bent (bright at front rows) and extended (bright at back rows).

def make_68_cilia_sweep() -> None:
    name = "68_cilia_sweep.webm"
    print(f"[68] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6800)

    TOTAL     = FPS * 70
    PERIOD    = FPS * 3         # 3 s per full beat cycle
    PHASE_LAG = 0.018           # rad/column — metachronal wave lag
    phases    = rng2.uniform(0, 2 * math.pi, GW).astype(np.float32)
    # Small per-column variation in frequency for natural feel
    freqs     = (2 * math.pi / PERIOD * (
        1 + rng2.uniform(-0.06, 0.06, GW))).astype(np.float32)
    ys        = np.arange(GH, dtype=np.float32)

    for f in range(TOTAL):
        col_phase = phases + freqs * f + PHASE_LAG * np.arange(GW, dtype=np.float32)
        # Beat position: 0 = fully bent (bright at Y=GH-1), 1 = extended (Y=0)
        beat       = 0.5 + 0.5 * np.sin(col_phase)   # (GW,)
        # Each column: Gaussian centred on beat*tip + (1-beat)*root
        tip_y = beat * (GH - 1) + (1 - beat) * 0.0
        bright = 80 + 120 * (0.5 + 0.5 * np.sin(col_phase - math.pi / 6))
        canvas = np.exp(-0.5 * ((ys[:, None] - tip_y[None, :]) / 0.7) ** 2) * bright[None, :]
        write_frame(w, np.clip(canvas.astype(np.float32), 0, 210))
    close(w, name)


# 69 – MAGNETOTAXIS
# Magnetic bacteria particles sense an invisible slowly-rotating field
# and align themselves to it.  Their collective orientation forms a drifting
# bright streak that slowly arcs across both X and Y.

def make_69_magnetotaxis() -> None:
    name = "69_magnetotaxis.webm"
    print(f"[69] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(6900)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    N      = 80
    px     = rng2.uniform(0, GW, N)
    py     = rng2.uniform(0, GH - 1, N)
    angle  = rng2.uniform(0, 2 * math.pi, N)
    br     = rng2.uniform(80, 170, N)
    TOTAL  = FPS * 70
    FIELD_PERIOD = FPS * 28   # field rotates 360° in 28 s

    for f in range(TOTAL):
        buf *= 0.93
        field_angle = 2 * math.pi * f / FIELD_PERIOD
        # X component governs horizontal drift; tiny Y component
        target_ax = field_angle
        target_ay = field_angle + math.pi / 4

        for i in range(N):
            # Turn toward field direction
            da = (target_ax - angle[i] + math.pi) % (2 * math.pi) - math.pi
            angle[i] += da * 0.04 + rng2.uniform(-0.06, 0.06)
            spd_x = math.cos(angle[i]) * 1.1
            spd_y = math.sin(angle[i]) * 0.09
            px[i] = (px[i] + spd_x) % GW
            py[i] = float(np.clip(py[i] + spd_y, 0, GH - 1))
            iy    = int(py[i])
            ix    = int(px[i]) % GW
            buf[iy, ix] = max(buf[iy, ix], br[i])

        write_frame(w, buf)
    close(w, name)


# 70 – SURFACE TENSION
# Water surface tension makes droplets form and split.
# Modelled as competing Gaussian blobs that merge when close
# and split when they exceed a size threshold.  Dark background.

def make_70_surface_tension() -> None:
    name = "70_surface_tension.webm"
    print(f"[70] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(7000)

    TOTAL = FPS * 70
    xs    = np.arange(GW, dtype=np.float32)
    ys    = np.arange(GH, dtype=np.float32)

    N    = 7
    bx   = rng2.uniform(30, GW - 30, N).astype(float)
    by_  = rng2.uniform(1.0, GH - 2.0, N)
    bvx  = rng2.uniform(-0.25, 0.25, N)
    bvy  = rng2.uniform(-0.015, 0.015, N)
    bsig = rng2.uniform(18, 42, N).astype(float)   # X sigma
    bamp = rng2.uniform(80, 150, N).astype(float)

    for f in range(TOTAL):
        canvas = np.zeros((GH, GW), dtype=np.float32)
        for i in range(N):
            bx[i]   = (bx[i] + bvx[i]) % GW
            by_[i]  = float(np.clip(by_[i] + bvy[i], 0.5, GH - 1.5))
            if by_[i] <= 0.5 or by_[i] >= GH - 1.5:
                bvy[i] *= -1

            x_env = np.exp(-0.5 * ((xs - bx[i]) / bsig[i]) ** 2)
            y_env = np.exp(-0.5 * ((ys - by_[i]) / 0.9) ** 2)
            canvas += np.outer(y_env, x_env) * bamp[i]

            # Blob drifts toward nearest neighbour (surface-tension cohesion)
            dists = [(math.hypot(bx[i] - bx[j], (by_[i] - by_[j]) * 7.88), j)
                     for j in range(N) if j != i]
            dists.sort()
            nb_d, nb_j = dists[0]
            if nb_d > 1.0:
                bvx[i] += (bx[nb_j] - bx[i]) * 0.0003
                bvy[i] += (by_[nb_j] - by_[i]) * 0.002

            # Split if sigma grows too large (repulsion at very close range)
            if nb_d < bsig[i] * 0.3:
                bvx[i] -= (bx[nb_j] - bx[i]) * 0.004
                bvy[i] -= (by_[nb_j] - by_[i]) * 0.02

            bvx[i] *= 0.97
            bvy[i] *= 0.97

        write_frame(w, np.clip(canvas, 0, 200))
    close(w, name)


# 71 – TURBULENCE BURST
# Kolmogorov-inspired: periodic energy injections at large scale cascade
# into smaller swirling eddies.  Implemented as a sum of random-phase
# sine waves at geometrically decreasing wavelengths whose amplitude
# peaks and decays after each injection event.

def make_71_turbulence_burst() -> None:
    name = "71_turbulence_burst.webm"
    print(f"[71] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(7100)

    TOTAL  = FPS * 70
    xs     = np.linspace(0, 2 * math.pi * 6, GW, dtype=np.float32)
    ys     = np.linspace(0, 2 * math.pi * 6 * GH / GW, GH, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    N_OCT   = 6     # octaves
    lambdas = [240 / (2 ** i) for i in range(N_OCT)]   # px wavelengths
    kxs     = [(2 * math.pi / l) for l in lambdas]

    # Random wave directions and phases per octave
    kx_arr = np.array([kxs[i] * math.cos(rng2.uniform(0, 2*math.pi))
                       for i in range(N_OCT)], dtype=np.float32)
    ky_arr = np.array([kxs[i] * math.sin(rng2.uniform(0, 2*math.pi))
                       for i in range(N_OCT)], dtype=np.float32)
    phi    = rng2.uniform(0, 2*math.pi, N_OCT).astype(np.float32)
    spd    = rng2.uniform(0.01, 0.06, N_OCT).astype(np.float32)

    amp_env     = np.zeros(N_OCT, dtype=np.float32)
    next_inject = 0
    INJECT_INT  = FPS * 9

    for f in range(TOTAL):
        if f >= next_inject:
            amp_env[0] = rng2.uniform(1.2, 2.0)
            next_inject = f + INJECT_INT

        # Cascade energy down octaves with decay
        for i in range(N_OCT - 1, 0, -1):
            amp_env[i] = amp_env[i] * 0.94 + amp_env[i-1] * 0.06
        amp_env *= 0.97

        canvas = np.zeros((GH, GW), dtype=np.float32)
        for i in range(N_OCT):
            amp_i = amp_env[i] / (2 ** i)      # Kolmogorov -5/3 approximate
            canvas += amp_i * np.sin(kx_arr[i] * XX + ky_arr[i] * YY + phi[i] + spd[i] * f)

        canvas = (canvas + N_OCT) / (2.0 * N_OCT) * 160.0
        write_frame(w, np.clip(canvas, 0, 200))
    close(w, name)


# 72 – SYMBIOSIS
# Two distinct populations of particles (A and B) coexist:
# A particles drift slowly rightward and emit brightness;
# B particles drift leftward and are attracted toward clusters of A.
# When B reaches A it briefly flares both populations before detaching.
# Dark background — intimate and biological.

def make_72_symbiosis() -> None:
    name = "72_symbiosis.webm"
    print(f"[72] {name} ...")
    w    = open_writer(name)
    rng2 = np.random.default_rng(7200)
    buf  = np.zeros((GH, GW), dtype=np.float32)

    NA  = 20    # population A (emitters)
    NB  = 15    # population B (seekers)

    ax  = rng2.uniform(GW * 0.1, GW * 0.9, NA)
    ay  = rng2.uniform(1, GH - 2, NA).astype(float)
    avx = rng2.uniform(0.15, 0.40, NA)
    avy = rng2.uniform(-0.012, 0.012, NA)
    a_br= rng2.uniform(100, 180, NA)

    bx  = rng2.uniform(GW * 0.1, GW * 0.9, NB)
    by_ = rng2.uniform(1, GH - 2, NB).astype(float)
    bvx = rng2.uniform(-0.30, -0.10, NB)
    bvy = rng2.uniform(-0.010, 0.010, NB)
    b_br= rng2.uniform(60, 130, NB)

    flare = np.zeros(NA, dtype=float)   # countdown for A flare

    TOTAL = FPS * 70

    for _ in range(TOTAL):
        buf *= 0.92

        # Move A
        ax = (ax + avx) % GW
        ay = np.clip(ay + avy, 0, GH - 1)

        # Move B, attracted to nearest A
        for j in range(NB):
            dists = [math.hypot(bx[j] - ax[i], (by_[j] - ay[i]) * 7.88)
                     for i in range(NA)]
            ni    = int(np.argmin(dists))
            nd    = dists[ni]
            if nd > 0.5:
                bvx[j] += (ax[ni] - bx[j]) * 0.003
                bvy[j] += (ay[ni] - by_[j]) * 0.018
            elif nd <= 0.5:
                # Contact — flare and detach
                flare[ni] = 18
                bx[j]  = rng2.uniform(0, GW)
                by_[j] = rng2.uniform(1, GH - 2)
                bvx[j] = rng2.uniform(-0.30, -0.10)
                bvy[j] = rng2.uniform(-0.010, 0.010)

            bvx[j] *= 0.92
            bvy[j] *= 0.92
            bx[j]   = (bx[j] + bvx[j]) % GW
            by_[j]  = float(np.clip(by_[j] + bvy[j], 0, GH - 1))
            iy = int(by_[j])
            ix = int(bx[j]) % GW
            buf[iy, ix] = max(buf[iy, ix], float(b_br[j]))

        # Draw A (with optional flare)
        for i in range(NA):
            iy  = int(ay[i])
            ix  = int(ax[i]) % GW
            val = a_br[i] * (1 + (flare[i] / 18) * 1.4)
            buf[iy, ix] = max(buf[iy, ix], min(float(val), 255.0))
            if flare[i] > 0:
                flare[i] -= 1

        write_frame(w, buf)
    close(w, name)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    print(f"Grid: {GW} × {GH} px  (preview render: {VW} × {VH})")
    print(f"Output -> {OUT_DIR}  (starting from video {start_from:02d})\n")

    steps = [
        make_01_scan_gol,              # 01  row scan → bounce → GoL explosion
        make_02_rain,                  # 02  per-column drops falling through Y
        make_03_wave_ripple,           # 03  3 interfering sine waves along X
        make_04_lightning,             # 04  stochastic horizontal discharge bolts
        make_05_meteor_shower,         # 05  diagonal streaks with comet tails
        make_06_breathing_noise,       # 06  12-layer harmonic noise + breath
        make_07_cellular_automata,     # 07  Rule-30 CA per row, fast
        make_08_rain_horizontal,       # 08  comet streaks along X axis
        make_09_slow_aurora,           # 09  slow drifting aurora bands
        make_10_deep_pulse,            # 10  front-to-back breathing pulse
        make_11_fireflies,             # 11  sparse single-pixel sparks
        make_12_slow_cellular_drift,   # 12  Rule-110 CA, 1 step/8 frames
        make_13_tide,                  # 13  slow horizontal luminous tide
        make_14_stellar_parallax,      # 14  3-layer drifting stars, parallax depth
        make_15_heat_diffusion,        # 15  heat-equation blooms from seeds
        make_16_column_choir,          # 16  close harmonics per column, moiré
        make_17_depth_ripple,          # 17  expanding rings, physical Y scale
        make_18_wanderer,              # 18  one pixel slow drunk-walk, long trail
        make_19_pixel_migration,       # 19  ~20 pixels drifting rightward
        make_20_cold_sparks,           # 20  1-frame single-pixel flashes
        make_21_pixel_telegraph,       # 21  dot/dash morse-like pixel signals
        make_22_random_walk_swarm,     # 22  12 pixels independent random walks
        make_23_pixel_heartbeat,       # 23  pixels pulse in cardiac lub-dub rhythm
        make_24_pixel_orbits,          # 24  pixels orbit fixed centre points
        make_25_trailing_cursor,       # 25  single pixel scans row by row
        make_26_sparse_static,         # 26  2–6 dim pixels per frame, fast fade
        make_27_gravity_well,          # 27  pixels attracted to 2 drifting wells
        make_28_pixel_erosion,         # 28  full row erodes pixel by pixel
        make_29_pixel_repulsion,       # 29  pairs of pixels repel each other
        make_30_quantum_dots,          # 30  near-invisible probabilistic shimmer
        make_31_pixel_drift_field,     # 31  40 pixels follow rotating vector field
        make_32_pixel_constellations,  # 32  star pattern with slow pulsing edges
        # ── Y-axis / depth movement series ────────────────────────────────
        make_33_depth_scan,            # 33  full-width row sweeps back↔front
        make_34_depth_rain,            # 34  drops travel front→back along Y
        make_35_depth_cascade,         # 35  rows activate back→front sequentially
        make_36_depth_curtain,         # 36  brightness curtain rolls front↔back
        make_37_y_stripes_drift,       # 37  each row breathes at its own rate
        make_38_depth_meteor,          # 38  bright row-bands streak along Y axis
        make_39_row_sequencer,         # 39  random rhythm of row activations
        make_40_depth_wander,          # 40  3 pixels drift mostly in Y, rare X
        make_41_front_glow_pulse,      # 41  pulse originates front row, travels back
        make_42_y_axis_ripple_chain,   # 42  ripples spawn at front, expand toward back
        # ── Natural patterns: waves / flows / swarms / particles ───────────
        make_43_murmuration,           # 43  boids flocking (cohesion + separation + alignment)
        make_44_kelp_sway,             # 44  per-column kelp tips oscillate in Y with current
        make_45_caustics,              # 45  water caustics: interfering plane waves
        make_46_smoke_plume,           # 46  particles drift upward from emitter with turbulence
        make_47_lava_lamp,             # 47  gaussian blobs oscillate slowly in Y
        make_48_bioluminescence,       # 48  sweeping trigger wave lights each column
        make_49_shoal_scatter,         # 49  fish school + predator scatter/regroup
        make_50_thermal_convection,    # 50  hot/cool column convection cells
        make_51_chladni,               # 51  chladni standing-wave interference figures
        make_52_river_flow,            # 52  particles in curl flow field with eddies
        make_53_mycelium,              # 53  branching fungal growth from seed points
        make_54_pheromone_trail,       # 54  ant colony pheromone trail between food sources
        make_55_coral_pulse,           # 55  pulsing radial rings from fixed polyp positions
        make_56_deep_current,          # 56  laminar/poiseuille shear flow by depth row
        make_57_phosphene,             # 57  bright concentric rings expand from random origins
        # ── Organic Y-axis / depth movements ──────────────────────────────
        make_58_breath_of_deep,        # 58  sigmoid inhale/exhale front-to-back
        make_59_silt_settling,         # 59  particles born mid-depth, sink to front
        make_60_pressure_wave,         # 60  damped pressure front bouncing along Y
        make_61_depth_weed,            # 61  per-column kelp roots at front, tips sway in Y
        make_62_upwelling,             # 62  deep cold columns surge bright to front row
        make_63_peristalsis,           # 63  muscular contraction ring travels Y=0→8
        make_64_tidal_bore,            # 64  fast bright front + long turbulent aftermath
        make_65_neural_propagation,    # 65  bidirectional action-potential wave along Y
        # ── More natural patterns ──────────────────────────────────────────
        make_66_slime_mold,            # 66  physarum agent network converging to bright lattice
        make_67_sand_ripple,           # 67  two interfering aeolian ripple systems
        make_68_cilia_sweep,           # 68  metachronal cilia beat wave along X
        make_69_magnetotaxis,          # 69  magnetic bacteria aligning to slow-rotating field
        make_70_surface_tension,       # 70  droplet blobs merge/split via surface tension
        make_71_turbulence_burst,      # 71  kolmogorov energy cascade octaves
        make_72_symbiosis,             # 72  two particle populations attract and flare on contact
    ]
    for idx, fn in enumerate(steps, start=1):
        if idx >= start_from:
            fn()

    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".webm"))
    print("\n── Output files ────────────────────────────────")
    for f in files:
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1_048_576
        print(f"  {f:45s}  {size:5.1f} MB")
    print("Done.")
