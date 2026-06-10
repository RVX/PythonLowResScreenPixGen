"""
MONOLITH – B&W Generative Video
Art Basel 2026 / Julian Charrière

Animation phases:
  1. Scan line    – horizontal line sweeps top → bottom (left-to-right cursor)
  2. Bounce       – line bounces back and forth
  3. Converge     – line eases to vertical centre
  4. Circle       – particles arc outward from the line to form a ring
  5. Game of Life – ring + interior seeds explode via Conway's rules

Requirements:  pip install numpy opencv-python
Output:        monolith_generative.webm  (WebM / VP8, configurable resolution @ 30 fps)

grandMA3 Bitmap notes:
  - Set W × H to match your fixture grid (default canvas is 64 × 64)
  - Max supported: 1920 × 1080 @ 30 fps (DMX output ceiling)
  - Import via Media Pool → Video Pool, then assign in Bitmap Configuration
"""

import cv2
import math
import subprocess
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
# Set W × H to your fixture grid dimensions for direct grandMA3 Bitmap mapping.
# Examples: 64×64 (default canvas), 100×50, 192×108 (1920/10 × 1080/10), 1920×1080 (max)
W, H = 1920, 1080
FPS  = 30        # 30 Hz recommended (grandMA3 DMX output ceiling)
OUT  = "monolith_generative.webm"

# Phase lengths (frames)
SCAN_N     = 180    # 6 s  – scan line top → bottom
BOUNCE_N   = 140    # frames per half-sweep
BOUNCES    = 8     # half-sweeps (= 4 full cycles)
CONVERGE_N = 120    # 4 s  – ease to centre
CIRCLE_N   = 120   # 4 s  – morph to ring
FLASH_N    = 25    # ~0.8 s – explosion flash
GOL_N      = 600   # 20 s – Game of Life

RADIUS   = min(W, H) // 4   # ring radius in pixels (270 px for 1080p)
GOL_CELL = 5                 # pixels per GoL cell → 384 × 216 grid

# ── Video writer (FFmpeg subprocess – avoids VP8 YUV chroma artifacts) ──────────────
# Piping raw grayscale to ffmpeg keeps Cb=Cr=128 exactly — no greenish tint
# in fade tails that OpenCV's BGR→YUV→VP8 path would produce.
cmd = [
    "ffmpeg", "-y",
    "-loglevel", "error",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-pix_fmt", "gray",
    "-s", f"{W}x{H}",
    "-r", str(FPS),
    "-i", "pipe:0",
    "-c:v", "libvpx",
    "-crf", "10", "-b:v", "0",
    "-auto-alt-ref", "0",
    "-pix_fmt", "yuv420p",
    OUT,
]
_proc = subprocess.Popen(
    cmd, stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
)


def write(canvas: np.ndarray) -> None:
    """Clip, convert to uint8 and pipe one grayscale frame to ffmpeg."""
    gray = np.clip(canvas, 0, 255).astype(np.uint8)
    _proc.stdin.write(gray.tobytes())


# ── Easing functions ──────────────────────────────────────────────────────────
def ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def ease_out3(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


# ── Glow line ─────────────────────────────────────────────────────────────────
def glow_h_line(canvas: np.ndarray, y: int,
                peak: float = 255.0, sigma: float = 5.0) -> None:
    """Overlay a soft-glowing horizontal band at row y (in-place max)."""
    rows = np.arange(H, dtype=np.float32)
    col  = peak * np.exp(-0.5 * ((rows - y) / sigma) ** 2)
    np.maximum(canvas, col[:, None], out=canvas)


# ── Working canvas ────────────────────────────────────────────────────────────
canvas = np.zeros((H, W), dtype=np.float32)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 – Scan line: top → bottom, cursor sweeps left → right
# ═════════════════════════════════════════════════════════════════════════════
print("Phase 1 / 5 – scan line …")

for i in range(SCAN_N):
    canvas *= 0.93
    p = i / (SCAN_N - 1)
    y = int(p * (H - 1))
    glow_h_line(canvas, y)

    # Bright leading cursor that also sweeps left → right
    x = int(p * (W - 1))
    canvas[y, x] = min(255.0, canvas[y, x] + 140.0)

    write(canvas)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 – Bounce (back and forth)
# ═════════════════════════════════════════════════════════════════════════════
print("Phase 2 / 5 – bounce …")

line_y = H - 1
for b in range(BOUNCES):
    end_y   = 0 if b % 2 == 0 else H - 1
    start_y = line_y
    for i in range(BOUNCE_N):
        canvas *= 0.93
        t      = ease_in_out(i / (BOUNCE_N - 1))
        line_y = int(start_y + (end_y - start_y) * t)
        glow_h_line(canvas, line_y)
        write(canvas)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 – Converge to vertical centre
# ═════════════════════════════════════════════════════════════════════════════
print("Phase 3 / 5 – converge …")

center_y = H // 2
start_y  = line_y
for i in range(CONVERGE_N):
    canvas *= 0.93
    t      = ease_out3(i / (CONVERGE_N - 1))
    line_y = int(start_y + (center_y - start_y) * t)
    glow_h_line(canvas, line_y)
    write(canvas)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 – Line → Circle
# ═════════════════════════════════════════════════════════════════════════════
print("Phase 4 / 5 – circle …")

cx, cy  = W // 2, H // 2
N_DOTS  = 720

angles  = np.linspace(0, 2 * math.pi, N_DOTS, endpoint=False)
end_pts = np.column_stack([
    cx + RADIUS * np.cos(angles),
    cy + RADIUS * np.sin(angles),
]).astype(np.float32)

start_pts = np.column_stack([
    np.linspace(0, W - 1, N_DOTS),
    np.full(N_DOTS, center_y, dtype=np.float32),
])

layer = np.zeros((H, W), dtype=np.uint8)   # reusable scratch layer

for i in range(CIRCLE_N):
    canvas *= 0.93
    t   = ease_out3(i / (CIRCLE_N - 1))
    pts = (start_pts + (end_pts - start_pts) * t).astype(np.int32)

    layer[:] = 0
    valid = ((pts[:, 0] >= 0) & (pts[:, 0] < W) &
             (pts[:, 1] >= 0) & (pts[:, 1] < H))
    for p in pts[valid]:
        cv2.circle(layer, (int(p[0]), int(p[1])), 2, 255, -1)

    blurred = cv2.GaussianBlur(layer.astype(np.float32), (11, 11), 4)
    np.maximum(canvas, blurred, out=canvas)
    write(canvas)

# Explosion flash: white burst that decays to black
print("Phase 4 → 5 – explosion flash …")
for i in range(FLASH_N):
    t     = i / (FLASH_N - 1)
    alpha = (1.0 - t) ** 2          # quadratic decay
    write(np.full((H, W), 255.0 * alpha, dtype=np.float32))

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5 – Conway's Game of Life explosion
# ═════════════════════════════════════════════════════════════════════════════
print("Phase 5 / 5 – Game of Life …")

GW  = W // GOL_CELL   # 384
GH  = H // GOL_CELL   # 216
gcx = GW // 2
gcy = GH // 2
gr  = RADIUS // GOL_CELL    # ~54 cells

gol = np.zeros((GH, GW), dtype=bool)

# 1. Circle ring seed
theta = np.linspace(0, 2 * math.pi, 5000, endpoint=False)
xs    = np.clip((gcx + gr * np.cos(theta)).astype(int), 0, GW - 1)
ys    = np.clip((gcy + gr * np.sin(theta)).astype(int), 0, GH - 1)
gol[ys, xs] = True

# 2. Interior random seed (~25 % density inside ring)
rng       = np.random.default_rng(2026)
yy, xx    = np.mgrid[0:GH, 0:GW]
dist_grid = np.hypot(xx - gcx, yy - gcy)
interior  = dist_grid < gr * 0.82
gol[interior & (rng.random((GH, GW)) < 0.25)] = True


def gol_step(g: np.ndarray) -> np.ndarray:
    """One Conway's Game of Life generation, toroidal boundary."""
    gi = g.astype(np.uint8)
    # Sum of 8 neighbours (max = 8, fits in uint8)
    n = (np.roll(np.roll(gi, -1, 0), -1, 1) +   # ↘ lower-right
         np.roll(gi, -1, 0)                   +   # ↓ below
         np.roll(np.roll(gi, -1, 0),  1, 1)  +   # ↙ lower-left
         np.roll(gi,  1, 1)                   +   # ← left
         np.roll(np.roll(gi,  1, 0), -1, 1)  +   # ↗ upper-right
         np.roll(gi,  1, 0)                   +   # ↑ above
         np.roll(np.roll(gi,  1, 0),  1, 1)  +   # ↖ upper-left
         np.roll(gi, -1, 1))                       # → right
    # Conway rules: birth=3, survive=2 or 3
    return (n == 3) | (g & (n == 2))


ones    = np.ones((GOL_CELL, GOL_CELL), dtype=np.uint8)
gol_buf = np.zeros((H, W), dtype=np.float32)

for i in range(GOL_N):
    if i % 100 == 0:
        print(f"  frame {i:4d} / {GOL_N} …")

    gol_buf *= 0.97                                        # decay trail
    frame    = np.kron(gol.astype(np.uint8), ones) * 255  # upscale cells
    np.maximum(gol_buf, frame.astype(np.float32), out=gol_buf)
    write(gol_buf)
    gol = gol_step(gol)

# ── Finalise ──────────────────────────────────────────────────────────────────
_proc.stdin.close()
_stderr = _proc.communicate()[1]
if _proc.returncode != 0 and _stderr:
    print(f"[ffmpeg] {_stderr.decode(errors='replace').strip()}")

total = SCAN_N + BOUNCES * BOUNCE_N + CONVERGE_N + CIRCLE_N + FLASH_N + GOL_N
print(f"\nDone  →  {OUT}")
print(f"Frames :  {total}  ({total / FPS:.1f} s at {FPS} fps)")
