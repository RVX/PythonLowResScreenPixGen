#!/usr/bin/env python3
"""
monolith_osc_spatial.py
───────────────────────
Real-time OSC → 480 × 9 LED matrix  +  live OpenCV preview
Julian Charrière / Monolith ceiling  –  Art Basel 2026

Listens for IEM MultiEncoder spatial data (azimuth / elevation per source)
on UDP port 9010 and renders a live pixel-art preview window — same look as
the ceiling video renders.  Each source is a labelled Gaussian blob moving
in real time as OSC messages arrive.

OSC addresses (IEM MultiEncoder, zero-based index N):
    /MultiEncoder/azimuthN    <float>  degrees [-180 .. +180]
    /MultiEncoder/elevationN  <float>  degrees [  -90 ..  +90]

Coordinate mapping (IEM MultiEncoder convention):
    Azimuth convention:
      0deg   = front (audience edge)
      90deg  = left
      -90deg = right
      180deg = back (wall)
    Elevation is stored but NOT used for position.

    X = round((1 - sin(az_rad)) / 2 * (GW-1))   ->  left(0) .. right(479)
    Y = round((1 - cos(az_rad)) / 2 * (GH-1))   ->  front(0) .. back(8)

    Corner check:
      az=  0  front  -> X=240  Y=0
      az= 90  left   -> X=0    Y=4
      az=-90  right  -> X=479  Y=4
      az=180  back   -> X=240  Y=8

Controls (preview window):
    Q / Esc  - quit
    R        - reset / clear all sources
    V        - cycle viz mode  (dot / blob-XS / blob-S / blob-M / blob-L)
    F        - enter source filter  (e.g. 0-10,44  or  *  for all)
    Enter    - confirm filter
    Esc      - cancel filter input

Dependencies:
    pip install python-osc numpy opencv-python

Usage:
    python monolith_osc_spatial.py
    python monolith_osc_spatial.py --ip 0.0.0.0 --port 9010
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import cv2
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

# ─── FFmpeg video writer (identical to monolith_ceiling_videos.py) ───────────

class FfmpegWriter:
    """Pipes raw 480×9 grayscale frames to ffmpeg → VP8 WebM."""
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
        self.path = path

    def write(self, gray: np.ndarray) -> None:
        self._proc.stdin.write(gray.tobytes())

    def release(self) -> None:
        self._proc.stdin.close()
        _, stderr = self._proc.communicate()
        if self._proc.returncode != 0 and stderr:
            print(f"[rec]  ffmpeg: {stderr.decode(errors='replace').strip()}")

    def isOpened(self) -> bool:
        return self._proc.poll() is None


# ─── Grid constants (must match monolith_ceiling_videos.py) ─────────────────

GW: int = 480   # columns  (azimuth axis)
GH: int = 9     # rows     (elevation / depth axis)

# ─── Preview window ──────────────────────────────────────────────────────────
# Asymmetric upscale: X × 2, Y × 20  ->  960 × 180 px pixel-art grid
# + 48 px control panel at the bottom

SCALE_X: int = 2
SCALE_Y: int = 20
DISP_W:  int = GW * SCALE_X   # 960
DISP_H:  int = GH * SCALE_Y   # 180
CTRL_H:  int = 110            # control panel height below grid
WIN_H:   int = DISP_H + CTRL_H

# ─── Recorded video output resolution ───────────────────────────────────────
# Matches monolith_ceiling_videos.py: 4× upscale → 1920 × 36 px WebM
PREVIEW_SCALE: int = 4
VW: int = GW * PREVIEW_SCALE   # 1920
VH: int = GH * PREVIEW_SCALE   #   36
FPS: int = 30

# ─── Default network config ──────────────────────────────────────────────────

DEFAULT_IP:   str = "192.168.0.1"
DEFAULT_PORT: int = 9010

# ─── Dispatch rate-limit ─────────────────────────────────────────────────────

DISPATCH_HZ:  float = 40.0

# Position smoothing (exponential lerp applied every display frame ~30 fps).
# 0.0 = frozen, 1.0 = instant snap.  0.15 gives ~0.3 s settling time.
SMOOTH_ALPHA: float = 0.15

# Motion trails (T key)
TRAIL_LEN: int = 25   # positions kept per source

# Polar panner overview (P key)
PANNER_SIZE: int = 420   # window size in pixels

# ─── Shared font constants ───────────────────────────────────────────────────
_SF  = cv2.FONT_HERSHEY_SIMPLEX    # clean proportional font
_SFB = cv2.FONT_HERSHEY_DUPLEX     # bolder variant
_SFT = cv2.FONT_HERSHEY_TRIPLEX    # thick/title weight
_LW  = cv2.LINE_AA

# ─── Source state ────────────────────────────────────────────────────────────

@dataclass
class SourceState:
    """Holds the latest spatial coordinates for one MultiEncoder source."""
    azimuth:   float = 0.0
    elevation: float = 0.0
    x:         int   = GW // 2      # target x (OSC-derived, integer grid col)
    y:         int   = GH // 2      # target y (OSC-derived, integer grid row)
    sx:        float = float(GW // 2)  # smoothed display x (float for sub-pixel lerp)
    sy:        float = float(GH // 2)  # smoothed display y
    gain_db:   float = 0.0             # /MultiEncoder/gainN (dB); 0 = full, -80 = off
    last_dispatch: float = field(default=0.0, repr=False)

# Thread-safe registry
_lock:    threading.Lock          = threading.Lock()
_sources: Dict[int, SourceState] = {}

def _get_source(idx: int) -> SourceState:
    with _lock:
        if idx not in _sources:
            _sources[idx] = SourceState()
        return _sources[idx]

def _snapshot() -> Dict[int, SourceState]:
    """Return a shallow copy for safe reading from the main/display thread."""
    with _lock:
        return {k: SourceState(v.azimuth, v.elevation, v.x, v.y, v.sx, v.sy, v.gain_db)
                for k, v in _sources.items()}


# ─── Viz mode ────────────────────────────────────────────────────────────────

# Mode index cycles with V.  Each entry: (label, sigma_x, sigma_y)
# sigma_x/y == 0  means single-pixel dot mode.
_VIZ_MODES = [
    ("dot",     0.0,  0.0),
    ("blob-XS", 4.0,  0.4),
    ("blob-S",  8.0,  0.6),
    ("blob-M", 14.0,  0.9),
    ("blob-L", 30.0,  1.6),
]
_viz_idx: int = 0   # default: dot

# ─── Source filter ───────────────────────────────────────────────────────────
# None = show all.  A set of ints = only show those source indices.
_filter_set:   set[int] | None = None
_filter_input: str             = ""    # text being typed
_filter_mode:  bool            = False  # True while F-input is active

# Feature toggles (main thread only)
_trails_on:  bool = False   # T key
_frozen:     bool = False   # Space key
_panner_on:  bool = False   # P key
_recording:  bool = False   # E key
_rec_writer: "FfmpegWriter | None" = None
_rec_frames: int  = 0

# Trail buffers: 0-indexed src idx -> deque of (sx, sy)  (main thread only, no lock)
_trails: Dict[int, deque] = {}

# Source name map: 0-indexed OSC idx -> display label  (loaded from CSV)
_source_names: Dict[int, str] = {}


def _parse_filter(text: str) -> set[int] | None:
    """
    Parse a filter string into a set of source indices.
    Input is 1-indexed (matching MultiEncoder UI labels) -> stored 0-indexed internally.
      "*" or ""  -> None (show all)
      "1-10,44"  -> OSC indices {0,...,9,43}
      "5"        -> OSC index {4}
    Returns None on parse error (keep previous filter).
    """
    text = text.strip()
    if text in ("", "*", "all"):
        return None
    result: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                result.update(range(int(lo) - 1, int(hi)))
            except ValueError:
                return None
        else:
            try:
                result.add(int(part) - 1)
            except ValueError:
                return None
    return result


# ─── Coordinate mapping ──────────────────────────────────────────────────────

def az_to_xy(azimuth: float) -> tuple[int, int]:
    """
    Map IEM MultiEncoder azimuth to (X, Y) on the 480x9 ceiling grid.

    Azimuth convention (degrees):
      0   = front (audience edge)    -> Y=0  X=240
      90  = left                     -> Y=4  X=0
     -90  = right                    -> Y=4  X=479
      180 = back (wall)              -> Y=8  X=240

    Formula (circular -> rectangular projection):
      X = (1 - sin(az)) / 2 * (GW-1)
      Y = (1 - cos(az)) / 2 * (GH-1)
    """
    rad = math.radians(azimuth)
    s   = math.sin(rad)
    c   = math.cos(rad)
    # snap floating-point near-zero artifacts (e.g. sin(pi) ~= 1.2e-16)
    if abs(s) < 1e-10: s = 0.0
    if abs(c) < 1e-10: c = 0.0
    x   = round((1.0 - s) / 2.0 * (GW - 1))
    y   = round((1.0 - c) / 2.0 * (GH - 1))
    return max(0, min(GW - 1, x)), max(0, min(GH - 1, y))

# ─── Source name loader ────────────────────────────────────────────────────────────────

def load_names(path: str) -> None:
    """
    Load source name labels from a CSV file (optional).
    Format: one row per source -> <1-indexed-number>,<label>
    Example: 1,ViolinI   2,Viola   3,Cello
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    try:
                        _source_names[int(row[0].strip()) - 1] = row[1].strip()
                    except ValueError:
                        pass
        print(f"[names]  loaded {len(_source_names)} labels from '{path}'")
    except FileNotFoundError:
        pass   # optional file – no error if absent
    except Exception as e:
        print(f"[names]  warning: {e}")

# ─── Hardware dispatch (stub – wire up sACN / Art-Net here later) ────────────

def dispatch(source_idx: int, x: int, y: int, state: SourceState) -> None:
    """
    Called at most DISPATCH_HZ times per second per source.
    Visual preview is handled by the main thread; add sACN here when ready.

    ── sACN / grandMA3 (add when ready) ────────────────────────────────────
    Build a flat DMX universe from render_frame() and send via sacn:

        frame = render_frame(_snapshot())
        gray  = np.clip(frame, 0, 255).astype(np.uint8).flatten()
        for u in range(math.ceil(len(gray) / 512)):
            sender[u + 1].dmx_data = tuple(gray[u*512 : (u+1)*512])
    ──────────────────────────────────────────────────────────────────────
    """
    pass   # no console spam – display thread shows everything visually


# ─── Frame renderer ──────────────────────────────────────────────────────────

_PEAK: float = 220.0

def render_frame(
    sources: Dict[int, SourceState],
    filter_set: set[int] | None,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    """Return float32 (GH x GW) canvas respecting current viz mode and filter."""
    canvas = np.zeros((GH, GW), dtype=np.float32)
    if not sources:
        return canvas
    ys = np.arange(GH, dtype=np.float32)
    xs = np.arange(GW, dtype=np.float32)
    for idx, src in sources.items():
        if filter_set is not None and idx not in filter_set:
            continue
        # gain-modulated peak: 0 dB = full, negative dB = dim, <= -80 dB = off
        peak = _PEAK * (min(1.0, 10 ** (src.gain_db / 20.0)) if src.gain_db > -80.0 else 0.0)
        if sigma_x == 0.0:   # dot mode – single pixel at smoothed position
            px = max(0, min(GW - 1, round(src.sx)))
            py = max(0, min(GH - 1, round(src.sy)))
            canvas[py, px] = peak
        else:
            gx = np.exp(-0.5 * ((xs - src.sx) / sigma_x) ** 2)
            gy = np.exp(-0.5 * ((ys - src.sy) / sigma_y) ** 2)
            canvas += peak * np.outer(gy, gx)
    return np.clip(canvas, 0.0, 255.0)


def build_display(
    sources: Dict[int, SourceState],
    filter_set: set[int] | None,
    viz_idx: int,
    filter_mode: bool,
    filter_input: str,
    trails: "Dict[int, deque] | None" = None,
    frozen: bool = False,
) -> np.ndarray:
    """
    Render grid canvas + control panel.
    """
    viz_label, sigma_x, sigma_y = _VIZ_MODES[viz_idx]

    # ── grid canvas (GH × GW → DISP_W × DISP_H) ─────────────────────────
    gray_small = np.clip(
        render_frame(sources, filter_set, sigma_x, sigma_y), 0, 255
    ).astype(np.uint8)
    big = cv2.resize(gray_small, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
    grid = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    # ── grid lines ───────────────────────────────────────────────────────
    for row in range(1, GH):
        py = row * SCALE_Y
        cv2.line(grid, (0, py), (DISP_W, py), (22, 22, 28), 1)

    # ── motion trails ────────────────────────────────────────────────────
    if trails:
        for idx, trail in trails.items():
            if filter_set is not None and idx not in filter_set:
                continue
            pts = list(trail)
            n   = len(pts)
            if n < 2:
                continue
            for i, (tsx, tsy) in enumerate(pts):
                alpha = (i + 1) / n   # 0 = oldest (dark), 1 = newest (bright)
                tcx   = int(tsx * SCALE_X + SCALE_X // 2)
                tcy   = int(tsy * SCALE_Y + SCALE_Y // 2)
                b     = int(200 * alpha)
                cv2.circle(grid, (tcx, tcy), 2, (0, b, int(b * 1.1)), -1)

    # ── per-source overlay (marker + label) ──────────────────────────────
    for idx, src in sources.items():
        visible = filter_set is None or idx in filter_set
        cx = int(src.sx * SCALE_X + SCALE_X // 2)
        cy = int(src.sy * SCALE_Y + SCALE_Y // 2)
        if visible:
            label = _source_names.get(idx, str(idx + 1))
            (tw, th), _ = cv2.getTextSize(label, _SF, 0.42, 1)
            radius = max(9, tw // 2 + 5)
            # outer glow ring
            cv2.circle(grid, (cx, cy), radius + 3, (0, 60, 80), -1)
            cv2.circle(grid, (cx, cy), radius,     (0, 200, 255), -1)
            cv2.circle(grid, (cx, cy), radius + 1, (0, 120, 160), 1)
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(grid, label, (tx, ty), _SF, 0.42, (0, 0, 0), 1, _LW)
        else:
            cv2.circle(grid, (cx, cy), 4, (35, 35, 35), -1)
            cv2.circle(grid, (cx, cy), 5, (55, 55, 55), 1)

    # ── frozen overlay ────────────────────────────────────────────────────
    if frozen:
        overlay = grid.copy()
        cv2.rectangle(overlay, (0, 0), (DISP_W, DISP_H), (0, 0, 40), -1)
        cv2.addWeighted(overlay, 0.4, grid, 0.6, 0, grid)
        (tw, th), _ = cv2.getTextSize("FROZEN", _SFT, 1.1, 2)
        cv2.putText(grid, "FROZEN", (DISP_W // 2 - tw // 2, DISP_H // 2 + th // 2),
                    _SFT, 1.1, (60, 100, 255), 2, _LW)

    # ── REC badge (top-right) ────────────────────────────────────────────
    if _recording:
        secs = _rec_frames // 30
        rec_txt = f"\u25cf REC  {secs//60:02d}:{secs%60:02d}"
        (tw, th), _ = cv2.getTextSize(rec_txt, _SF, 0.55, 1)
        cv2.rectangle(grid, (DISP_W - tw - 18, 4), (DISP_W - 4, th + 12), (0, 0, 0), -1)
        cv2.putText(grid, rec_txt, (DISP_W - tw - 10, th + 7), _SF, 0.55, (0, 50, 240), 1, _LW)
    # ════════════════════════════════════════════════════════════════════
    # Control panel
    # ════════════════════════════════════════════════════════════════════
    panel = np.zeros((CTRL_H, DISP_W, 3), dtype=np.uint8)
    panel[:] = (14, 14, 18)   # near-black blue-grey
    # top separator
    cv2.line(panel, (0, 0), (DISP_W, 0), (0, 160, 200), 1)

    def _pill(img, x, y, text, active=False, col_act=(0,220,120), col_bg=(30,30,36)):
        """Draw a rounded-rect pill button."""
        (tw, th), _ = cv2.getTextSize(text, _SF, 0.42, 1)
        pad = 6
        x1, y1 = x - pad, y - th - pad
        x2, y2 = x + tw + pad, y + pad
        bg = col_act if active else col_bg
        fg = (0, 0, 0) if active else (90, 90, 100)
        cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 70), 1)
        cv2.putText(img, text, (x, y), _SF, 0.42, fg, 1, _LW)
        return x2 + 8   # next x

    # ── Row 1  (y=28): viz mode pills + status pills ──────────────────────
    ROW1 = 28
    (lw, _), _ = cv2.getTextSize("VIZ", _SFB, 0.44, 1)
    cv2.putText(panel, "VIZ", (10, ROW1), _SFB, 0.44, (0, 160, 200), 1, _LW)
    x_off = 10 + lw + 12
    for i, (lbl, _, _) in enumerate(_VIZ_MODES):
        x_off = _pill(panel, x_off, ROW1, lbl, active=(i == viz_idx))

    # status pills on same row, right-side
    status_x = DISP_W - 10
    if frozen:
        (tw, _), _ = cv2.getTextSize("FROZEN", _SF, 0.42, 1)
        status_x -= tw + 20
        _pill(panel, status_x, ROW1, "FROZEN", active=True, col_act=(40, 80, 220))
    if trails is not None:
        (tw, _), _ = cv2.getTextSize("TRAIL", _SF, 0.42, 1)
        status_x -= tw + 20
        _pill(panel, status_x, ROW1, "TRAIL", active=True, col_act=(0, 180, 100))
    if _recording:
        secs = _rec_frames // 30
        rec_lbl = f"REC {secs//60:02d}:{secs%60:02d}"
        (tw, _), _ = cv2.getTextSize(rec_lbl, _SF, 0.42, 1)
        status_x -= tw + 20
        _pill(panel, status_x, ROW1, rec_lbl, active=True, col_act=(20, 20, 200))

    # thin rule
    cv2.line(panel, (0, 38), (DISP_W, 38), (30, 30, 36), 1)

    # ── Row 2 (y=60): filter bar + source count ──────────────────────────
    ROW2 = 60
    n_total   = len(sources)
    n_visible = sum(1 for idx in sources if filter_set is None or idx in filter_set)
    if filter_mode:
        flt_lbl  = f"FILTER>"
        flt_val  = f" {filter_input}_"
        lbl_col  = (0, 200, 255)
        val_col  = (255, 255, 255)
    else:
        flt_lbl  = "FILTER"
        flt_val  = "  " + ("*" if filter_set is None else ",".join(
            str(i + 1) for i in sorted(filter_set)))
        if len(flt_val) > 52: flt_val = flt_val[:49] + "..."
        lbl_col  = (0, 120, 160)
        val_col  = (160, 160, 180)
    (lw, _), _ = cv2.getTextSize(flt_lbl, _SFB, 0.44, 1)
    cv2.putText(panel, flt_lbl, (10, ROW2), _SFB, 0.44, lbl_col, 1, _LW)
    cv2.putText(panel, flt_val, (10 + lw, ROW2), _SF, 0.44, val_col, 1, _LW)
    # source count badge (right-aligned)
    cnt_txt = f"{n_visible} / {n_total}  src"
    (tw, _), _ = cv2.getTextSize(cnt_txt, _SF, 0.40, 1)
    cv2.putText(panel, cnt_txt, (DISP_W - tw - 10, ROW2), _SF, 0.40, (70, 70, 85), 1, _LW)

    cv2.line(panel, (0, 70), (DISP_W, 70), (30, 30, 36), 1)

    # ── Row 3 (y=90): debug info or key hint ─────────────────────────────
    ROW3 = 90
    if filter_set is not None and n_visible <= 10:
        parts = []
        for idx in sorted(filter_set):
            if idx in sources:
                s    = sources[idx]
                name = _source_names.get(idx, f"#{idx+1}")
                parts.append(f"{name}  az={s.azimuth:.1f}\u00b0  el={s.elevation:.1f}\u00b0"
                              f"  g={s.gain_db:.0f}dB  \u2192({s.x},{s.y})")
        info = "     ".join(parts) if parts else "no data yet for selected sources"
        cv2.putText(panel, info, (10, ROW3), _SF, 0.40, (0, 200, 100), 1, _LW)
    else:
        hint = "V=viz   T=trail   Space=freeze   P=panner   E=rec   F=filter   R=reset   Q=quit"
        cv2.putText(panel, hint, (10, ROW3), _SF, 0.38, (50, 50, 60), 1, _LW)

    # row labels on grid (re-render over what was set earlier)
    for row in range(GH):
        py = row * SCALE_Y + SCALE_Y // 2 + 5
        cv2.putText(grid, f"Y{row}", (3, py), _SF, 0.32, (45, 45, 50), 1, _LW)

    return np.vstack([grid, panel])


# ─── Polar panner overview ────────────────────────────────────────────────────

def build_panner(sources: Dict[int, SourceState], filter_set: set[int] | None) -> np.ndarray:
    """
    420 × 420 px top-down azimuth sphere.
    Front = top, Left = left, Right = right, Back = bottom.
    Elevation shrinks the distance from centre (el=90° = centre, el=0° = edge).
    """
    sz = PANNER_SIZE
    img = np.zeros((sz, sz, 3), dtype=np.uint8)
    img[:] = (10, 10, 14)
    cx, cy = sz // 2, sz // 2
    R  = sz // 2 - 30

    # concentric rings with degree labels
    for i, (frac, deg_lbl) in enumerate([(0.33, "60°"), (0.67, "30°"), (1.0, "")]):
        r = int(R * frac)
        cv2.circle(img, (cx, cy), r, (28, 28, 36), 1)
        if deg_lbl:
            (tw, th), _ = cv2.getTextSize(deg_lbl, _SF, 0.32, 1)
            cv2.putText(img, deg_lbl, (cx + r - tw - 2, cy - 4), _SF, 0.32, (45, 45, 55), 1, _LW)

    # crosshair
    cv2.line(img, (cx, cy - R - 6), (cx, cy + R + 6), (28, 28, 36), 1)
    cv2.line(img, (cx - R - 6, cy), (cx + R + 6, cy), (28, 28, 36), 1)

    # compass labels
    _labels = [("FRONT", cx, cy - R - 14), ("BACK", cx, cy + R + 22),
               ("L", cx - R - 22, cy + 5), ("R", cx + R + 8, cy + 5)]
    for txt, lx, ly in _labels:
        (tw, _), _ = cv2.getTextSize(txt, _SFB, 0.42, 1)
        cv2.putText(img, txt, (lx - tw // 2, ly), _SFB, 0.42, (55, 55, 70), 1, _LW)

    # title
    cv2.putText(img, "PANNER", (8, 18), _SFB, 0.48, (0, 120, 160), 1, _LW)
    n_vis = sum(1 for idx in sources if filter_set is None or idx in filter_set)
    cv2.putText(img, f"{n_vis} src", (sz - 54, 18), _SF, 0.38, (50, 50, 65), 1, _LW)

    # source dots
    for idx, src in sources.items():
        visible = filter_set is None or idx in filter_set
        rad    = math.radians(src.azimuth)
        el_rad = math.radians(max(-90.0, min(90.0, src.elevation)))
        dist   = math.cos(el_rad) * R
        px = int(cx - math.sin(rad) * dist)
        py = int(cy - math.cos(rad) * dist)
        if visible:
            lbl = _source_names.get(idx, str(idx + 1))
            (tw, th), _ = cv2.getTextSize(lbl, _SF, 0.38, 1)
            radius = max(9, tw // 2 + 5)
            # glow
            cv2.circle(img, (px, py), radius + 3, (0, 50, 70), -1)
            # fill
            cv2.circle(img, (px, py), radius, (0, 200, 255), -1)
            # border
            cv2.circle(img, (px, py), radius + 1, (0, 100, 140), 1)
            # centered label in black
            cv2.putText(img, lbl, (px - tw // 2, py + th // 2),
                        _SF, 0.38, (0, 0, 0), 1, _LW)
        else:
            cv2.circle(img, (px, py), 3, (35, 35, 45), -1)
            cv2.circle(img, (px, py), 4, (55, 55, 65), 1)
    return img


# ─── OSC address patterns ────────────────────────────────────────────────────

_AZ_RE = re.compile(r"^/MultiEncoder/azimuth(\d+)$")
_EL_RE = re.compile(r"^/MultiEncoder/elevation(\d+)$")
_GA_RE = re.compile(r"^/MultiEncoder/gain(\d+)$")


# ─── OSC handlers ────────────────────────────────────────────────────────────

def _handle_azimuth(address: str, *args) -> None:
    m = _AZ_RE.match(address)
    if not m or not args:
        return
    idx = int(m.group(1))
    src = _get_source(idx)
    with _lock:
        src.azimuth = float(args[0])
        src.x, src.y = az_to_xy(src.azimuth)
    _maybe_dispatch(idx, src)


def _handle_elevation(address: str, *args) -> None:
    m = _EL_RE.match(address)
    if not m or not args:
        return
    idx = int(m.group(1))
    src = _get_source(idx)
    with _lock:
        src.elevation = float(args[0])
        # elevation not used for position; stored for future sACN use
    _maybe_dispatch(idx, src)


def _handle_gain(address: str, *args) -> None:
    m = _GA_RE.match(address)
    if not m or not args:
        return
    idx = int(m.group(1))
    src = _get_source(idx)
    with _lock:
        src.gain_db = float(args[0])


def _handle_default(address: str, *args) -> None:
    pass   # silently ignore unknown addresses


def _maybe_dispatch(idx: int, src: SourceState) -> None:
    """Rate-limit dispatch() calls to at most DISPATCH_HZ per source."""
    now = time.monotonic()
    if now - src.last_dispatch >= 1.0 / DISPATCH_HZ:
        src.last_dispatch = now
        dispatch(idx, src.x, src.y, src)


# ─── OSC server (runs in background daemon thread) ───────────────────────────

def _osc_thread(ip: str, port: int) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    d = Dispatcher()
    d.map("/MultiEncoder/azimuth*",   _handle_azimuth,   needs_reply_address=False)
    d.map("/MultiEncoder/elevation*", _handle_elevation, needs_reply_address=False)
    d.map("/MultiEncoder/gain*",      _handle_gain,      needs_reply_address=False)
    d.set_default_handler(_handle_default)

    async def _serve():
        server = AsyncIOOSCUDPServer((ip, port), d, loop)
        transport, _ = await server.create_serve_endpoint()
        print(f"[OSC]  listening on {ip}:{port}")
        try:
            await asyncio.sleep(float("inf"))
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            transport.close()

    loop.run_until_complete(_serve())


# ─── Main: OpenCV preview loop (30 fps) ──────────────────────────────────────

def main(ip: str, port: int) -> None:
    global _viz_idx, _filter_set, _filter_input, _filter_mode
    global _trails_on, _frozen, _panner_on
    global _recording, _rec_writer, _rec_frames

    t = threading.Thread(target=_osc_thread, args=(ip, port), daemon=True)
    t.start()

    print(f"[preview]  480 x 9 grid  ->  display {DISP_W} x {WIN_H} px")
    print(f"[preview]  V=viz mode   F=filter   R=reset   Q=quit")

    cv2.namedWindow("Monolith – OSC spatial", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Monolith – OSC spatial", DISP_W, WIN_H)

    _panner_was_on = False
    while True:
        if not _frozen:
            # ── smooth display positions ──────────────────────────────────
            with _lock:
                for src in _sources.values():
                    src.sx += SMOOTH_ALPHA * (src.x - src.sx)
                    src.sy += SMOOTH_ALPHA * (src.y - src.sy)

        snap = _snapshot()

        # ── update trails (main thread only) ─────────────────────────────
        if not _frozen:
            if _trails_on:
                for idx, src in snap.items():
                    if idx not in _trails:
                        _trails[idx] = deque(maxlen=TRAIL_LEN)
                    _trails[idx].append((src.sx, src.sy))
            else:
                _trails.clear()

        frame = build_display(snap, _filter_set, _viz_idx,
                              _filter_mode, _filter_input,
                              trails=(_trails if _trails_on else None),
                              frozen=_frozen)
        cv2.imshow("Monolith – OSC spatial", frame)
        # ── write raw 480×9 canvas to WebM if recording ─────────────────
        if _recording and _rec_writer is not None and not _frozen:
            raw = render_frame(snap, _filter_set, *_VIZ_MODES[_viz_idx][1:])
            gray = np.clip(raw, 0, 255).astype(np.uint8)
            big  = cv2.resize(gray, (VW, VH), interpolation=cv2.INTER_NEAREST)
            _rec_writer.write(big)
            _rec_frames += 1
        # ── panner window ─────────────────────────────────────────────────
        if _panner_on:
            if not _panner_was_on:
                cv2.namedWindow("Monolith – Panner", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Monolith – Panner", PANNER_SIZE, PANNER_SIZE)
            cv2.imshow("Monolith – Panner", build_panner(snap, _filter_set))
        elif _panner_was_on:
            try:
                cv2.destroyWindow("Monolith – Panner")
            except Exception:
                pass
        _panner_was_on = _panner_on

        key = cv2.waitKey(33) & 0xFF   # ~30 fps

        # ── filter input mode ────────────────────────────────────────────
        if _filter_mode:
            if key == 13:   # Enter – apply
                parsed = _parse_filter(_filter_input)
                if parsed is not None or _filter_input.strip() in ("", "*", "all"):
                    _filter_set   = parsed
                    _filter_input = ""
                    _filter_mode  = False
                    lbl = "*" if _filter_set is None else str(sorted(i + 1 for i in _filter_set))
                    print(f"[filter]  set -> {lbl}")
                else:
                    print(f"[filter]  parse error: '{_filter_input}'")
                    _filter_input = ""
            elif key == 27:  # Esc – cancel
                _filter_input = ""
                _filter_mode  = False
            elif key == 8 or key == 127:  # Backspace
                _filter_input = _filter_input[:-1]
            elif 32 <= key <= 126:  # printable ASCII
                _filter_input += chr(key)
            continue

        # ── normal key handling ──────────────────────────────────────────
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            with _lock:
                _sources.clear()
            _filter_set   = None
            _filter_input = ""
            _trails.clear()
            print("[preview]  sources reset")
        elif key in (ord('v'), ord('V')):
            _viz_idx = (_viz_idx + 1) % len(_VIZ_MODES)
            print(f"[viz]  mode -> {_VIZ_MODES[_viz_idx][0]}")
        elif key in (ord('t'), ord('T')):
            _trails_on = not _trails_on
            if not _trails_on:
                _trails.clear()
            print(f"[trail]  {'on' if _trails_on else 'off'}")
        elif key == 32:   # Space – freeze
            _frozen = not _frozen
            print(f"[freeze]  {'on' if _frozen else 'off'}")
        elif key in (ord('p'), ord('P')):
            _panner_on = not _panner_on
            print(f"[panner]  {'on' if _panner_on else 'off'}")
        elif key in (ord('e'), ord('E')):
            if not _recording:
                out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "Monolith_video_folder")
                os.makedirs(out_dir, exist_ok=True)
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(out_dir, f"osc_live_{ts}.webm")
                _rec_writer = FfmpegWriter(path, FPS, VW, VH)
                _rec_frames = 0
                _recording  = True
                print(f"[rec]  started -> {path}")
            else:
                _recording = False
                if _rec_writer is not None:
                    _rec_writer.release()
                    dur = _rec_frames / FPS
                    print(f"[rec]  saved  {_rec_frames} frames  ({dur:.1f}s) -> {_rec_writer.path}")
                    _rec_writer = None
        elif key in (ord('f'), ord('F')):
            _filter_input = ""
            _filter_mode  = True

    cv2.destroyAllWindows()
    # flush any open recording
    if _recording and _rec_writer is not None:
        _rec_writer.release()
        dur = _rec_frames / FPS
        print(f"[rec]  saved  {_rec_frames} frames  ({dur:.1f}s) -> {_rec_writer.path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSC -> Monolith 480x9 LED matrix  |  live preview + sACN hook"
    )
    p.add_argument("--ip",   default=DEFAULT_IP,
                   help=f"Interface to bind (default: {DEFAULT_IP})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"UDP port (default: {DEFAULT_PORT})")
    p.add_argument("--names", default=None, metavar="CSV",
                   help="CSV file mapping source numbers to names (e.g. sources.csv)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    names_file = args.names or ("sources.csv" if os.path.exists("sources.csv") else None)
    if names_file:
        load_names(names_file)
    main(args.ip, args.port)
