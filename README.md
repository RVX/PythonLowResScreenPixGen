# MONOLITH — Generative Ceiling Lighting Suite
**Art Basel 2026 / Julian Charrière**

Generative B&W video content for a ceiling of GLP Impression X4 Bar 20 fixtures,
controlled via grandMA3 Bitmap fixture mapping.

---

## System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONTENT GENERATION                               │
│                                                                         │
│   monolith_ceiling_videos.py                                            │
│   ┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐   │
│   │  NumPy       │    │  Pattern algorithm   │    │  FfmpegWriter   │   │
│   │  canvas      │───▶│  (57 generators)     │───▶│  pipe: gray→VP8│   │
│   │  480×9 px    │    │  float32 0–255       │    │  WebM output    │   │
│   │  float32     │    └─────────────────────┘    └────────┬─────────┘   │
│   └──────────────┘                                        │            │
└───────────────────────────────────────────────────────────┼────────────┘
                                                            │
                              Monolith_video_folder/*.webm  │
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          grandMA3  v2.3                                 │
│                                                                         │
│   Media Pool ──▶  Video Pool ──▶  Bitmap Object                        │
│                                   ┌───────────────────────────────┐    │
│                                   │  Bitmap Configuration         │    │
│                                   │  Canvas Width  = 480          │    │
│                                   │  Canvas Height = 9            │    │
│                                   │  Content Mode  = Clip         │    │
│                                   │  Bitmap Channel Source = Luma │    │
│                                   └──────────────┬────────────────┘    │
│                                                  │ Luma → Dimmer attr  │
└──────────────────────────────────────────────────┼─────────────────────┘
                                                   │  DMX (Art-Net / sACN)
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CEILING FIXTURE ARRAY                             │
│                                                                         │
│   9 rows × 24 bars × 20 LEDs = 4,320 individual LED pixels             │
│   GLP Impression X4 Bar 20  (Mode 1 = 20-pixel, 47 DMX ch/bar)         │
│   CW + WW channels driven by the same Dimmer value                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Ceiling Grid — Physical Layout

### Axes

| Axis | Direction | Unit | Range | Physical span |
|------|-----------|------|-------|---------------|
| **X** | Horizontal (left→right) | 1 px = 1 LED = ~50 mm | 0 – 479 | ~24 m |
| **Y** | Depth (back wall→audience) | 1 row = one bar row | 0 – 8 | ~56 m (rows ~7 m apart) |

- **Y = 0** — back of the room / far wall
- **Y = 8** — front of the room / audience edge

Each bar spans **20 contiguous X pixels**. Bar 0 occupies X 0–19, Bar 1 occupies X 20–39, … Bar 23 occupies X 460–479.

### Grid Map (top view, audience at bottom)

```
BACK WALL ──────────────────────────────────────────────────────────────────
                 BAR 0   BAR 1   ...  BAR 11  BAR 12  ...  BAR 22  BAR 23
                 X:0–19  X:20–39      X:220   X:240        X:440   X:460–479
                 ┌──────┬──────┬─────┬──────┬──────┬──────┬──────┬──────┐
 Y=0  (row 0)   │██████│██████│ ... │██████│██████│ ... │██████│██████│
                 ├──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┤
 Y=1  (row 1)   │██████│██████│ ... │██████│██████│ ... │██████│██████│
                 ├──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┤
 Y=2  (row 2)   │██████│██████│ ... │██████│██████│ ... │██████│██████│
                 ├──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┤
 Y=3  (row 3)   │██████│██████│ ... │██████│██████│ ... │██████│██████│
                 ├──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┤
 Y=4  (row 4)   │██████│██████│ ... │██████│██████│ ... │██████│██████│  ← depth mid
                 ├──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┤
 Y=5  (row 5)   │██████│██████│ ... │██████│██████│ ... │██████│██████│
                 ├──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┤
 Y=6  (row 6)   │██████│██████│ ... │██████│██████│ ... │██████│██████│
                 ├──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┤
 Y=7  (row 7)   │██████│██████│ ... │██████│██████│ ... │██████│██████│
                 ├──────┼──────┼─────┼──────┼──────┼──────┼──────┼──────┤
 Y=8  (row 8)   │██████│██████│ ... │██████│██████│ ... │██████│██████│
                 └──────┴──────┴─────┴──────┴──────┴──────┴──────┴──────┘
AUDIENCE ───────────────────────────────────────────────────────────────────

Each ██ block above = 20 LEDs (one GLP X4 Bar 20)
Total: 24 bars/row × 9 rows = 216 fixtures × 20 LEDs = 4,320 pixels
```

### Per-bar pixel coordinates

```
Bar  0 → X:   0 –  19    Bar 12 → X: 240 – 259
Bar  1 → X:  20 –  39    Bar 13 → X: 260 – 279
Bar  2 → X:  40 –  59    Bar 14 → X: 280 – 299
Bar  3 → X:  60 –  79    Bar 15 → X: 300 – 319
Bar  4 → X:  80 –  99    Bar 16 → X: 320 – 339
Bar  5 → X: 100 – 119    Bar 17 → X: 340 – 359
Bar  6 → X: 120 – 139    Bar 18 → X: 360 – 379
Bar  7 → X: 140 – 159    Bar 19 → X: 380 – 399
Bar  8 → X: 160 – 179    Bar 20 → X: 400 – 419
Bar  9 → X: 180 – 199    Bar 21 → X: 420 – 439
Bar 10 → X: 200 – 219    Bar 22 → X: 440 – 459
Bar 11 → X: 220 – 239    Bar 23 → X: 460 – 479
```

---

## Pixel → DMX Mapping

grandMA3 Bitmap reads the video frame and maps each pixel's **luma value (0–255)** directly to the **Dimmer attribute** of the corresponding fixture channel.

```
Video pixel  (X, Y)
      │
      ▼
Luma  0–255
      │
      ▼
grandMA3 Bitmap fixture at (col = X/20, row = Y)
      │
      ▼
GLP X4 Bar 20 — LED pixel (X mod 20) on bar (X/20) in row Y
      │
      ▼
DMX Dimmer value  0–255  ──▶  CW + WW LED brightness
```

---

## Code Architecture

```
monolith_ceiling_videos.py
│
├── Constants
│   ├── GW = 480          canvas width  (24 bars × 20 LEDs)
│   ├── GH = 9            canvas height (9 depth rows)
│   ├── FPS = 30          frame rate
│   ├── PREVIEW_SCALE = 4 upscale factor for grandMA3 (→ 1920×36)
│   └── OUT_DIR           Monolith_video_folder/
│
├── FfmpegWriter (class)
│   ├── Spawns ffmpeg subprocess
│   ├── Input:  raw grayscale bytes piped to stdin
│   ├── Codec:  libvpx (VP8) at CRF 10
│   └── Output: .webm file  (no chroma artifacts — Cb=Cr=128 guaranteed)
│
├── Helpers
│   ├── open_writer(filename)  → FfmpegWriter
│   ├── write_frame(writer, canvas)
│   │     canvas: float32 (GH×GW), 0–255
│   │     → clip → uint8 → nearest-neighbour upscale (×4) → pipe to ffmpeg
│   ├── close(writer, path_hint)
│   ├── ease_out(t)
│   └── ease_in_out(t)
│
├── Pattern functions  make_01_…() … make_57_…()
│   └── (see Video Library below)
│
└── __main__
    ├── Accepts optional argv[1] = start index  (e.g. `python script.py 43`)
    └── Runs all functions from start_from onward
```

### Why FFmpeg subprocess instead of cv2.VideoWriter?

OpenCV's VP8 encoder converts `BGR → YUV 4:2:0` internally. The lossy YUV quantisation introduces ±1 Cb/Cr rounding errors that appear as a **greenish tint** in near-black fade tails. By piping raw single-channel (gray) frames directly to a standalone `ffmpeg` process, chroma channels are set to exactly `Cb = Cr = 128` — no colour error at any brightness level.

---

## Running the Script

```powershell
# Render all 57 videos (takes ~5–10 min)
$env:PYTHONIOENCODING = "utf-8"
python monolith_ceiling_videos.py

# Resume from video N (e.g. after a crash at video 35)
python monolith_ceiling_videos.py 35

# Requirements
pip install numpy opencv-python
# ffmpeg must be in PATH (https://www.gyan.dev/ffmpeg/builds/)
```

Output goes to `Monolith_video_folder/` relative to the script.

---

## grandMA3 Import Workflow

1. Copy all `.webm` files from `Monolith_video_folder/` to the grandMA3 media server folder.
2. In grandMA3: **Media Pool → Import Video** — select all files.
3. Create a **Bitmap** fixture object:
   - Canvas Width = `480`
   - Canvas Height = `9`
   - Content Mode = `Clip`
   - Bitmap Channel Source = `Luma`
4. Assign a video to the Bitmap object's content slot.
5. The Bitmap object drives the Dimmer attribute of the 216 X4 Bar 20 fixtures in the 9×24 patch.

---

## Video Library (57 patterns)

### Energetic / Structured  (01–08)

| # | File | Description |
|---|------|-------------|
| 01 | `01_scan_bounce_gol.webm` | Row scan → bounce → Conway's Game of Life explosion |
| 02 | `02_rain.webm` | Per-column vertical rain drops with fading trails |
| 03 | `03_wave_ripple.webm` | 3 interfering sine waves along X axis |
| 04 | `04_lightning.webm` | Stochastic horizontal discharge bolts |
| 05 | `05_meteor_shower.webm` | Diagonal streaks with comet tails |
| 06 | `06_breathing_noise.webm` | 12-layer harmonic noise with slow breath envelope |
| 07 | `07_cellular_automata.webm` | Rule-30 cellular automaton, one per row |
| 08 | `08_rain_horizontal.webm` | Comet streaks travelling along X axis |

### Delicate / Organic / Slow  (09–17)

| # | File | Description |
|---|------|-------------|
| 09 | `09_slow_aurora.webm` | Soft drifting bands with Gaussian Y falloff |
| 10 | `10_deep_pulse.webm` | Front-to-back breathing pulse, 12 s period |
| 11 | `11_fireflies.webm` | Sparse single-pixel sparks against black |
| 12 | `12_slow_cellular_drift.webm` | Rule-110 CA at 1 step / 8 frames, long phosphor decay |
| 13 | `13_tide.webm` | Slow horizontal luminous wash (~35 % peak) |
| 14 | `14_stellar_parallax.webm` | 3-layer drifting stars with parallax depth effect |
| 15 | `15_heat_diffusion.webm` | Heat-equation blooms spreading from random seeds |
| 16 | `16_column_choir.webm` | Per-column harmonic frequencies creating moiré shimmer |
| 17 | `17_depth_ripple.webm` | Expanding rings from random origin points, Y physically scaled |

### Single-Pixel Movements  (18–32)

| # | File | Description |
|---|------|-------------|
| 18 | `18_wanderer.webm` | One pixel on a slow drunk walk with long phosphor trail |
| 19 | `19_pixel_migration.webm` | ~22 pixels drifting slowly rightward |
| 20 | `20_cold_sparks.webm` | 1-frame pixel flashes — cosmic ray interference |
| 21 | `21_pixel_telegraph.webm` | Dot/dash morse-like signals from fixed stations |
| 22 | `22_random_walk_swarm.webm` | 12 pixels each on an independent random walk |
| 23 | `23_pixel_heartbeat.webm` | Pixels pulse in cardiac lub-dub rhythm |
| 24 | `24_pixel_orbits.webm` | Pixels in elliptical orbits around a fixed centre |
| 25 | `25_trailing_cursor.webm` | Single pixel scanning row by row |
| 26 | `26_sparse_static.webm` | 2–6 dim random pixels per frame, fast fade |
| 27 | `27_gravity_well.webm` | Pixels attracted to 2 slowly drifting gravity wells |
| 28 | `28_pixel_erosion.webm` | A full row slowly erodes to black, reseeds |
| 29 | `29_pixel_repulsion.webm` | Pairs of pixels repel each other, bounce at edges |
| 30 | `30_quantum_dots.webm` | Near-invisible probabilistic shimmer (~3–8/255) |
| 31 | `31_pixel_drift_field.webm` | 40 pixels following a slowly rotating vector field |
| 32 | `32_pixel_constellations.webm` | Fixed stars with periodic bright edges between them |

### Y-Axis / Depth Movements  (33–42)

| # | File | Description |
|---|------|-------------|
| 33 | `33_depth_scan.webm` | Full-width row sweeping back ↔ front |
| 34 | `34_depth_rain.webm` | Drops travelling front→back along Y rows |
| 35 | `35_depth_cascade.webm` | Rows activating back→front sequentially |
| 36 | `36_depth_curtain.webm` | Brightness curtain rolling front ↔ back |
| 37 | `37_y_stripes_drift.webm` | Each row breathes at its own independent rate |
| 38 | `38_depth_meteor.webm` | Bright row-bands streaking along the Y axis |
| 39 | `39_row_sequencer.webm` | Random rhythmic row activations (step sequencer) |
| 40 | `40_depth_wander.webm` | 3 pixels drifting mostly in Y with rare X steps |
| 41 | `41_front_glow_pulse.webm` | Pulse originates at audience edge, travels to back |
| 42 | `42_y_axis_ripple_chain.webm` | Ripples spawning at front row, expanding toward back |

### Natural Patterns — Waves / Flows / Swarms / Particles  (43–57)

| # | File | Description |
|---|------|-------------|
| 43 | `43_murmuration.webm` | Boids flocking: cohesion + separation + alignment |
| 44 | `44_kelp_sway.webm` | Per-column kelp tips oscillating in Y with a slow current |
| 45 | `45_caustics.webm` | Water caustics from 6 interfering plane waves (squared) |
| 46 | `46_smoke_plume.webm` | Turbulent particles rising from a fixed emitter |
| 47 | `47_lava_lamp.webm` | 7 Gaussian blobs oscillating slowly in Y |
| 48 | `48_bioluminescence.webm` | Sweeping trigger wave lights each column (dinoflagellates) |
| 49 | `49_shoal_scatter.webm` | Fish school + periodic predator strike scatter + regroup |
| 50 | `50_thermal_convection.webm` | Hot/cool column convection cells rolling across depth |
| 51 | `51_chladni.webm` | Chladni standing-wave figures with evolving frequency ratio |
| 52 | `52_river_flow.webm` | Particles in a curl flow field producing meanders and eddies |
| 53 | `53_mycelium.webm` | Branching fungal tendrils growing from random seed points |
| 54 | `54_pheromone_trail.webm` | Ant colony pheromone trail converging between two food sources |
| 55 | `55_coral_pulse.webm` | Radial rings pulsing from 8 fixed polyp positions |
| 56 | `56_deep_current.webm` | Laminar Poiseuille shear — fastest flow in centre rows |
| 57 | `57_phosphene.webm` | Bright concentric rings expanding from random origins |
| 58 | `58_breath_of_deep.webm` | Sigmoid inhale/exhale from back to front Y with X ripple texture |
| 59 | `59_silt_settling.webm` | Particles born mid-depth, sink to front row, pool and evaporate |
| 60 | `60_pressure_wave.webm` | Two damped pressure fronts bouncing along Y axis |
| 61 | `61_depth_weed.webm` | Per-column stems rooted at Y=8, tips sway in Y with slow current |
| 62 | `62_upwelling.webm` | Column groups surge bright from back to front row like deep-sea upwelling |
| 63 | `63_peristalsis.webm` | Muscular contraction ring travels Y=0→8 with trailing halo |
| 64 | `64_tidal_bore.webm` | Fast bright front rushes Y=0→8 with long turbulent aftermath |
| 65 | `65_neural_propagation.webm` | Bidirectional action-potential wave from Y origin |
| 66 | `66_slime_mold.webm` | Physarum agents with sensors converge to bright network lattice |
| 67 | `67_sand_ripple.webm` | Two interfering aeolian ripple systems with Y shear |
| 68 | `68_cilia_sweep.webm` | Metachronal cilia beat wave along X axis |
| 69 | `69_magnetotaxis.webm` | Magnetic bacteria aligning to slowly rotating field |
| 70 | `70_surface_tension.webm` | Droplet blobs cohese/split via surface-tension forces |
| 71 | `71_turbulence_burst.webm` | Kolmogorov energy cascade across octaves |
| 72 | `72_symbiosis.webm` | Two particle populations: A drifts right, B seeks A and flares on contact |

---

## Fixture Reference — GLP Impression X4 Bar 20

| Parameter | Value |
|-----------|-------|
| Model | GLP Impression X4 Bar 20 |
| DMX mode used | Mode 1 (20-pixel, 47 channels/fixture) |
| LED count per bar | 20 |
| Colour | CW + WW (cool white + warm white) |
| Pixel pitch | ~50 mm |
| Bar physical width | ~1 m |
| Bars per row | 24 |
| Rows (depth) | 9 |
| Total fixtures | 216 |
| Total pixels / DMX channels driven | 4,320 pixels |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | All canvas arithmetic and array operations |
| `opencv-python` | `cv2.resize` for nearest-neighbour upscale only |
| `ffmpeg` (system binary) | VP8 encoding via subprocess pipe |

```
pip install numpy opencv-python
```

`ffmpeg` must be installed and on `PATH`. Windows build: https://www.gyan.dev/ffmpeg/builds/
