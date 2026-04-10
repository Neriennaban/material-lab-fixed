# v1.0.0 — Last working generation

First public release of the Fe-C metallography suite. Bundles the
procedural microstructure generator and the virtual microscope
viewer as standalone Windows executables — no Python install, no
dependencies.

Built from commit [`d89d4b0`](https://github.com/Neriennaban/material_lab/commit/d89d4b0)
(`perf(fe-c): Phase D.2/D.3 — faster render + bright cementite network`).

## What's in this release

| Asset | Size | Description |
|---|---|---|
| `ferro_micro_generator.exe` | 109 MB | Fe-C microstructure generator (PySide6 GUI). Loads the 49 bundled presets from `presets_v3/` and renders PNG micrographs at user-selectable resolution. |
| `ferro_micro_microscope.exe` | 102 MB | Virtual microscope viewer for the PNG samples produced by the generator. |
| `SHA256SUMS.txt` | — | SHA-256 checksums of both executables. |
| `fe_pure_armco_bw_v3.png` | 336 KB | 1024×1024 sample: pure Armco iron, bright ferrite grains without dark clusters (Phase D.2 result). |
| `aisi_1040_nital_warm_v3.png` | 487 KB | 1024×1024 sample: AISI 1040 in the nital-warm colour palette (Phase A10.1). |
| `fe_c_eutectoid_textbook.png` | 635 KB | 1024×1024 sample: eutectoid 0.77 %C pearlite. |
| `steel_u12_annealed_v3.png` | 639 KB | 1024×1024 sample: tool steel U12 with the bright secondary cementite grain-boundary network (Phase D.3 result). |
| `cast_iron_white_hypereutectic_v3.png` | 742 KB | 1024×1024 sample: hypereutectic white cast iron (5.5 %C) with primary cementite needles on a leopard-ledeburite matrix (A1 + A2). |

## Highlights

Everything introduced during Phases A through D of the ferro-micro
integration plan is included:

### Morphology
- **Stage taxonomy**: opt-in `white_cast_iron_hypoeutectic / eutectic / hypereutectic`, `bainite_upper / bainite_lower` — specialised dispatchers in `fe_c_unified.py` for each.
- **Primary cementite needles** (A1) for hypereutectic white cast iron.
- **Leopard ledeburite** (A2) — bright cementite matrix with quasi-periodic dark pearlite blobs.
- **Fe-C austenite dendrites** (A3) for hypoeutectic white cast iron via an L-system.
- **Magnification-aware pearlite** (A4): `S₀ = 8.3 / ΔT` drives the lamella period; switches to uniform dark blob when lamellae are sub-pixel.
- **Cementite network thickness** (A5): scales 1.5 px → 7 px between 0.77 %C and 2.14 %C.
- **Upper / lower bainite textures** (A6): feathery packets vs needle-like laths with intra-lath Fe₃C.
- **Ferrite annealing twins** (A7) and **anisotropic nital etching** (A9).
- **Retained austenite localisation** (A8): opt-in boundary-bias knob in `_build_martensitic_render`.

### Visual / colour
- **RGB post-process palette** (A10.0): `grayscale_nital`, `nital_warm`, `dic_polarized`, `tint_etch_blue_yellow`. Backward-compatible: existing grayscale presets unchanged.
- **AISI 1020-1050 nital-warm preset series** (A10.1), **Armco DIC polarized** (A10.3), **Klemm tint-etched upper bainite** (A10.4), **Pure Armco BW 100×** (A10.5), **eutectoid 1000× high-mag** (A10.2).
- **11 standard grade presets** (Phase B1): Армко, Steel 08/10/20, Steel 45 quench, U10/U12/U13, white cast iron (hypoeutectic / eutectic / hypereutectic).

### Phase D fixes (from the user's field review)
- **D.1 — bright pure ferrite, no dark clusters.** Removed the directional per-grain gradient that produced aligned "dark half-moons", narrowed the per-grain base tone to ±0.025, halved the residual LF noise amplitude.
- **D.2 — near-linear render scaling up to 16K.** Four hot-paths optimised:
  - `_power_voronoi_labels` → `cKDTree` on augmented 3D points (power distance → Euclidean distance), `workers=-1`.
  - `realism_utils.smooth` → downsample/blur/upsample pyramid for sigma ≥ 5.
  - `pure_ferrite_generator` → dedicated `_fast_low_frequency_field` (128×128 buffer + zoom-up).
  - `_lift_small_dark_blobs` (morphology_engine + pipeline_v3) → vectorised with `np.bincount` / `np.isin`, O(N+K) instead of O(N·K).
- **D.3 — pearlite_cementite: bright grain-boundary network.** `pearlite_cementite` stage now grows the proeutectoid cementite as a dilated Voronoi boundary mask (not whole grains), tone 238-240, and the pipeline repaints the mask to 240 after every post-process pass so the thin bright network survives the final histogram stretch.

### API / CLI / metrics
- **`ferro_micro_api.generate(...)`** (Phase B2) — Python facade with the §10 parameter names from the TZ.
- **`scripts.ferro_micro_cli`** (Phase B3) — argparse CLI with `--atlas` batch mode, `--thermal-program` JSON import, `--color-mode` switch.
- **Quality metrics** (Phase C1) — `phase_fraction_error`, `histogram_intersection`, `ssim_vs_reference`, `fft_lamellae_period_px`, `hough_orientation_histogram`, `grain_size_astm`.
- **Atlas generator** (Phase C3) — 18 reference compositions spanning 0-6.67 %C with manifest + `--with-metrics` mode.
- **Blind-eval script** (Phase C5) — shuffled real/synthetic sample mixer with answer key.
- **Optional GPU backend** (Phase C4) — CuPy fallback helper; CPU scipy path is the default.

### Regression
180+ tests green on the final commit, snapshot baseline regenerated
for all 49 presets.

## System requirements

- Windows 10 or 11, x64.
- No external Python or dependencies needed — everything is bundled.
- ≥ 4 GB free RAM for 1024×1024 renders; 8+ GB recommended for 4K and above.
- 16K rendering runs in roughly 10-30 minutes on a modern CPU.

## Quick start

Generator (choose preset, adjust parameters, hit "Финальный рендер"):

```
ferro_micro_generator.exe
```

Microscope viewer (browse rendered samples from a directory):

```
ferro_micro_microscope.exe --samples-dir path\to\your\samples
```

Both executables launch the Qt GUI directly — no console window.

## Verify integrity

```powershell
# PowerShell
Get-FileHash ferro_micro_generator.exe,ferro_micro_microscope.exe -Algorithm SHA256
```

Compare the output to `SHA256SUMS.txt`. Expected values:

```
2dea7dd94bce1778749a7af93286fc885da9a20111fa8e57499b22bd045b6957  ferro_micro_generator.exe
3070de6a5f48a09872403b8bea6bb65154bbd29d64d2b11940dfcd61c442e5d9  ferro_micro_microscope.exe
```

## Known limitations

- **16K rendering** takes several minutes; the 4K benchmark lands at 67-88 seconds on a single-core CPU.
- **Pro-realistic generation mode** ships with the grayscale colour palette only — `nital_warm`, `dic_polarized` and `tint_etch_blue_yellow` are honoured in the default engineering mode.
- **`pyqtgraph` thermal plots** are disabled in the bundled exe to keep the size down; the warning message in stdout is cosmetic.
- **UPX-compressed** — some antivirus engines flag UPX binaries as suspicious heuristics. The binaries are unsigned; Windows SmartScreen may require "Run anyway" on first launch.

## Source

Full source, test suite and the full commit history that produced this release live at:
- Repository: https://github.com/Neriennaban/material_lab
- Branch: `last-working-generation`
- HEAD: `d89d4b0`
