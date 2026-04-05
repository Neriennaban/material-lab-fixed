# COACH/SSLLB Project Integration Note

Source: `Optical_Imaging_Using_Coded_Aperture_Correlation_H.pdf`

## Gap analysis

Before this batch, the project had:
- a static `pro` reflected-light renderer with several optical modes but no engineered PSF vocabulary;
- a live microscope path with physical focus blur but no beam-family-inspired axial shaping;
- no shared abstraction between the static render path and the live microscope path for DOF engineering, sectioning, or axial sculpting.

The paper is most applicable to:
- PSF engineering
- depth-of-field engineering
- single-view sectioning
- axial sculpting by hybrid lens/axicon-like surrogates

The paper is not directly applicable to:
- phase transformation kinetics
- phase fraction prediction
- metallurgical morphology generation

## Implemented subset

### Implemented now
- Shared engineered PSF vocabulary:
  - `standard`
  - `bessel_extended_dof`
  - `airy_push_pull`
  - `self_rotating`
  - `stir_sectioning`
  - `lens_axicon_hybrid`
- Shared parameters in `microscope_profile`:
  - `psf_profile`
  - `psf_strength`
  - `sectioning_shear_deg`
  - `hybrid_balance`
- Live microscope integration:
  - new PSF-aware metadata in `simulate_microscope_view()`
  - realtime-friendly image-domain surrogates for DOF and sectioning
- Static `pro` renderer integration:
  - profile-aware image-domain PSF surrogates
  - metadata describing applied axial profile behavior
- Validation:
  - profile-aware diagnostics for DOF and sectioning behavior
- UI round-trip:
  - generator V3 and microscope UI both expose the same PSF vocabulary

### Deferred
- Full COACH IPSF library acquisition and correlation reconstruction inside the runtime microscope loop
- Literal LRRA/INDIA per-frame reconstruction in live mode
- Full wave-optics Airy/Bessel propagation for every render

## Mapping from paper to implementation

- Bessel beams
  - mapped to `bessel_extended_dof`
  - intended effect: weaker axial sensitivity with better lateral retention than plain defocus

- Airy beams
  - mapped to `airy_push_pull`
  - intended effect: depth-coded lateral asymmetry/shift surrogate

- Self-rotating beams
  - mapped to `self_rotating`
  - intended effect: angle-dependent local contrast rotation surrogate

- RQPF / DOF engineering
  - mapped to `bessel_extended_dof` and `lens_axicon_hybrid`
  - intended effect: engineered axial response without replacing the current pipeline

- STIR sectioning
  - mapped to `stir_sectioning`
  - intended effect: off-slice suppression and directional sectioning signature

- Hybrid lens/axicon axial sculpting
  - mapped to `lens_axicon_hybrid`
  - intended effect: controlled interpolation between standard focus and extended DOF

## Validation closure

This batch is considered integrated because:
- both static and live optics paths now share the same profile vocabulary;
- tests verify that `standard` remains backward-compatible;
- tests verify that engineered profiles produce measurable metadata and image changes;
- the book’s applicable ideas were translated into project-safe surrogates instead of remaining as notes-only recommendations.
