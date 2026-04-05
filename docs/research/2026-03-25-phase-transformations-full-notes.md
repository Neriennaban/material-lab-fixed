# Phase Transformations in Metals and Alloys: Full Notes

Source: `docs/Literature/Easterling_Kenneth_E_Porter_Phase_Trans.pdf`

Status: `closed_for_current_project_batch`

This note captures the full-project reading pass over chapters 1-6 and records what was implemented now versus what remains deferred for the current Fe-C `pro` backend.

## Chapter 1. Thermodynamics and Phase Diagrams
Status: `implemented_now`

### Key points
- Equilibrium remains the reference state for all transformation reasoning; kinetics only determines how fast the system approaches it.
- The Gibbs free energy definition `G = H - T S` is the core scalar for transformation direction and driving force.
- Chemical potential, common tangents, and binary equilibrium remain the correct conceptual basis for phase fraction and local-equilibrium reasoning.
- Interfacial energy shifts equilibrium and therefore matters when metastable products compete with equilibrium products.
- The Arrhenius temperature dependence of activated processes is introduced as:
  - `rate ∝ exp(-ΔHa / R T)`

### Engineering extraction for this project
- No new CALPHAD or free-energy minimizer is introduced.
- Chapter 1 is used as the thermodynamic justification for:
  - driving-force-aware family competition;
  - temperature-window logic expressed as surrogates;
  - preserving `ferrite_pearlite_competition_index` as a selector rather than deleting it prematurely.

### Implemented now
- `continuous_cooling_shift_factor` and `hardenability_factor` are treated as thermodynamics-informed surrogates rather than arbitrary heuristics.
- Provenance now explicitly separates thermodynamics from diffusion and growth.

## Chapter 2. Diffusion
Status: `implemented_now`

### Key points
- Diffusion is thermally activated and strongly temperature-dependent.
- Interstitial and substitutional diffusion must be treated differently conceptually, even if the current code uses engineering surrogates.
- Fickian diffusion and nonsteady-state diffusion remain the right foundation for diffusional transformations.
- Local equilibrium across moving interphase boundaries is valid only when interface transfer is not rate-limiting.
- For multiphase diffusion the boundary velocity is controlled by the diffusive flux balance; interface migration can be diffusion-controlled or interface-controlled.

### Engineering extraction for this project
- Full diffusion PDE solving is out of scope for the current generator.
- Chapter 2 is implemented as:
  - Arrhenius-like temperature-sensitive weighting;
  - family-specific effective exposures for ferrite/pearlite/bainite;
  - `diffusional_equivalent_time_s` as an additivity-style surrogate.

### Implemented now
- `diffusional_equivalent_time_s`
- `ferrite_nucleation_drive`
- `pearlite_nucleation_drive`
- `bainite_nucleation_drive`
- diffusion-informed suppression of diffusional products at high hardenability / stronger continuous-cooling shift

## Chapter 3. Crystal Interfaces and Microstructure
Status: `implemented_now`

### Key points
- Coherent, semicoherent, and incoherent interfaces differ strongly in mobility and growth behavior.
- Interface-controlled and diffusion-controlled growth are distinct limiting cases and must not be collapsed into one physical story.
- Grain boundary migration and grain growth kinetics are central to microstructure scale.
- Morphology is not cosmetic; it follows interface energetics and growth constraints.

### Engineering extraction for this project
- The current system remains surrogate-based, but interface character is now reflected in morphology choices.
- Chapter 3 directly supports:
  - allotriomorphic versus Widmanstätten ferrite split;
  - boundary-following versus side-plate growth morphology;
  - the decision to keep upper/lower bainite as morphology families rather than separate stage taxonomies.

### Implemented now
- `allotriomorphic_ferrite_binary` is preserved as a first-class downstream signal.
- `widmanstatten_sideplates_binary` remains the ferrite side-plate signal.
- Reflected-light rendering now supports a dedicated low-amplitude allotriomorphic ferrite modulation.
- Validation now exposes allotriomorphic ferrite area fraction separately from Widmanstätten coverage.

## Chapter 4. Solidification
Status: `deferred_with_reason`

### Key points
- Nucleation and growth during solidification, dendrites, segregation, rapid solidification, and metallic glasses are treated here.
- The chapter is highly relevant to casting and melt-based routes.
- The chapter is not the correct immediate driver for the current Fe-C solid-state `pro` path.

### Deferred with reason
- The current batch is constrained to the Fe-C solid-state pathway:
  - `continuous transformation state -> morphology -> surface -> reflected-light`
- Solidification insights are documented for future `Al-Si` / casting routes, but not implemented in this batch to avoid mixing melt-solidification logic into the current steel microscope pipeline.

## Chapter 5. Diffusional Transformations in Solids
Status: `implemented_now`

### Key points
- The central kinetics form remains Johnson-Mehl-Avrami:
  - `f = 1 - exp(-(k t)^n)`
- The half-transformation time follows:
  - `t0.5 = (0.7 / k)^(1 / n)`
- TTT curves are C-shaped because near-equilibrium temperatures give low driving force, while low temperatures suppress diffusion.
- Ferrite, pearlite, and bainite must be treated as diffusional products with different nucleation and growth windows.
- Hardenability is fundamentally about delaying ferrite/pearlite decomposition so that bainite and martensite become accessible at slower cooling rates.
- Scheil additivity remains the right conceptual bridge from TTT-style reasoning to continuous cooling:
  - transformation begins when `Σ Δt / t(T) >= 1`

### Engineering extraction for this project
- Chapter 5 is the main source for the current batch.
- Implemented as surrogate, not literal TTT/CCT table solving:
  - family-specific effective exposure;
  - diffusional equivalent time;
  - hardenability factor;
  - continuous-cooling shift factor;
  - separate ferrite/pearlite/bainite nucleation drives.

### Implemented now
- `hardenability_factor`
- `continuous_cooling_shift_factor`
- family-specific diffusional weighting and handoff
- stronger ferrite-side evidence for selecting `allotriomorphic` versus `widmanstätten`
- upper/lower bainite split remains morphology-only but is now more fully materialized via spacing, width bias, density target, and split-aware validation

## Chapter 6. Diffusionless Transformations
Status: `implemented_now`

### Key points
- Martensite remains diffusionless and structurally displacive.
- The lath/plate transition is linked to transformation temperature, alloying, and deformation mode of the lattice-invariant shear.
- Growth, stabilization, retained austenite, and tempering must be treated explicitly as part of the transformation story.
- Tempering is not a new generator; it is a post-martensitic evolution path.

### Engineering extraction for this project
- The existing Koistinen-Marburger-style martensite branch is retained as the main quantitative driver.
- Chapter 6 is used to strengthen:
  - martensite subtype semantics;
  - plate/lath crossover logic;
  - provenance around tempered products and derived labels.

### Implemented now
- `troostite_*` and `sorbite_*` remain derived engineering labels over existing families.
- Martensite remains a separate displacive branch with subtype-aware morphology.
- Provenance now explicitly records diffusionless transformation semantics separately from diffusional and interface-growth semantics.

## What changed in code now
- `core/metallography_pro/transformation_fe_c.py`
  - added Porter/Easterling-driven surrogate kinetics metadata
  - integrated hardenability and continuous-cooling shift into family competition
  - strengthened ferrite morphology gating with ferrite-side evidence
- `core/metallography_pro/morphology_fe_c.py`
  - upper/lower bainite split now uses thickness and density state fields
  - allotriomorphic ferrite remains explicit in feature maps
- `core/metallography_pro/reflected_light.py`
  - allotriomorphic ferrite gets a separate low-amplitude modulation
- `core/metallography_pro/validation_pro.py`
  - allotriomorphic ferrite coverage and split-aware bainite metrics added
- `core/metallography_pro/pipeline_pro.py`
  - new kinetics fields mirrored into metadata blocks

## Deferred items
- Full CALPHAD/free-energy solver: deferred; current surrogate layer is sufficient for this batch.
- Explicit PDE diffusion solver: deferred; too expensive and unnecessary for current microscope realism target.
- Chapter 4 solidification logic in Fe-C `pro`: deferred by scope boundary.
- Full TTT/CCT table interpolation from literature/atlas data: deferred until the next kinetics-focused batch or external dataset integration.

## Closure decision
This book is considered closed for the current project phase because:
- all chapters 1-6 were read and triaged;
- chapters 1, 2, 3, 5, and 6 have direct code impact in the current batch;
- chapter 4 is documented and intentionally deferred;
- tests now cover the changed selector semantics, metadata consistency, and morphology signal propagation.
