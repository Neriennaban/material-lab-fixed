# Phase Transformations in Metals and Alloys: Chapter-to-Code Matrix

Source: `docs/Literature/Easterling_Kenneth_E_Porter_Phase_Trans.pdf`

| Chapter | Topic | Project relevance | Repo target | Implemented now | Deferred reason |
| --- | --- | --- | --- | --- | --- |
| 1 | Thermodynamics and phase diagrams | High | `core/metallography_pro/transformation_fe_c.py` | Driving-force framing, surrogate hardenability/continuous-cooling semantics, provenance separation | No CALPHAD rewrite in this batch |
| 2 | Diffusion | High | `core/metallography_pro/transformation_fe_c.py` | Arrhenius-style diffusional weighting, `diffusional_equivalent_time_s`, family-specific nucleation drives | No PDE diffusion solver |
| 3 | Crystal interfaces and microstructure | High | `core/metallography_pro/morphology_fe_c.py`, `core/metallography_pro/reflected_light.py`, `core/metallography_pro/validation_pro.py` | Coherence/growth-mode ideas translated to allotriomorphic vs Widmanstätten ferrite and downstream signals | No explicit interface-energy minimizer |
| 4 | Solidification | Medium for repo, low for current batch | Notes only | No | Outside current Fe-C solid-state batch |
| 5 | Diffusional transformations in solids | Critical | `core/metallography_pro/transformation_fe_c.py`, tests | Additivity-style surrogate, TTT/CCT-inspired family competition, hardenability shift, ferrite/pearlite/bainite weighting | No direct TTT/CCT atlas interpolation yet |
| 6 | Diffusionless transformations | Critical | `core/metallography_pro/transformation_fe_c.py`, `core/metallography_pro/morphology_fe_c.py`, tests | Martensite subtype semantics, derived tempered labels, provenance split | Koistinen-Marburger still remains from prior source, not replaced by a new chapter-6-specific law |

## Metadata mapping

| Metadata block | Porter/Easterling influence |
| --- | --- |
| `continuous_transformation_state` | hardenability, continuous-cooling shift, diffusional equivalent time, nucleation drives, family semantics |
| `kinetics_model` | chapter-level source trail and exact mirrored kinetics values |
| `morphology_state` | pearlite/ferrite/bainite/martensite family outputs and mirrored selectors/progress values |
| `provenance` | explicit separation of thermodynamics, diffusion, interface growth, diffusional transformations, diffusionless transformations |

## Test mapping

| Test file | What it now proves |
| --- | --- |
| `tests/test_pro_scheduler_progress.py` | progress/exposure monotonicity, leakage guards, metadata key shape |
| `tests/test_pro_transformation_and_morphology.py` | real selector semantics, ferrite split, bainite split, provenance completeness |
| `tests/test_pro_surface_and_reflected_light.py` | downstream render/validation propagation of new morphology signals |
| `tests/test_pipeline_v3_pro_mode.py` | cross-block metadata consistency and absence of stale key names |
| `tests/test_pro_ui_presets.py` | LR1-safe request path remains intact after kinetics/morphology updates |
