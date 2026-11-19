# PulmoSIMU R6 — Prototype Integration & Verification Gate

**Status:** R1–R5 experimental software, subject to GitHub CI. This document is a *software prototype freeze*, **not** anatomical, physiological, pharmacological, or clinical validation.

## Canonical boundaries

- The existing PNEUMO·VR advanced procedure remains implemented in root `index.html`, with its original navigation, gesture controls and bronchoscopic experience. R1–R5 do **not** rewrite it.
- The CT/DICOM processing implementation remains in `pipeline.py`, `server.py`, `synthetic.py`.
- The new modular experimental foundation remains under `experimental/` and is deliberately not linked from production navigation until a clinical/engineering review authorizes integration.
- The original airway reference is a **versioned snapshot** with 18 proximal/intermediate branches, not complete human distal lung anatomy.
- The 3D lobar envelope is schematic; it must not be described as anatomically exact.

## Verification matrix

| Check | Automated gate | Limits |
|---|---|---|
| Anatomical graph connectivity and lobe mapping | Node tests | Source procedural snapshot, not CT registration |
| Numerically finite simulated pressures, flows, tidal volumes | Node tests | Uncalibrated reduced-order linear RC approximation |
| Network mass-flow conservation | Node tests | Does not establish in-vivo validity |
| Disease localization and mechanism state isolation | Node tests | Simplified exploratory mechanisms |
| Treatment counterfactuals, zero-effect invariants, structural tumor preservation | Node tests | Assumed pathway sensitivity, not efficacy |
| Original procedure source unchanged versus `origin/main` | GitHub source-diff guard | Doesn't prove procedure works at runtime |
| Browser opens experimental lab, airway data, overlays, treatment and output changes | Chromium/Playwright smoke | Requires network access for Three.js CDN, tests functionality not image quality |
| Original procedure page basic DOM load, original view canvases present | Chromium/Playwright smoke | Not a full interactive bronchoscopic procedure regression |
| Three-dimensional anatomy and shading visual quality | **Manual pending** | Independent expert inspection required |
| Patient DICOM pipeline on real de-identified reference CT | **Not yet performed** | Segmentation dependency verification, registration and geometry units required |
| Literature-calibrated physiological/PKPD parameters | **Not yet performed** | Never use prototype output to guide treatment |

## Freeze definition

The initial **R1–R6 experimental software prototype** may be marked *engineering freeze candidate* after all automated CI jobs are green, verified browser capture inspection, and manual original-procedure regression. It is *not yet valid* to mark the platform as a scientific/clinical digital twin.

## Scientific gates before further disease expansion

1. Acquire licensed/consented or de-identified reference thoracic CT with segment/lobe contours and consistent patient-coordinate orientation, physical voxel spacing and provenance.
2. Validate segmentation/canonical anatomy mapping against imaging reference; distinguish directly imaged geometry from algorithmically inferred distal structures.
3. Quantify model sensitivity to airway radius, compliance and distal resistance assumptions; compare calculated waveforms with representative bench or clinical reference measurements.
4. Replace demonstrative phenotype/drug parameters with curated, indication-specific literature-supported evidence including appropriate uncertainty. Never infer individual efficacy from a pathway label.
5. Freeze model versions, validation datasets and regression outputs before publishing performance claims.

**After this software freeze, further work should proceed as explicitly scoped and independently testable increments, rather than an unbounded phase sequence.**

## Post-R6 engineering validation increment — anatomy provenance and sensitivity

This increment **does not reopen** the finite R1–R6 feature roadmap; it addresses two outstanding verification gates.

1. `anatomy/ct-metadata-adapter.js` interprets the original PNEUMO·VR `pipeline.py` JSON metadata format, checks positive scaling, validates a single rooted tree, normalizes arbitrary source airway IDs, and computes **metres per display-scene unit as `1 / transform.scale`**. It includes reversible scene⇄patient-mm coordinate conversion. Input patient identifiers and pathology metadata are **not propagated** into the adapter output. There is no automatic client-side patient upload or external sharing.
2. `core/sensitivity-analysis.js` runs an explicit one-factor-at-a-time ±20% numerical stress test of assumed scene scale, gas viscosity, unmeasured distal resistance and respiratory compliance, ranking the effect on modeled tidal volume. The experimental canvas exposes an **on-demand** sensitivity inspector so it does not add unnecessary visual or runtime burden.

**Important:** The source reference model is *still not converted to measured anatomy*. CT metadata can be mapped to measured SI units **only when the source patient-derived transform exists and its scale is trustworthy**. The R3 solver continues to use reference parameters until a separate, independently reviewed CT-specific region-mapping adapter is built. The R2 graph's fixed-ID lobe mapping is intentionally NOT reused for patient-derived CT IDs. This avoids silently assigning an incorrect lobe or calculating a misleading physiological result.

The sensitivity result is a mathematical stress test around assumed inputs, **not a clinical confidence interval** or validation against spirometry. Its purpose is to reveal dependence on modelling assumptions before anyone interprets simulated changes.
