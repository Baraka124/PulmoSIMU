# PulmoSIMU

**One research platform. One authoritative code branch: `main`.**

PulmoSIMU is an integrated pulmonary research and spatial procedure laboratory built on the original procedural engine. It combines an interactive reference airway model, bronchoscopy, experimental respiratory computation, source-linked CT review, desktop workstation and headset experiences. It is not yet a clinically validated medical simulation or patient-specific digital twin.

## Canonical entries

- **PulmoSIMU launcher:** https://pulmosimu-xr-preview.vercel.app/quest-short/
- **Desktop procedural workstation (no Quest or webcam required):** https://pulmosimu-xr-preview.vercel.app/index.html?xr=procedural&entry=desktop
- **CT source / MPR inspector:** https://pulmosimu-xr-preview.vercel.app/experimental/anatomy-inspector.html

Older Vercel aliases, including `pulmo-e4-preview.vercel.app` and `pulmovr-chi.vercel.app`, are historical compatibility addresses, not separate release targets. The single production project is **`pulmosimu-xr-preview`**. The earlier `pulmovr` project is obsolete and subject to permanent deletion in the Vercel dashboard.

## Authoritative implementation

- `index.html`: original airway, clinical reference procedure, scope state, desktop controls and WebXR adapter installation.
- `experimental/xr/procedural-immersive.js`: VR/MR session lifecycle, controller/hand tracking, spatial UI, original camera feed and procedure station.
- `experimental/xr/bronchoscopy-station.js`: bounded XR-16 reference instrument controls, configurable endoscopic monitor quality and spatial layouts.
- `experimental/anatomy-inspector.html`: local CT / NIfTI / GLB/manifest source review and multiplanar section visualization, separate from procedural validity claims.
- `experimental/core/`: reduced-order computational research models, **not** automatically coupled to patient CT.
- `experimental/tests/`: numerical, source, controller, browser and procedure regression tests.
- `experimental/UNIFIED_SIMULATION_ARCHITECTURE.md`: core source and scientific boundaries.
- `experimental/XR17_REALITY_CONTINUUM_ARCHITECTURE.md`: proposed future capability-negotiated VR/MR/AV roadmap. This is **not** a completed feature.

## Development and release policy

1. All changes are integrated in `main`; do not release from a feature branch or spin up a competing Vercel project.
2. Preserve the original procedural engine and CT source provenance. Never equate CT lung-shell meshes with validated bronchial pathways or attach uncalibrated physiology to a patient study.
3. Run the existing GitHub Actions model and browser checks before deploying the exact `main` SHA. The Vercel automatic Git integration is **not yet independently confirmed**; manual API-linked builds currently require checking their source SHA.
4. Keep patient or private CT source files local; regression tests use synthetic fixtures.
5. Preserve archived experiments as source history/test fixtures, not public flagship applications. Delete archival assets only after dependency and regression review.
6. Physical Quest interaction and performance must be verified separately from browser and source-code tests.

## Current release

**Desktop/Quest software baseline:** XR-16. **Continuum architecture design:** XR-17 (planned). Hardware-validated medical device claims, stable physical MR anchoring, depth-based room occlusion and verified patient-specific endoscopy are **not available**.

The repository documents and code define a single product boundary. Compatibility domain aliases and the temporary cleanup branch are administrative remnants, not new products.
