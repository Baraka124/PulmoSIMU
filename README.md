# PulmoSIMU

**One research platform. One authoritative code branch: `main`.**

PulmoSIMU is an integrated pulmonary research and spatial procedure laboratory built on the original procedural engine. It combines an interactive reference airway model, bronchoscopy, experimental respiratory computation, source-linked CT review, desktop workstation and headset experiences. It is not yet a clinically validated medical simulation or patient-specific digital twin.

## Canonical entries

- **One canonical PulmoSIMU entrance:** https://pulmosimu.vercel.app/ (the old `/quest-short/`, `/quest.html` and `/quest4.html` shortcuts redirect here)
- **Original desktop procedural workstation (no Quest or webcam required):** https://pulmosimu.vercel.app/index.html?entry=desktop
- **CT source / MPR inspector:** https://pulmosimu.vercel.app/experimental/anatomy-inspector.html
- **Clinical and physical verification lab (local-only):** https://pulmosimu.vercel.app/experimental/validation-lab.html

Historical Vercel aliases are not release targets; use only the canonical entrance above. The single production project is **`pulmosimu`**. The earlier `pulmovr` project is obsolete and subject to permanent deletion in the Vercel dashboard.

## Authoritative implementation

- `vercel.json` routes `/` to the retained original launcher at `quest-short/index.html`, with the legacy Quest shortcuts redirected to `/`. The launcher no longer carries XR09/XR14 query labels.
- `index.html`: the single original airway, clinical reference procedure, scope state, desktop controls and WebXR adapter installation. Desktop/VR/MR choices open this same engine directly without a webcam prompt.
- `experimental/xr/procedural-immersive.js`: original-engine VR/MR session lifecycle, controller/hand tracking, spatial UI, original scope camera feed and procedure station.
- `experimental/xr/xr-capabilities.js`: XR-17A capability registry distinguishing desktop availability, VR/MR support, requested optional session features and verified grants. MR requests optional hit-test and anchors without assuming availability.
- `experimental/xr/mr-placement.js`: XR-17B original-scene spatial MR placement from actual WebXR hit-test results, explicitly confirmed surface placement, optional in-session anchor pose correction and manual unregistered preview. No physical depth occlusion, cross-session persistence or patient CT registration.
- `experimental/xr/instrument-replay.js`: XR-17C local, read-only external trace comparison and timeline. The clinical reference scope is never driven by imported measurements.
- `experimental/xr/physical-calibration.js`: fail-closed, operator-armed three-point physical-sensor reference mapping. No coupling to the original scope and no metrological or clinical certification.
- `experimental/xr/tracked-tool-proxy.js`: XR-17C optional right WebXR controller 6DoF visual marker, strictly illustrative and uncalibrated.
- `experimental/xr/display-path.js`: XR-17C compatibility evidence for WebXR compositor layers; the original same-renderer endoscopic feed remains the active display.
- `experimental/xr/bronchoscopy-station.js`: bounded XR-16F reference instrument controls, dynamic source-video CPU safeguard, configurable monitor quality and spatial layouts.
- `experimental/xr/physical-instrument-bench.js`, `clinical-review-gate.js` and `experimental/validation-lab.html`: fail-closed external evidence intake. Does **not** certify devices, clinical fidelity or physical Quest performance.
- `experimental/anatomy-inspector.html`: local CT / NIfTI / GLB/manifest source review and multiplanar section visualization, separate from procedural validity claims.
- `experimental/core/`: reduced-order computational research models, **not** automatically coupled to patient CT.
- `experimental/tests/`: numerical, source, controller, browser and procedure regression tests.
- `experimental/UNIFIED_SIMULATION_ARCHITECTURE.md`: core source and scientific boundaries.
- `experimental/XR16F_CLINICAL_PHYSICAL_INSTRUMENT_ACCEPTANCE.md`: explicit review and physical bench measurements still required before hardware/clinical claims.
- `experimental/XR17_REALITY_CONTINUUM_ARCHITECTURE.md`: XR-17A, XR-17B and XR-17C software integration and finite acceptance gates. Real Quest/device acceptance and clinical validity remain unverified.
- `experimental/XR18_CONSOLIDATION_ACCEPTANCE.md`: release inventory, native-compositor renderer limitation, physical-instrument test plan and Quest acceptance sign-off.

## Development and release policy

1. All changes are integrated in `main`; do not release from a feature branch or spin up a competing Vercel project.
2. Preserve the original procedural engine and CT source provenance. Never equate CT lung-shell meshes with validated bronchial pathways or attach uncalibrated physiology to a patient study.
3. Run the existing GitHub Actions model and browser checks before deploying the exact `main` SHA. The Vercel automatic Git integration is **not yet independently confirmed**; manual API-linked builds currently require checking their source SHA.
4. Keep patient or private CT source files local; regression tests use synthetic fixtures.
5. Preserve archived experiments as source history/test fixtures, not public flagship applications. Delete archival assets only after dependency and regression review.
6. Physical Quest interaction and performance must be verified separately from browser and source-code tests.

## Current release

**Desktop/Quest software baseline:** XR-16F. **Canonical entry + calibration preview:** XR-18 software changes integrated. **Capability-negotiated runtime:** XR-17A implemented. **Spatial MR surface placement:** XR-17B implemented in software. **Advanced procedure/input research:** XR-17C read-only trace review, tracked-controller visual proxy and compositor capability assessment implemented in software. Physical headset acceptance and clinically calibrated instrument coupling remain pending. The physical-instrument evidence reader and clinical review checklist are software, not measured proof. Hardware-validated device claims, stable physical MR anchoring, depth-based room occlusion and verified patient-specific endoscopy are **not available**.

**Cleanup status:** The duplicate development branch was deleted after proving it contained no unique commits. The obsolete `pulmovr` Vercel project remains paused and requires its owner's Delete Project confirmation; GitHub Pages depublication also requires repository settings because the automation token received HTTP 403. Historical aliases are redirected to the canonical production host by the release routing configuration. GitHub automatic deployment in the remaining Vercel project must still be verified in the Vercel Git settings.
