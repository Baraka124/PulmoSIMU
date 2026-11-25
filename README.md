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

## XR-19 renderer and simulator input integration

The original workstation now pins Three.js r147 for WebXR projection-layer support. The native quad adapter copies the original endoscopy render target to a granted XR quad layer; the same-context mesh remains the fallback. Native pixels are only reported active after a successful GPU copy. Actual Quest acceptance remains required. The original workstation also offers optional local serial setup and measured calibration import, manually armed simulator-only control and a left-grip dead-man. No physical commands, patient-specific control, or clinical certification are provided. Device sampling, axis orientation, latency and calibration must be independently measured before acceptance. Use the official root domain only.

## XR-20 physical acceptance recovery

Corrected the original clearance gate's undefined diameter variable, added explicit forward/retract controls to the spatial guide, hand pinch menu actions and remote/near anatomical grabs, closer default anatomy, MR local-space entry and explicit manual placement when real hit testing is unavailable. Native compositor activation is disabled for this compatibility test release; source endoscopy remains available. Physical retesting is mandatory.

## XR-21 inspection and optics

Reduced reference mucosal specular glare and normalized optical falloff and scope offset to model display scale. Three light profiles, original source route breadcrumb and cumulative reference path distance, all source junction choices, and reversible individual-lobe inspection using the existing five reference meshes. Reassembly is automatic before bronchoscopy and on XR exit. This is a visual reference exploration mode, not physical tissue separation or clinical anatomy validation.

## XR-22 continuous reference lumen

A bounded implicit union of the existing tapered centerline pieces provides one continuous endoscopic viewing surface, removing overlap walls at bifurcations and trifurcations. Original source graph, radii, scope kinematics and external anatomical meshes are unchanged. Shared optical rendering across desktop and XR, with automatic fallback to original tubes on construction/budget error. Terminal closures are numerical reference boundaries, not inferred distal airways. Headset rendering and orifice appearance require physical acceptance.

## XR-23 reference video bronchoscope

A generic dimensionally referenced flexible video scope adds a shaped control body, lever, suction valve, channel port, flexible 600 mm display shaft and optical tip (4.2 mm distal OD, 4.1 mm insertion OD, 2 mm channel reference, 110 degree forward view; Olympus BF-P190 published specification https://olympusmedical.asia/products/pulmonology/bronchoscopy/endoscopes/video-bronchoscopes/bf-p190/index-en.html). Housing geometry is illustrative, not manufacturer CAD. The original camera now uses a roll-coupled distal frame and aspect-correct horizontal field-of-view convention. In VR, the right controller holds the instrument; a left-grip relative-motion clutch feeds/retracts the original scope, while original sticks remain available. No physical actuation, measured shaft mechanics, haptics or clinical validation.

Manufacturer reference also specifies 2–50 mm depth of field; this release records it but does not simulate measured lens MTF, focus blur or sensor response. Published field of view is modeled as horizontal, an explicit simulation convention. https://content.medical.olympusamerica.com/products/diagnostic-bronchoscope-bf-p190

## XR24–26 device research pass

XR24 adds a depth-texture-driven focus approximation (reference 2–50 mm), bounded radial distortion and static luminance response in the XR monitor, with original image fallback on non-WebGL2 devices. XR25 displays an inserted shaft constrained to the original source route, and reports source-field geometric margin and estimated curvature. It does not solve forces or tissue deformation; the original navigation remains authoritative. XR26 adds a 2 mm reference working-channel visual mechanism for forceps, brush and suction indication. Deployment and jaw movement are bounded; no actual biopsy, fluid removal or clinical sampling is performed. Headset acceptance and bench calibration remain outstanding.

## XR-27 / HIL-01 parallel start

Local 20 Hz bounded procedure observation records reference segment bins, within-segment travel, movement-speed summaries and estimated shaft-state sample counts. No visual surface coverage, clinical score, contact force or raw trajectory export. HIL-01 preliminary passive training handle concept is stored in hardware/hil-01 with parametric OpenSCAD, dimensional design manifest and manufacturing gates. CAD not yet rendered or printed.
