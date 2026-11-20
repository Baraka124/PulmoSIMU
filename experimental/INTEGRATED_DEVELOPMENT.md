# PulmoSIMU — Integrated Development Contract (ID-01)

**Status:** Active integration branch `feature/mechanistic-lung-foundation` · 20 November 2026.

This document is the repository-level contract for future feature development. Do **not** build another disconnected anatomy or headset demo. Every feature must declare its anatomical source, reference frame, units, selection semantics, dependencies, and integration test before being exposed in Studio.

## 1. The single user journey

`PulmoSIMU Studio` (`studio.html`) is the single front door.

| Workspace | Canonical code | What it owns | Shared state |
| --- | --- | --- | --- |
| Procedures | `procedural-workspace.html` → existing `index.html` | Original PNEUMO·VR scopes, surgical view, camera, instrumentation | Lobar reference focus read from the original reference branch hierarchy; do not change clinical procedure state remotely |
| Mechanisms | `experimental/lab.html` + `core/` | Disease, pressure-flow RC research network, therapy counterfactuals, CT and surface inspection | Reference lobe only when original 18-branch reference graph is active |
| Spatial anatomy / Quest | `experimental/vr.html` + `xr/` | Controller handling, spatial inspector, anatomical isolation and reference airway guide | Same canonical reference lobe in Studio; a validated reference-lobe URL when opening a standalone headset tab |

### What works now

- A single **versioned procedural-reference focus** through `reference-focus.js`, passed using trusted same-origin `postMessage`. The Studio parent is authoritative and checks the message's `origin`, currently active iframe `source` and declared workspace source.
- Five explicit reference lobe IDs: `RUL`, `RML`, `RLL`, `LUL`, `LLL`. Procedural branch identifiers can be projected onto these lobes only in the canonical 18-node reference graph.
- When changing from Procedures → Mechanisms → XR and back, accepted focus is retained. In XR, desktop or trigger selection publishes it to Studio. XR's selected structure also follows an incoming Studio focus.
- Standalone Quest mode accepts only `?focus=<canonical-lobe>`, never patient metadata, anatomy bytes, disease state or procedure state. The in-headset **Procedures** action exits the immersive session before returning to Studio with `?mode=procedure&focus=...`.
- Within XR, the existing validated **topology** snapshot supplies the reference-airway hierarchy; geometry and physiological measurements remain explicitly **not validated**.
- `experimental/tests/workspace-route.test.mjs` and the augmented `browser-smoke.mjs` test the integrated route and trusted workspace focus handoffs.

## 2. Source and provenance boundaries — no false fusion

| Source | Implemented adapter | Can share the reference-focus bridge? | Limit |
| --- | --- | --- | --- |
| PNEUMO·VR original 18-branch procedural source | `anatomy/reference-airways.json` + `anatomy/airway-graph.js` | **Yes**, only from reference mode | Original scene units are not calibrated SI geometry |
| CT centerline metadata | `anatomy/ct-metadata-adapter.js` + `ct-geometry-validation.js` | **No** until independent lobe/segment registration is checked | Alignment and physical units are necessary but insufficient for anatomical correspondence |
| External surface atlas (lobes/vessels) | `anatomy/surface-atlas.js` | **No** until coverage, mapping, source rights, frame, dimensions and registration pass | A named GLB object is not proof of complete anatomy or vascular fidelity |
| Reduced-order physiology and therapeutic experiments | `core/breathing-network.js`, `disease-engine.js`, `therapeutics-engine.js` | Reference region selection only | Uncalibrated demonstration resistance/compliance parameters; not CFD, patient predictions or clinical guidance |
| Surgical/procedural instrumentation | Original `index.html` | **Read-only lobe focus** | No automatic intervention, scope movement, biopsy or procedure annotation transfer |
| Microscopic / vessel / new medical-device modules | Future controlled adapters, potentially from other repos | **Not yet** | Must declare physical dimensions, mesh topology and transforms; never infer registration by filenames or colors |

**Critical:** UI integration is not numerical solver integration or physical registration. Do not label separate objects, unmeasured ellipsoids or unverified imported meshes as a unified clinical digital twin.

## 3. Required engineering pattern for every new capability

1. **Source contract**: immutable source ID, schema revision, origin, units, coordinate frame, calibration/registration status, rights/provenance and geometry coverage. Missing source evidence must be visible as missing, not replaced with synthetic content without labeling.
2. **Core or adapter**: implement a small importable module below `experimental/anatomy/`, `core/` or `xr/`. No isolated HTML application with a second untracked anatomy model. Reuse reference airway IDs **only** for the canonical graph.
3. **One Studio entry**: expose the capability within an existing workspace or register a new explicit workspace in Studio. A direct Quest URL is a headset entry into the same project, not another product.
4. **Typed focus / experiment scope**: only small, validated reference-navigation focus crosses workspaces today. A future research session may add explicit source IDs and deterministic experiment config **after** schema and registration tests. CT voxels, identifiers, patient records and clinical decisions must not cross this bridge.
5. **Regression gate**: preserve `index.html` and CT pipelines, run `npm test`, browser smoke, and physical Quest validation for XR changes. No merge to the original live site before the gate.

## 4. Finite integration plan

- **ID-01 · Common workspace and focus (implemented)**: Studio includes Procedures, Mechanisms and XR, with canonical focus transfer and Quest handoff. Acceptance: automated route/focus/browser smoke passes.
- **ID-02 · One anatomy-source interface (next)**: strengthen source schemas and source registry; enable a common source loader to feed 2D, VR and analysis **only when the required geometry, units and frame are present**. Work towards a geometry-backed CT/vessel VR adapter; do not silently reuse procedurally invented dimensions.
- **ID-03 · Coupled instrumentation and mechanics with evidence (future gate)**: verified position/branch mapping, shared telemetry and source-appropriate numerical coupling; acceptance includes traceable measurements, uncertainty and hardware validation. Freeze the interaction baseline after a human Quest 2 validation, then advance science separately.

## 5. Cross-repository rule

Projects such as separate thoracic simulation, heart/vascular modeling, medical-device CAD, CT pipelines or AI segmentation must not be bulk copied into this repo. Introduce a **versioned adapter**, source manifest and explicit rights/provenance. Only integrate a module after identifying its owner/source of truth and validating units, orientation, registration and CI regression. This prevents the same anatomy and physics logic from diverging across products.

## 6. Deployment

Work on `feature/mechanistic-lung-foundation` and PR #1 until integration and headset gates pass. Preview deployments are for nonclinical testing. The original public `main` route is not the current integrated workspace.

Reference links: `SHARED_STATE_INTEGRATION.md` (protocol), `QUEST2_PILOT.md` (headset), `R6_VERIFICATION_AND_FREEZE.md` (numerics), and `VIDEO_UI_AUDIT_2026-11-20.md` (visual audits).
