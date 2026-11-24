# PulmoSIMU · E1–E4 integrated research engineering release record

**Date:** 2026-11-24 · **Branch:** `feature/mechanistic-lung-foundation` · **Target:** canonical original-engine PulmoSIMU XR · **Status:** SOFTWARE RESEARCH CANDIDATE; physical Quest / scientific release gates OPEN.

This is the finite E1–E4 implementation pass requested for the ten-capability immersive laboratory, **not a declaration of clinical validation or headset acceptance**. No second lung renderer, invented source-derived airway, replacement scope state, or duplicate respiratory solver has been introduced.

## Delivered by gate

### E1 — spatial architecture and controlled anatomical interaction

- Preserves XR-13 architectural room/studio/focus states, original mesh, guided scope, expert world-space panels, hand/controller source ownership, near-surface contact, bounded translation and scale, undo/redo and recenter.
- The existing yaw/pitch/roll precision selection now has a fourth option, **ORIENTATION LOCKED**, reached by cycling the axis control. During one-/two-hand grabs the lung can move/scale while held orientation stays fixed. Rotation locking does not teleport the headset or mutate geometry. The lock resets on new XR session or recenter.
- Controller and hand behavior still require real device trials; browser tests cannot certify pose stability or comfort.

### E2 — real CT sections in the same XR source frame

- Added `experimental/xr/ct-spatial-mpr.js` to the canonical engine and reused `experimental/anatomy/nifti-source.js`: **actual HU voxels** in axial, coronal and sagittal planes, positioned with the original NIfTI RAS sform and the exact `model_from_ras_mm` study transform.
- Added **Load verified CT volume** to the existing CT research toolbar. Select the exact original **uncompressed .nii** study after choosing its matching research GLB and JSON manifest. The browser checks NIfTI source SHA-256, GLB SHA-256 and sform/HU metadata. A mismatch blocks any XR coframing.
- In immersive XR, choose **EXPERT TOOLS → CT SECTIONS**, **SELECT PLANE**, **SLICE - 1 / + 1**, **HIDE/SHOW MPR**. Real voxels appear on actual planes attached to the same lung source transform. A visible plane intercepts pointer hits and blocks anatomy grabs through CT pixels.
- Sections use local max-512-pixel textures per plane and dispose textures, meshes and buffers when exiting XR. This is a memory/efficiency choice, not benchmarked Quest performance.
- This is CT image registration and visual co-location under verified source metadata; it **does not validate mask boundaries, fissures or actual airway geometry**. DICOM direct ingestion and on-device source-file logistics are separate gates.

### E3 — explicit scientific domains and CT-safe procedure limits

- Added `experimental/xr/science-integrity.js`: original reference-only R3 physics, hash-verified but unvalidated CT image domain, block on CT-specific bronchoscopy, source-independent tissue validation and a fail-closed 4D motion metadata contract.
- The original `experimental/core/breathing-network.js` remains the only XR reference respiratory solver. A CT source cannot acquire invented pressure, ventilation, wall contact, force or tissue deformation by opening a menu.
- CT MPR has actual image voxels but no manufactured vessel, bronchial lumen, pleural layer, 4D motion or CFD. Existing desktop source measurements and hash-bound fissure hypotheses remain unchanged.
- Real 4D respiratory mechanics, anatomy source masks, clinical/endoscopic lumen verification and independent physiological calibration require external datasets and measurement-based acceptance; these are **not implemented as fake defaults**.

### E4 — software tests, evidence collection and explicit freeze blockers

- Added `experimental/xr/acceptance-gates.js`: the XR report lists measured/software observations separately from **Quest physical testing**, independent scientific validation, GPU-frame timing and motion-to-photon latency, which remain unmeasured.
- Integrated evidence into `pulmosimu-e1-e4-session-report.json`, including source domain, three CT section indices and RAS coordinate when available, anatomical comfort bounds, original monitor state and locked-orientation state.
- Added `experimental/tests/e1-e4-integrated.test.mjs` (hash/affine gate, true voxel coordinates, model separation, orientation-lock regression, physical freeze blocked) plus an expanded Chromium smoke test that constructs real CT section planes in-browser and checks disposal.
- Retained all existing original scene, CT Inspector, XR-12–XR-15, guide and numerical tests. Historical report-label and axis-count tests were updated to assert the new functionality rather than prevent it.

## Hands-on verification

**Desktop development:** checkout `feature/mechanistic-lung-foundation`; run a repository-root HTTP server such as `python -m http.server 8080`; open `http://localhost:8080/`. The active preview deployment must be checked against the commit SHA; a legacy URL is not proof of deployment.

1. From original workstation choose GLB + matching JSON together (research source).
2. Select **Load verified CT volume**, choose its original `.nii`; confirm the exact hash, dimensions and header match. Bytes remain in local browser memory. A wrong CT must fail.
3. Use desktop **Lobe study** for provisional atlas editing and source-based annotations, or open separate source inspector for MPR and full CT QA.
4. On **HTTPS Quest-hosted** branch preview, load the study into the Quest browser context (or use an approved source-transfer workflow). Enter the original procedural XR, choose Expert Tools → CT Sections, change planes and indices, hide/show. Return to reference mode for original bronchoscopy; CT-specific procedure remains blocked.
5. In XR, test world-anchored studio/focus, anatomy near-grab, controller/thumbstick movement, two grips, undo/redo, axis choice to ORIENTATION LOCKED, tracked hand loss, recenter and leave/return to session. Export E1–E4 session report after exiting.
6. Execute `cd experimental && npm test` and inspect GitHub Actions browser job. **Source and real headset checks are separate.**

## Final gates: honest exit status

| Gate | Software deliverable | Acceptance evidence still required |
|---|---|---|
| E1 | Room, original engine, bounded 6DoF, orientation lock, reversible controls | Physical Quest 2 stereo/hand/controller/comfort acceptance |
| E2 | Matched local NIfTI, RAS-to-model, three real XR MPR planes, clipping interaction isolation | Exact source study browser review; physical Quest texture/legibility/performance; full source mask QA |
| E3 | Reference R3/domain segregation, original bronchoscopy, fail-closed scientific claims | Independently validated lumen/lobe/vessel data, numerical calibration and measured instrument constraints; 4D data when available |
| E4 | Automated checks, research-only session report, conservative freeze evidence | Actual Quest 2 endurance, GPU/frame timing, headset version and clinical-scientific approval; deployment parity |

**Decision: DEVELOPMENT PASS COMPLETE ONLY WHEN CI IS GREEN. RELEASE FREEZE WITHHELD.** This remains a coherent expert research prototype; it is not yet a patient-specific validated digital twin, an approved medical device, a clinically accurate bronchoscopy trainer or an accepted Quest hardware release.
