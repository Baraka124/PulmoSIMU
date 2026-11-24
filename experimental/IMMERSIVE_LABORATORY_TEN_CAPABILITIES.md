# PulmoSIMU — Immersive Laboratory / Ten-Capability Completion Contract

**Adopted:** 2026-11-24. **Active branch:** `feature/mechanistic-lung-foundation`.

This specification governs future development and acceptance of the **one canonical PulmoSIMU anatomical, procedural, computational and XR environment**. It is not a new viewer or permission to replace original work. Reuse `index.html` and `experimental/xr/procedural-immersive.js`, established instrumentation, medical-image inspector and R3 numerical modules. Existing `experimental/UNIFIED_SIMULATION_ARCHITECTURE.md`, source QA, and XR-12–XR-15 preservation gates remain authoritative.

## Non-negotiable architectural principles

- **The lung is primary.** Spatial UI stays contextual, restrained and recoverably hidden; no fake clinical dashboards or decorative physics particles.
- **One source of truth.** Reference procedural anatomy, independently registered CT surfaces, image voxels and numerical estimates are separate labelled data domains, never presented as interchangeable.
- **Source-first semantics.** No invented CT vessels, bronchi, fissures, pleura, tumor or airways; missing/uncertified geometry is clearly absent or labelled provisional.
- **Precise interaction.** Near-grab requires actual surface proximity; selection, handling, anchored positions, undo/home and two-hand gestures have bounded and reversible states.
- **Clinical and engineering claims are gated.** A passing browser check cannot certify Quest 2 performance, CT validation, CFD, instrument safety or a clinical product.
- **Finite release.** Engineering implementation, source validation, physical Quest testing and a freeze decision are distinct outputs, not endless feature additions.

## Ten product capabilities and explicit acceptance criteria

| # | Capability | Current verified foundation (24 Nov 2026) | Required exit criterion |
| --- | --- | --- | --- |
| 1 | Architectural immersive laboratory | XR-13 original-engine studio/focus stage, spatial panel and quiet world-space datums implemented | Real Quest session proves stable room-scale spatial depth, legible context controls, no head-locked room/organ, no obstructions |
| 2 | Precision 6DoF anatomical manipulation | XR-12/XR-15 controller/hand ownership, bounded near-grab and two-hand transform software plus undo/recenter | One/two-hand poses, constrained rotation, reliable dropout recovery and comfortable size/position demonstrated on physical Quest |
| 3 | Real CT inside immersive space | Desktop NIfTI CT source inspector with MPR, GLB, manifest and registration checks; XR actual CT MPR unverified | Spatial MPR with verified image affine, source plane selection in millimetres, reproducible voxel/mesh alignment, compatible DICOM LPS→RAS when sourced |
| 4 | Semantic anatomical exploration | A2.2 **editable atlas-seed** territory colouring, original reference branches and structure selection; no validated CT lobe/vessel/pleura masks | Validate source-labeled masks, independent mesh identity, isolate/opacity/clip; unavailable labels hidden, not synthesized |
| 5 | Immersive bronchoscopy simulator | Original `index.html` bronchoscopy state and XR-14 source-driven procedure guide/monitor integrated | Verified lumen geometry, steering and clearance measurements, tested tool limits and procedure navigation; clearly separate reference simulator from patient-specific claims |
| 6 | Spatial respiratory physics | R3 conserved reduced-order resistance/compliance solver available for original reference with source/assumption labelling | Attach numerical outputs to matching **verified** source IDs/geometry/units and measured conditions; reserve 3D CFD/FEM for separately validated solvers |
| 7 | Hand-controlled anatomical sectioning | Desktop source inspector has axial/coronal/sagittal clipping and position controls; tracked XR plane not proven | Actual tracked world-space section plane with real source-volume slice/coordinate correspondence, reversible inspection and no fabricated cut interior |
| 8 | Four-dimensional respiratory mechanics | No measured, registered 4D lung motion available; numerical breathing is not tissue deformation | Independently sourced phase-resolved geometry/motion, defined registration/deformation model, conservation/biomechanics checks and quantitative uncertainty |
| 9 | Spatial measurement and research annotations | CT inspector review, A2.3 hash-bound RAS surface markers, import/export and source-affine metrics | Reproducible lengths/angles/landmarks/notes in source millimetres with dataset digests, transform chain, audit trail and export/import invariants |
| 10 | Adaptive XR performance and expert modes | XR-13 focus/studio, expert menus, fixed foveation request and CPU/frame callback reports; source performance on Quest unverified | Measured Quest performance, comfort, object-detail budgets, uninterrupted anatomy/procedure/research modes, recovery and baseline evidence; shared multi-user sessions are a later separate scope |

**Status rule:** software implementation ≠ accepted headset experience ≠ scientifically validated tissue model. Treat all incomplete exit criteria as OPEN even when modules or tests exist.

## Finite completion sequence

### Pass E1 — Immersive interaction and scene acceptance (immediate)
Preserve existing architectural room and focus/studio variants. Audit source-organ placement and release behavior; test orientation locks, bounded axis rotations, undo/home, panel precedence, locomotion safety and physical Quest usability. Export actual session evidence. **Do not invent extra demo scenes**.

### Pass E2 — CT spatial inspection and source integrity
Extend the existing CT/MPR inspector into *the same immersive engine* by a shared immutable study/transform/section state. Confirm RAS/LPS, millimetre geometry, slice orientation, ROI selection, clipping and annotation export parity. Keep binary medical imaging local until provenance, de-identification and publication rights are verified. **No CT-specific procedure until reviewed lumen masks exist**.

### Pass E3 — Semantic, quantitative and procedural integration
Connect source-verified structures to selection and measurements; bind numerical reference solver values only to matching model IDs/units. Review actual bronchoscope lumen, calibrated diameter and contact before calling any instrument navigation patient-specific. Distinguish validated pressure/flow from reduced-order assumptions; reserve 4D tissue motion for a separate evidence-supplied dataset and solver.

### Pass E4 — Performance, review and freeze
Run static source/anatomy tests, Chromium interaction tests, actual Quest 2 seated endurance and controller/hand/mixed-reality checks. Record image/geometry provenance, numerical and measurement uncertainty, reversible transformations, headset comfort, stability and repeatability. Freeze a **research-prototype baseline** only with both software and physical evidence; clinical/device-grade status requires additional independent validation and regulatory assessment.

## Provenance and regression boundary

- Original reference model, clinical procedure, endoscope render and source state: `index.html`.
- XR input/session/adaptive controls: `experimental/xr/procedural-immersive.js` and XR utility modules.
- Source CT desktop lab: `experimental/anatomy-inspector.html`, `experimental/anatomy/inspector-app.js`.
- A2.2 study territories: `experimental/anatomy/lobe-territories.js`; **provisional fissure planes only**.
- A2.3 source surface annotations: `experimental/anatomy/ct-review.js`; study-hash-bound research hypotheses.
- Procedural reference numerical physics: `experimental/core/breathing-network.js`.
- Canonical architecture and acceptance references: `experimental/UNIFIED_SIMULATION_ARCHITECTURE.md`, `experimental/XR12_QUEST_PHYSICAL_ACCEPTANCE.md`, `experimental/XR13_SPATIAL_OBSERVATORY.md`, `experimental/XR14_GUIDED_PROCEDURE_AND_MR.md`.

**Release rule:** Never substitute decorative geometry or effects for evidence. The ultimate product is an expert, image-grounded, interaction-credible pulmonary research laboratory; the current version remains a working, incomplete research system.
