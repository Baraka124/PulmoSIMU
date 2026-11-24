# PulmoSIMU — S3 Integrated Three-Milestone Software Delivery

**Scope:** first functional integration across the imaging foundation, canonical procedure and real R3 reference simulation. **Not** a claim that all source anatomy, hardware and clinical-validation milestones are finished.

**Active branch:** feature/mechanistic-lung-foundation. Public main remains unchanged.

## 1 / Imaging foundation — physical CT measurement

The existing Source Anatomy Laboratory continues to load local NIfTI, GLB and the exact study manifest with SHA-256/header matching, local multi-planar viewer, source clipping, source-bound review marks and fail-closed CT procedure gating.

New source distance controls: **Set A**, **Set B**, **Clear distance**, with both points explicitly selected from the current CT crosshair and visually shown in MPR. Measurement uses the original 4×4 NIfTI voxel-to-RAS affine, including oblique orientation and anisotropic spacing. It returns straight-line physical distance in millimetres; it does not infer a fissure, bronchial path, vessel length or pathology. The workspace resets metrics on every source replacement. **No patient data leave the browser.**

Files: experimental/anatomy/source-review.js; experimental/anatomy/inspector-app.js; experimental/anatomy-inspector.html.

## 2 / Procedural and spatial workstation — one guided rehearsal

Studio remains the shell with the **original** procedural-workspace.html and original-engine index.html XR. New **Guided session** button opens a compact collapsible review panel on the right, separate from anatomy/bronchoscopy canvases. It offers five contextual, navigable checkpoints:

1. Exact CT anatomy review.
2. Original bronchoscopy rehearsal.
3. Quest/XR spatial rehearsal.
4. R3 reference physiology.
5. Session evidence export.

**Open checkpoint workspace** uses existing Studio routes. Users **manually mark reviewed** and can browse the checkpoints; the UI does not fabricate headset telemetry or procedure success. The panel starts collapsed. Navigation between iframed workspaces reloads their local state; no uploaded CT data or scope commands are carried across the shell.

Files: studio.html; experimental/shared/guided-session.js.

## 3 / Computation and scientific demonstration — real solver and export

The guide's **Compute reference R3** button imports the existing 18-branch source graph and uses the pre-existing mechanistic-lung + R3 breathing-network solver for a deterministic ten-second reference simulation under its default assumed physics. It displays approximate tidal volume, peak inspiratory flow and maximum conservation residual. Outputs come from R3, not fabricated visual numbers.

**Export session JSON** captures self-reported checkpoint completion with timestamps and (when run) solver result, source/version, model parameters and explicit unvalidated status. The export is an in-memory, on-demand browser download: no CT, GLB, patient notes, facial/hand tracking, claimed hardware performance or patient-specific physiological data.

Files: experimental/shared/guided-session.js; studio.html.

## Quality gates and finite freeze

Regression coverage: experimental/tests/s3-integrated-milestones.test.mjs and the existing Chromium browser-smoke.mjs now exercise the guided report, R3 result and source metric initial gates. The original index.html, bronchoscopy interactions, Quest original-engine routing and clinical source gate were not rewritten.

**Software acceptance:** stable source measurements; real numerical run; guided workspace navigation; direct client JSON export; original tests pass.

**External gates not claimed complete:** reviewed CT lobe/vessel/lumen masks and quantitative segmentation accuracy; instrument–airway wall collision verification; geometry-convergent Quest LOD; measured true procedural kinematics; Quest 2 physical headset comfort, interaction and performance; validated 0D/1D/3D flow and tissue mechanics. These require independently sourced geometry, physical measurements and explicit expert approval.

**Important:** this is an operational three-domain integration increment; it does not falsely mark all ten long-range platform improvements as finished.
