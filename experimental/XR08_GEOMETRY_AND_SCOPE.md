# PulmoSIMU XR-08 — Source-aware anatomy and bounded procedural navigation

**Branch:** `feature/mechanistic-lung-foundation`  
**Application:** original PNEUMO·VR `index.html` with dedicated immersed original-engine adapter  
**Hardware status:** Quest 2 physical acceptance still pending. Browser tests are not a substitute.

## Why this release matters

XR-07 connected a stereoscopic WebXR environment to the **original** procedural scene and bronchoscope engine. XR-08 establishes the minimum source-integrity gates needed to use actual imported CT-derived GLB anatomy inside that environment and removes the cross-GPU-context endoscopy monitor copy.

This does **not** turn procedural geometry into patient-accurate anatomy, and does **not** constitute a validated physical or clinical simulator.

## A. GLB / metadata registration gate

The original PNEUMO·VR input is a local **GLB/GLTF + JSON sidecar** produced by `pipeline.py` or `synthetic.py`. The source loader is preserved. Before replacing the procedural airway graph, XR-08 now checks:

1. The `pneumovr-1.1` sidecar has a finite positive `transform.scale` and finite `transform.centre_mm`.
2. Every airway node has a unique integer ID, finite 3D centerline points and positive radii. Its parent graph has a unique root, no cycles/disconnected branches, and a consistent child map whenever declared.
3. The **actual GLB mesh geometry**, transformed into the original application's `mGrp` source-local frame, has non-degenerate bounds.
4. Centerline points and GLB surface bounds occupy a plausibly common source coordinate frame. A generous tolerance makes this a **plausibility check**—not a rigid alignment metric or segmentation quality measure.
5. For non-synthetic data the source-derived extent must lie within a **broad** engineering plausibility envelope of 2 cm–3 m. This check can reject grossly mistaken units; it cannot verify anatomical correctness.

If any check fails, the application **returns to original reference anatomy without replacing the existing procedural graph**, and displays a rejection warning. Original CT processing scripts are unchanged. The CT audit reports no patient identifiers.

### Transform semantics

The existing pipeline maps physical millimetres to centered display coordinates using `(point_mm - centre_mm) / 1000 * scale`. Thus **metres per scene unit = 1 / transform.scale**.

The XR workbench offers **SOURCE SCALE** only for co-framed **non-synthetic** GLBs. It changes the displayed *mGrp* scale from comfort-fit to the source-derived metre scale, while preserving the model's spatial center. This **does not independently verify** DICOM orientation, patient registration, segmentation, or exact anatomical dimensions.

If CT imaging and patient registration are independently audited later, the same source contract can carry a genuine validated physical frame; no unsupported accuracy badge is shown today.

### Source model categories

- **REFERENCE:** original procedurally generated airway geometry, original surgical pathology scenes, original clinical tools. Dimensions uncalibrated.
- **SYNTHETIC:** matched GLB+JSON can be explored after coframe checks, but no source physical-size toggle.
- **CT-DERIVED / COFRAMED:** a matched GLB+metadata pair has passed structural/spatial plausibility checks and has source-derived SI units. **Still not verified patient registration or clinical segmentation accuracy.**

## B. Bronchoscope clearance check

In XR mode only, the real original `advance()` function and original `enterBranch()` consult a deterministic clearance model. It computes local interpolated reference/CT-source airway diameter from each airway segment's `r0`, `r1` and longitudinal parameter `t`, multiplies by the source's `mmPerUnit`, and compares against the selected original bronchoscope diameter plus a small clearance margin.

- Prevents an XR advance or branch entry into an estimated lumen smaller than the tool.
- Clears positive scope velocity on constraint failure.
- Reuses the original surgical scope, mode and source graph. It does **not** generate a fake instrument.
- Exposes the estimated lumen/tool diameters on the spatial inspector.

**Scientific boundary:** this is a *source-radius fit gate*, not mesh-surface collision, mucosal contact simulation, bending stiffness, pressure/contact force, haptic realism, bleeding prediction or a validated bronchoscope device model. Radii and their sources still require independent validation. Desktop original behavior stays unchanged.

**XRFrame procedural ownership:** while a headset session is active, the original desktop `requestAnimationFrame` no longer advances the scope or updates its camera. The immersive `setAnimationLoop` explicitly calls the original `advance()`, `retract()` and `updateScope()` using headset frame delta, and applies frame-rate-adjusted velocity decay. This prevents throttled or double-integrated scope movement. On VR exit, the original desktop procedure loop regains ownership.

## C. Actual endoscopic imaging in the headset

The monitor now receives the actual original `camScope` image as a `THREE.WebGLRenderTarget` texture in the **same GPU/WebGL renderer** used for stereoscopic VR (`rBro`). This replaces the prior workaround of rendering in a different WebGL canvas then uploading that canvas back into VR.

- Original endoscopic scene and camera; **no recorded placeholder**.
- 640×360 offscreen target; bounded update cadence, initially ~11 Hz.
- Temporary disabling of Three's XR camera override during the auxiliary camera pass, restoring render-target, camera projection and XR state in `finally`. **Three.js r128 requires explicitly rebinding `session.renderState.baseLayer.framebuffer` through `renderer.state.bindXRFramebuffer`** after returning from the offscreen target; otherwise stereo XR can render to the wrong framebuffer.
- On failure, the monitor is flagged **unavailable** and its image hidden; never silently represented as a valid clinical feed.
- `window.pulmoProceduralXRDebug()` exposes `monitorSource`, `monitorStatus`, `monitorFrames`, source audit, scale mode and estimated clearance.

**Hardware gate:** inspect both headset eyes, correct endoscopic perspective, texture update stability, frame rate and color on Quest 2. A desktop browser smoke test cannot certify this render-target behavior in an active WebXR session.

## Finite release gate

1. **Automated source tests:** valid/invalid transforms; disconnected and inconsistent airway topology; coframe and physical extent; lumen clearance, explicit nonvalidation flags, no patient ID leakage.
2. **Original-browser integration:** reference scene still loads; XR adapter mounts; original scopes/pathology retained; unsupported sidecar data fail before mutating `SEGS`; desktop fallback still works.
3. **Quest 2 hardware acceptance:** enter VR, adjust comfort scale, direct grasp/rotation, A toggle between anatomy/procedure, live GPU endoscopy monitor, right-grip steering, target-branch entry, lumen block and source-scale mode (only with appropriately sourced GLB).
4. **Scientific freeze gate:** real imaging segmentation, named structure registration and instrument contact force/mechanics are **not** declared validated until independent datasets, numerical references and physical measurements are available.

## Next and final integration stage before hardware freeze

**XR-09 · Audited anatomical contact and hardware profiling.** Replace approximate box proximity and source-radius clearance with mesh-based contact surfaces and explicit instrument-radius geometry; keep motion conservative and reversible. Add frame-budget logs and handle/controller disconnect tests. **Do not** invent vascular meshes or tissue mechanics where no CT/physical data exists.

Reference: `experimental/XR07_ORIGINAL_ENGINE.md` for the original source integration, and `experimental/XR_PARITY_AUDIT.md` for remaining clinical parity.
