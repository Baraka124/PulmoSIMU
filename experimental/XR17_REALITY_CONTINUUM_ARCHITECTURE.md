# PulmoSIMU — XR Reality-Continuum Architecture (XR-17 Design Record)

**Recorded:** 2026-11-24  
**Repository / release authority:** `Baraka124/PulmoSIMU`, `main`  
**Status (2026-11-25):** **XR-17A software capability registry integrated in `main`**. XR-17B physical MR anchoring and XR-17C physical input/layer improvements remain **planned**. Hardware acceptance remains pending.  
**Product principle:** ONE source-grounded pulmonary simulation engine, several presentation and interaction modes. Never fork anatomy, clinical procedure or numerical solvers by headset modality.

## 1. Technical vocabulary and intended value

The user-supplied Milgram–Kishino-style taxonomy distinguishes assisted reality (aR), augmented reality (AR), mixed reality (MR), augmented virtuality (AV), and virtual reality (VR); diminished reality (DR) and transitions along the continuum are enabling techniques, not compulsory standalone products. These are **experience/design modes**, not a list of browser features assumed to exist.

| Modality | PulmoSIMU use case | Current capability | Missing gate |
| --- | --- | --- | --- |
| Desktop / conventional 3D | CT provenance review, anatomy inspection, baseline endoscopy | Real `index.html?entry=desktop` and local CT inspector | Further clinical and UI acceptance |
| VR | Interactive spatial lung anatomy and reference bronchoscopy lab | Original-engine WebXR `immersive-vr`, real 6DoF input, two-hand scaling, source render-target monitor, XR-16 station options | Quest frame timing, sustained interaction and physical controller/instrument verification |
| MR | Place shared lung/twin on a physical table or bench | WebXR `immersive-ar` capability-gated passthrough preview; **not** a registered room-aware workstation | Real hit-test placement, anchoring, persistence, stable scale, supported occlusion |
| AR | World-registered explanatory anatomy overlays | No separate calibrated AR mode | World registration, safe anchors, position drift checks, readable labeling |
| AV | Bring a real tracked bronchoscope controller, physical hand or instrument into a synthetic VR procedure room | Only optional headset hand/controller pose inputs | Explicit calibrated physical-device-to-virtual-tool transform, latency and motion-error study |
| aR | Minimal non-obstructive persistent scope HUD | Existing world-space menu and guide are **not** a verified assisted-reality workflow | Small optional headset-relative information layer; must not conceal airway or instruments |
| DR | Reduce visual distraction / mask background objects in MR | Not implemented, and low priority | Privacy, reliable masking, avoid false removal of physical hazards |
| Cross-reality transitions | Carry the same anatomy and procedure state when moving Desktop → VR → MR | Shared original engine; separate XR sessions | State continuity and origin/anchor transforms through session restart, explicit fallbacks |

**Correction to simplistic taxonomies:** Modern AR may use 6DoF; MR does not inherently require LiDAR, real-time dense meshes or semantic reconstruction; spatial understanding and occlusion depend on device and browser capabilities. Head-locked overlays are not properly characterized as '0DoF tracking'. Low passthrough latency does not resolve vergence-accommodation conflict. Asynchronous timewarp and spacewarp perform different tasks. Treat numerical latency targets as device-dependent performance engineering goals, not universal certification thresholds.

## 2. Non-negotiable system invariants

1. `index.html` `SEGS`/`SEG_MAP`/`scope` own original reference procedure; `experimental/xr/procedural-immersive.js` owns XR presentation only.
2. `experimental/anatomy-inspector.html` plus NIfTI/GLB/manifest source integrity owns the CT-derived source study. Imported CT-lung surface is **not** automatically a navigable, verified airway tree.
3. Reduced-order R3 and other numerical engines are separate physical models with source IDs, SI units and explicit uncertainty; no source-independent clinical claims.
4. Desktop remains usable without cameras or an XR device. Headset modes are progressive enhancements and fail closed when unsupported.
5. MR world coordinates and source CT RAS coordinates require separate transform chains, never conflated; log origin/anchor coordinate quality without uploading private CT by default.
6. `main` is the single release authority. QA gates precede deployment; previous build URLs must not become new flagship experiences.

## 3. What should change in the architecture

### A — A capability-negotiated scene host

Keep a shared source scene/interaction model; add a **capability registry** that inspects (not assumes) `navigator.xr`, `isSessionSupported`, and requested/available optional features. Features to negotiate separately include `local-floor`, `bounded-floor`, `hit-test`, `anchors`, `hand-tracking`, `layers` and, where supported, `depth-sensing`. A feature must be hidden or degraded when absent. Do not imply Quest 2 can supply dense depth reconstruction merely because it supports grayscale passthrough.

Shared status should report: `environmentMode`, `worldAnchorKind`, `anchorQuality`, `physicalOcclusionMode`, `monitorPresentation`, `inputTrackingSource`, `ctSourceStatus`, `procedureValidationStatus`. Distinguish **requested** features from **actual runtime results**.

### B — Medical-grade presentation discipline, not decorative XR

Keep the endoscopic video plane separate from airway source pixels, but evaluate hardware-available **WebXR quad layers** for legible monitors that are not unnecessarily re-sampled through the stereoscopic scene. Layer support and runtime performance must be feature-detected; preserve the existing same-context `WebGLRenderTarget` fallback and the original camera on any failure. Do not replace renderer/geometry to gain a label.

MR workbench: use a real `immersive-ar` session, plane hit-test when available, a user-confirmed placement point, retained transform on permitted anchor APIs, and a clear 'room registration unavailable' state. Stable tabletop scale and drift measurements outrank high-gloss material styling. Only add occlusion when trustworthy actual physical depth is available; otherwise clearly use overlay.

AV workbench: make external hardware an optional tracked input source. Use separate calibration for tool handle, tip articulation and shaft insertion; instrument kinematics and distal tip contact require bench verification. Passive game-controller poses are not sufficient evidence of tracked physical bronchoscope accuracy.

### C — Source and performance evidence

Measure on a real headset: frame interval percentiles, display readability, shader/monitor cost, controller loss recovery, locomotion comfort, placement drift, 10-minute continuous use and error recovery. At 90Hz the nominal whole-frame budget is ~11.1ms; GPU time must not be invented from CPU wall time. Distinguish user observation from instrumented evidence. Use deliberate adaptive render scale/quality rather than blindly increasing video resolution.

## 4. Finite implementation plan (not endless phases)

**XR-17A — Capability registry + entry correctness.** Implemented in `experimental/xr/xr-capabilities.js`, original-engine `experimental/xr/procedural-immersive.js`, and the canonical `quest-short/index.html` launcher. Runtime queries `isSessionSupported` for VR/MR, invokes `requestSession` directly from the user's click, records **requested** optional features separately from `XRSession.enabledFeatures` (when exposed), and exports runtime evidence with the original session report. Desktop remains unconditional. `local-floor`, `bounded-floor`, and `hand-tracking` are still optional; anchors, depth-sensing, hit-test, and layers are **not** claimed or requested by this stage. Node and Chromium regressions are the software acceptance gate; actual Quest validation is pending.

**XR-17B — Physical MR workbench.** Only on devices supporting required AR APIs: confirmed hit-test placement, optional persisted anchor, explicit reset/reposition, legible independent monitor. Deliver: device-observed placement errors and drift; downgrade cleanly when unavailable.

**XR-17C — Advanced VR/AV procedure station.** Evaluate XR layers for monitor and optional calibrated tracked instrument inputs; improve ergonomics and measured refresh while retaining original bronchoscopy state. Deliver: independently repeatable controller/device calibration and performance test.

**Freeze gate:** all three software tests pass; actual device run supplies minimum comfort, persistence, error recovery and measured source/pose behavior. Record per-device unsupported features, not 'universal XR support'. The patient-specific medical/tissue/device validation program is separate.

### XR-17A concrete integrity contracts

- `desktop:'available'` even when `navigator.xr` is absent.
- `sessions.vr` and `sessions.mr` are determined independently from each browser's `immersive-vr` and `immersive-ar` support checks; a successful VR support check never implies MR support.
- Only the browser can grant optional WebXR session features. A missing `session.enabledFeatures` is reported as `not-reported`, **not** treated as enabled.
- Headset runtime entry is gesture-bound; a separate asynchronous capability probe must not consume the activation required to request a session.
- Without a trusted physical depth source, `physicalOcclusion:'NOT_IMPLEMENTED'`; without placement/anchor implementation, `worldAnchor:'NOT_IMPLEMENTED'` and `spatialRegistration:'NOT_IMPLEMENTED'`.
- The procedural reference `scope` and `SEG_MAP` remain owned by the original engine. No patient CT-specific bronchoscopy is implied by a successful VR/MR entry.
- Each session's capability snapshot is included in the local export and debug diagnostics for reproducibility. It does not include head or controller poses, private CT data or a remote telemetry endpoint.

## 5. External standards and implementation references

- W3C WebXR Device API: https://www.w3.org/TR/webxr/
- W3C WebXR Layers API: https://www.w3.org/TR/webxrlayers-1/
- Meta Quest Browser MR: https://developers.meta.com/vr/documentation/web/webxr-mixed-reality/
- Meta Quest WebXR Layers: https://developers.meta.com/vr/documentation/web/webxr-layers/
- Meta Browser runtime feature detection: https://developers.meta.com/vr/documentation/web/browser-specs/

The taxonomy improves **how we organize, measure and select technologies**; it does not itself improve model fidelity. Clinical anatomy, tissue mechanics and instrument validation remain separate, evidence-driven workstreams.
