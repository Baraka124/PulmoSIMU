# PulmoSIMU UI and integration audit — recording 2026-11-20

## Evidence
User-submitted 44-second, 1918x916 PNEUMO·VR walkthrough. Review sampled multiple frames from start to finish, including external/dual bronchoscopy modes, scope gesture/biopsy panel, floating airway popover, surgical-planning sidebar and splash/reinitialisation.

## Findings from the demonstrated procedural application
1. **Anatomy occupies too little effective stage space** compared with the width of the display. In several sections, the lungs occupy a small cluster toward the center while both overlay panels and surgical planner draw attention.
2. **Competing navigation hierarchies**: topbar mode/stats, floating left gesture/navigation tools, floating anatomical popovers, right surgical planner, overlay readouts and scope status appear together. The user has no single obvious locus of attention.
3. **Low text legibility** due to extreme microtypography, faint labels, many narrowly separated numbers and high reliance on color in red/teal/purple micro-badges.
4. **Inconsistent view presentation**: switching from external to dual changes allocation of the canvas, while floating tools may cover the working anatomy. The right sidebar stays prominent even when inspection/camera manipulation is central.
5. **Transient overlays compete with task-critical content**: event banners, target labels, anatomical tooltip and gesture prompts can overlap the visual focus.
6. **The new R1–R6 computational work is not on screen**: root index.html is the original procedural system; new models and CT/atlas research are in experimental/lab.html, not synchronized with root state.

## What has now been implemented
- `studio.html` is an **opt-in** neutral workspace entrance with two explicit modes: original procedures, and mechanistic research. It does not simultaneously mount two full 3D experiences or place one project's controls on another's canvas.
- Workspaces share a discoverable top-level navigation and editorial shell. This **is navigation integration, NOT a shared state or physiology integration**; swapping workspaces reloads the iframe. No anonymous pass-through of patient files or simulation state.
- The original `index.html` (advanced bronchoscopy, navigation, gesture controls, surgical planning, and CT pipeline) remains unchanged.
- Chromium smoke checks assert mode navigation and original page presence.

## Targeted future UI quality pass (not yet implemented)
- **One main 3D stage:** maximize external lung when in external mode; treat bronchoscopy as a deliberate split-mode only.
- **One contextual inspector:** navigation, pathology, gesture and surgical planner functions exposed by task-aware workspace states, without duplicate simultaneous floating panels.
- **Task-critical metadata only:** show the active segment/target/mode in a quiet slim header; hide navigation telemetry until needed.
- **Progressive disclosure:** gesture camera and raw tracking stream should open on request; hide inactive measuring/biopsy options while exploring geometry.
- **Anatomy first:** use framing relative to organ bounding box and screen safe-area, not fixed or globally inconsistent offsets.
- **Typography:** prefer text that remains readable on 1080p laptop or mobile with fewer micro-labels and decorative tracking.
- **No breaking navigation:** preserve all current gesture, biopsy and surgical controls behind targeted interaction regression tests.

## Necessary engineering boundary
The same anatomy/physiology engine should eventually serve procedures and mechanistic research. That requires an explicit versioned shared airway model, physical units and coordinate registration. A shared navigation shell alone does **not** solve it. Do not expose the research surrogate as a validated patient-specific computation, and do not merge into production before manual anatomical and UI quality review.

## Implemented UI quality pass — 2026-11-20

New `procedural-workspace.html` is a **non-destructive presentation layer** around the canonical root `index.html`. `studio.html` routes its Procedural Simulation mode through that layer.

- A compact, readable command bar replaces redundant internal topbar presentation in this workspace; **External / Dual / Bronchoscopy** call the existing `setView()`.
- **Canvas** is the default mode; the original gesture and clinical panels are simply visually hidden, *not removed or reimplemented*. They can be opened as mutually exclusive contextual panels via **Clinical inspector** or **Gestures** controls.
- **Fit anatomy** invokes a camera framing helper in the original procedural rendering context; no modification of geometry, patient data, scope position or simulation physics. Original camera fitting and orbit interaction remain.
- Original microtypography, scanline and especially dense HUD chrome are quieted by view-local CSS. The user can still access every underlying surgical/gesture capability through the original DOM.
- This is not yet a physically shared engine between PNEUMO·VR and `experimental/lab.html`; switching studios does not synchronize state. The root legacy application and all its source scripts remain unchanged.
- A browser smoke test checks existence of both original canvases, the source-level camera helper, focused/clinical/gesture panel transitions, original view switching and Studio workspace navigation.
- The focused wrapper is an **opt-in development branch feature**, not deployed by default to the existing `index.html` URL.

### Quality gates still pending
1. Manual visual inspection of rendered frames from the new procedural wrapper on desktop and mobile.
2. Real gesture-camera navigation and scope/biopsy/measurement procedure regression (automated DOM presence alone is not a full workflow test).
3. User-facing comparison against the recording, checking actual anatomy framing, legible units and whether clinical panels obstruct anatomical interactions.

Avoid merging this branch until these review gates pass. The requested next engineering objective remains one authoritative anatomical and physical state contract shared across procedure and mechanistic research, not merely an iframe shell.
