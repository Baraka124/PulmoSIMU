# PulmoSIMU XR-13 · Spatial Anatomy Observatory

**Baseline:** canonical original-engine PulmoSIMU XR-12 scene, never a second viewer or generated lung.  
**Branch:** `feature/mechanistic-lung-foundation`  
**Quest entry:** `https://pulmovr-chi.vercel.app/`  
**Physical acceptance:** pending real Meta Quest 2 session.

## What XR-13 changes

- Adds quiet architectural world depth: a static, vertex-colored dark environment and single-draw-call vertical/ceiling orientation datums. This is environmental staging and has **no anatomical or clinical meaning**.
- Two scene presentations: **STUDIO VIEW** for contextual floor rings, plinth, grid and architectural datums; **FOCUS VIEW** for an uncluttered stage with retained subdued floor and slightly enhanced light on the original model.
- Reworks the spatial menu into a principal page (ANATOMY, BRONCHOSCOPY, RECENTER, FOCUS/STUDIO VIEW, EXPERT TOOLS, EXIT VR) and an Expert page (INSTRUMENT, SOURCE SCALE, view mode, RECENTER, BACK, EXIT VR). The original procedure commands and input handlers remain intact.
- Keeps left Y to show/hide the spatial panel, original grab and bounded size constraints, and the shared closest-surface pointer that prevents clicking through the menu.
- Requests fixed foveation 0.35 **only where the Quest Browser or Three.js exposes support**. This is a rendering optimization hint, not proof of a frame-rate target.
- Retains local XR frame/CPU callback data and adds the current scene-presentation and foveation-request state to diagnostics and exit report.

## Headset acceptance: 10 minutes, seated

1. Reopen Quest Browser with a new tab and launch XR-13 from the short Quest entry. Visually verify the environment is a stationary room, not head-locked to the visor; the original lung remains in front at comfortable viewing distance.
2. Point to the left-side spatial menu. Hover highlights and trigger actions must follow the same target. On the main page, select **FOCUS VIEW**. The architectural datums, plinth, grid and floor rings must disappear. The lung must not move, resize or change source.
3. Select **STUDIO VIEW** to restore environmental context. This should not reset or relocate the anatomy.
4. Select **EXPERT TOOLS**. Verify INSTRUMENT, SOURCE SCALE and BACK. Source scale is allowed only with eligible co-framed source metadata; otherwise it must remain blocked. BACK restores the main menu.
5. Use left Y to hide/reopen the panel; right grip or trigger to grab the lung; both grips and thumbsticks for deliberate movement/scale. Release preserves placement. Exit and export the local session report.
6. Switch to BRONCHOSCOPY; verify the original source feed, scope navigation and tool interactions. The stage must not obscure the endoscopic monitor.
7. Check comfort and stability. Look for double vision, frame drops, controller lag and text legibility. The moderate fixed-foveation request should not make the panel unreadable or cause abnormal peripheral blur.

## Acceptance gates

- No fake anatomical or physiological model is added.
- No automatic CT conversion or source mesh colour reinterpretation occurs.
- Scale is bounded to the established 0.22–0.92 m comfort span, not a patient-calibrated organ dimension.
- All mode controls select the nearest actual surface and are accessible without a persistent HUD.
- Automated tests plus physical headset review are required before a freeze or merge.
- `qualityStatus` indicates **requested or unsupported** foveation. Do not call it measured display FPS.
- Maintain a stable original-engine route and a short Quest launcher; no competing VR preview.

## Finite next gates

**XR-14: instrument and camera precision.** Source-derived reach, world-space transform gizmo, world-anchor presets, edit undo, controller dropout recovery and scale/readability checks.

**XR-15: native medical imaging.** Load the previously supplied NLST NIfTI and selected DICOM series; verify HU, affine orientation and MPR with original source visibility. Never silently invent CT-derived airways.

**XR-16: segmentation and source-geometry QA.** Register and verify lung/airway/vessel masks and mesh topology against raw CT; clearly mark uncertain structures.

**XR-17: instrument and physics linkage.** Validate centreline navigation, lumen geometry and the separation of reduced-order modelling from fully calibrated CFD or biomechanics.

**XR-18: physical validation and freeze.** Final Quest 2 comfort/performance acceptance and scientific source audit. Do not treat code compilation or browser-emulated XR as a replacement for physical headset measurements.

Reference guidance:
- Meta's WebXR performance workflow: https://developers.meta.com/vr/documentation/web/webxr-perf-workflow/
- Meta's fixed foveation: https://developers.meta.com/vr/documentation/web/webxr-ffr/
- SlicerVirtualReality reference: https://github.com/KitwareMedical/SlicerVirtualReality
