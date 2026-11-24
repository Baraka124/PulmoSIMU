# PulmoSIMU XR-12 — Physical Quest 2 acceptance gate

**Canonical source:** `feature/mechanistic-lung-foundation`, original `index.html?xr=procedural&release=XR09-UNIFIED&interaction=XR12`.  
**Launch:** `https://pulmovr-chi.vercel.app`.  
**Status:** desktop browser/Node tests are necessary but not a replacement for physical headset verification.

## The five acceptance experiments

1. **World-anchored scene (1 minute):** sit inside a safe stationary boundary. Start immersive VR; turn and translate your head. The dark architectural chamber, floor references, organ and left-side menu must remain in world space, rather than following head motion. Verify this is a WebXR immersive session, not a 2D browser window or an OS passthrough environment.
2. **Reliable grab and release (2 minutes):** aim at original source lung. Confirm the turquoise laser ends on the first actual triangle and the hit marker is visible. Hold trigger or grip; translate and rotate the controller. Release. Lung must stay still. **A quick trigger tap must not auto-summon or move the lung.** Check both controller handedness configurations.
3. **Control and scale (2 minutes):** right stick rotates/changes depth, left stick changes organ scale. Both grips can rotate/resize together. The longest original bounding-box axis may never exceed **0.92 m** during normal comfort visualization. Its centre is kept between **0.47 m and 2.30 m** from headset. No abrupt room-filling lung or reset without explicit recenter action.
4. **Spatial controls (2 minutes):** point at ANATOMY, BRONCHOSCOPY, INSTRUMENT, RECENTER and EXIT with the laser. Aiming highlights the button; the beam stops at the panel; trigger activates only the nearest surface. Empty panel space blocks click-through. Press **left Y** to hide the menu, then press **left Y** to reopen it at the current seated viewpoint. Haptics may be available but are not required for a pass.
5. **Original bronchoscopy (3 minutes):** open BRONCHOSCOPY through the real spatial panel. Test original scope advance/retreat, right grip steering, context actions and the separate same-renderer endoscopic monitor. No automatic patient vascular risk/safety validation is implied.

## Evidence capture

Exit VR. In the same browser page choose `EXPORT XR SESSION REPORT` to download `pulmosimu-xr12-session-report.json`.

Useful fields:
- `interactionAttempts.acceptedRay`, `acceptedNear`, `movedFrames`: distinguish a hit from a successful movement
- `interactionAttempts.menuActions`, `panelToggles`: prove that menu actions were executed, not merely hovered
- `interactionAttempts.hapticRequests`: only shows that optional supported haptic requests were attempted; cannot prove physical vibrations
- `xrGeometryAtExit.modelSpanM`, `scaleClamps`, `workspaceClamps`: bounded spatial behavior before desktop model restoration
- `session.callbackIntervals`, `callbackCpu`, `stereoRenderCallCpu`: CPU/WebXR observations, **not GPU frame time or motion-to-photon latency**
- `endoscopeMonitor.status`, `framesRendered`: source-monitor health

Record the Quest 2 firmware/browser version and whether each experiment passed. Do not merge or freeze the development branch based on CI alone.

## Scientific boundaries and next milestone

XR-12 concerns headset interaction and world-anchored visualization only. The source is still the original authored procedural reference model, not a patient-specific NIfTI/CT reconstruction. Neither the R3 reference solver nor numerical clearance estimates represent calibrated tissue dynamics.

**XR-13** should concentrate on measured Quest performance, real multi-planar NIfTI/DICOM inspection and separately validated CT-derived anatomy—not additional anatomical-looking primitives or mock clinical outputs.
