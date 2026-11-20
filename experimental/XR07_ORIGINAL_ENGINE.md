# XR-07 — Original-Engine Immersive Procedural Laboratory

**Status:** Integrated original PNEUMO·VR source on development branch. Browser and static regression gated; physical Quest 2 tests and medical validation NOT complete.

## What changed, and what did not

Before XR-07, the short Quest URL prominently launched a **different, simplified experimental scene** comprising five schematic ellipsoids plus a reference airway tree. That was a useful interaction proof, but it did not provide the user's original procedural engine.

XR-07 is fundamentally different:

- **Entry:** `/index.html?xr=procedural&release=XR07-ORIGINAL` on the dedicated `pulmosimu-xr-preview` test environment. The short `pulmovr-chi.vercel.app` gateway prioritizes this entry. `main` remains unchanged.
- **Sole scene:** the original `scene` and `mGrp` from `index.html`, including original procedurally generated bronchial-tube meshes, five *lobed procedural meshes* and imported patient GLTF when the user actually loads one. These are not separate XR stand-ins.
- **Original disease/pathology visuals:** original pathology meshes and their source branch IDs remain in the same scene. They are not re-created inside the headset adapter.
- **Original actual procedure state:** `scope`, `SEG_MAP`, `pathCurves`, `advance/retract` via the original animation tick, `setClinMode`, `enterBranch`, `fireAction`, `updateScope`, and `camScope` are the existing application functions and objects.
- **Stereo headset rendering:** `rBro.xr.setSession`, actual `XRFrame.getViewerPose`, controller grip poses and `rBro.render(scene, vrCam)` render the original scene to Meta Quest rather than moving the desktop canvas into a screen.
- **Actual virtual endoscopic feed:** a separate world-space monitor uses the original `camScope` rendered through the existing exterior renderer into its HTML Canvas and then a `CanvasTexture` on the VR monitor. This is a **live computational rendering**, not a prerecorded video or a fabricated 2D web dashboard. Cross-WebGL-context texture performance on Quest 2 requires physical test.
- **Separate spatial regions:** the anatomical model is framed in front of the seated viewer at a comfortable display scale, with a simple work-floor horizon, a calm control panel anchored independently to the environment, and the separate endoscopic display in PROCEDURE mode.
- **Real controller interaction:** distance select brings the actual original assembly into reach; one-hand grip holds/rotates; both grips rescale; right grip during PROCEDURE maps tracked relative controller rotation to the **original scope deflection**. Left stick advances/retracts, right stick offers backup deflection, left stick X rolls, trigger enters actual branches or executes the original clinical mode's action. This is interaction mapping—not validated physical haptic mechanics.
- **Desktop regression:** original `animate()` updates the source scene as before. Only duplicate 2D renders pause during immersive XR, and scene transforms/idle state are restored after session end. No webcam or MediaPipe permission is required just to launch XR.

## Data accuracy and physical truth

**The word “original” does not mean anatomically or clinically validated.** The reference PNEUMO·VR geometry is procedural (generated meshes with noncalibrated original scene units). The shader simulates appearance and pulsation; it is not high-fidelity CFD, tissue mechanics or an experimentally verified respiratory physiology model. A GLTF imported through the existing pipeline must have its original coordinate frame, scale, provenance, CT-derived registration and scientific validation independently verified before any clinical measurements or patient-specific claims.

XR-07 is a **real WebXR, real tracked input and real original procedural engine integration**. The next milestone for biological fidelity is calibrated, CT-registered anatomy and validated physical constraints, not prettier schematic lobes.

## Controller controls

| Mode | Input | Effect |
| --- | --- | --- |
| ANATOMY | Controller index trigger pointing at source mesh | Select; smoothly summon the actual original assembly to near-field distance |
| ANATOMY | Side grip while near object | One-hand model handling with preserved grip offset |
| ANATOMY | Both side grips | Two-handed rotate and resize the original anatomy |
| PROCEDURE | Left thumbstick Y | Real original scope advance/retract via existing velocity/physics loop |
| PROCEDURE | Left thumbstick X | Scope roll |
| PROCEDURE | Right thumbstick | Scope horizontal and vertical deflection when not physically steering |
| PROCEDURE | Hold right side grip and rotate controller | Steer original scope deflection using tracked controller orientation |
| PROCEDURE | Right index trigger | Follow available branch / execute current original clinical action |
| PROCEDURE | Left index trigger | Select first available branch when offered |
| ALL | Right A | Toggle anatomy/procedure |
| ALL | Left X | Recenter original anatomy |
| ALL | Right B | Cycle original navigation/biopsy/measure/annotate mode |
| ALL | World menu action | Mode selection, tool, recenter or exit |

Menu actions have priority over anatomy and instrument actions. Do not confuse VR controller vibration with real bronchoscope resistance.

## Verification and remaining acceptance

Automated:
1. `experimental/tests/procedural-xr-adapter.test.mjs` checks original-source reuse and removal of the old conflicting wrist-menu loop.
2. `experimental/tests/browser-smoke.mjs` opens the original `/index.html` and checks the actual source model, instrument adapter and pathologies initialize without browser exceptions.
3. The original `pipeline.py`, `server.py`, `synthetic.py` remain unchanged from `main`. `index.html` changes are confined to introducing the immersive adapter, renderer capture and XR-safe desktop render arbitration.
4. Existing research and experimental workbench tests must remain green.

Physical Quest validation **still required**:
- Open direct original-engine XR-07 (not experimental `vr.html`) and confirm original lobed geometry, bronchial tree, pathology.
- Confirm initial distance/size on a *seated* Quest 2.
- Confirm controller handedness even when input connection order changes.
- Confirm near-grab, two-grip manipulation, recenter and no headset locomotion.
- Switch to procedure, verify **live** video monitor (not black/frozen), real branch stepping and right-grip physical steering.
- Verify scope action controls, appropriate clinical-mode labels, XR session exit and desktop fallback.
- Record frame timing and 10-minute comfort test. Fail gracefully if input source or live monitor capture is unsupported.
- Freeze only when these pass. Do not claim FDA/MDR medical-device validation or clinically measured physical forces.

## Finite next milestone

**XR-08 · Medical imaging + instrument constraints:** add a verified registered CT/GLTF geometry source adapter with validated units, airway-to-vascular topology and mesh-based collisions; upgrade device contact modeling and dynamic physics **only from appropriate reference measurements**. Keep original procedural engine and scene identity as canonical; replace synthetic reference geometry only when provenance is verified. Do not add isolated prototypes.
