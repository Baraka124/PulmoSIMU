# PulmoSIMU XR-14 — Guided bronchoscopy + Quest mixed-reality preview

**Status:** software feature tests and browser regression checks; actual Quest 2 confirmation required.
**Canonical branch:** `feature/mechanistic-lung-foundation`; **no change to `main`**.
**Launcher:** https://pulmovr-chi.vercel.app/
**Single engine:** original `index.html` source, `experimental/xr/procedural-immersive.js`.

## 1. What has actually been added

### Source-driven guidance window
- A separate source-labelled Procedure Guide beneath the original endoscope monitor; the large original clinical scene is unobstructed.
- The guide derives its bronchial segment, generation, local progress and candidate child branches from the **original** `SEG_MAP` and `scope` state.
- Actual branch buttons become active **only** when `scope.atBranch` is true and the next branch exists in `SEG_MAP`. Pressing a branch calls original `enterBranch()`, preserving its geometry/source clearance gate.
- TOOL MODE cycles the original research modes via `setClinMode()`; it does not create an alternate clinical instrument engine.
- PREVIOUS / NEXT STEP navigate a manual five-stage **presentation agenda**, not a patient intervention workflow or clinical completion score.
- The guide can be hidden from within the guide and reopened through EXPERT TOOLS → PROCEDURE GUIDE. The left Y button controls the main spatial menu separately.
- Main menu, Procedure Guide and anatomical mesh all participate in one nearest-hit pointer route. The laser cannot pass through a closer guide to grab the lung.
- Endoscope monitor is original same-renderer source. The guide is hidden from the scope offscreen capture, so it does not appear inside the endoscopic image.

### Experimental WebXR mixed reality
- Meta Quest Browser advertises `immersive-ar` support on Quest 2; **passthrough is grayscale on Quest 2**. Query `navigator.xr.isSessionSupported('immersive-ar')` before displaying the MR entry.
- The same original Three.js renderer is `alpha:true`. A separate explicit button requests an `immersive-ar` session. The active immersive VR button continues to request `immersive-vr`.
- Only in AR: remove the opaque scene background/fog, request transparent clear alpha and hide the synthetic VR room shell, so real-camera passthrough can appear behind the existing source lung and guides.
- On exit, restore the exact original `scene.background`, `scene.fog` and renderer clear alpha. The live endoscope uses a temporary solid background in its offscreen pass only.
- AR is **not a verified physical tabletop anchor**. No plane detection, room mesh, depth-based real-world occlusion, persistent anchor or marker registration is claimed. The anatomy is initially placed relative to the current tracked headset pose.
- When MR is unsupported or its session is rejected, the button remains hidden or displays a clear error. Never substitute a camera texture or fake transparent immersive VR.

## 2. Event visitor experience — 4 minutes

1. **30 seconds / Orientation.** The presenter introduces the reference lung in Studio View, then switches to Focus View; the visitor uses one trigger grip to reposition the organ.
2. **45 seconds / Source anatomy.** The visitor explores the authored bronchial structure. Explicitly label this as procedural reference, not CT patient reconstruction.
3. **90 seconds / Guided bronchoscopy.** Select BRONCHOSCOPY, observe the same-engine endoscopic monitor and guide, advance with left stick and select a branch using the interactive guide when a real junction is reached. Change tool mode and demonstrate one clearly marked simulated action.
4. **45 seconds / Mixed reality.** End VR, then begin MR in Meta Quest Browser on Quest 2. The visitor sees the grayscale physical room with the virtual anatomy/controls composited over it (if the browser supports the mode). Remain seated; do not walk toward real obstacles.
5. **30 seconds / Review.** Exit XR and export the local aggregate report. Discuss actual source status, what is validated and what is still experimental.

This is a **proposed demonstration sequence**, not an event-tested operating procedure.

## 3. Physical Quest 2 acceptance checklist

- [ ] XR-14 VR starts and shows the original scene stereoscopically with bounded anatomical scale.
- [ ] ANATOMY → BRONCHOSCOPY opens the original endoscopic monitor and a separate legible guide beneath it.
- [ ] Guide states display actual current segment name, generation, percent-of-segment and selected instrument; no fabricated patient route.
- [ ] Branch buttons are disabled away from junctions and enabled at true branch points. Pressing one calls the original navigation path and updates the displayed segment.
- [ ] The pointer stops on the nearest menu, guide or original anatomy, with no click-through.
- [ ] The guide's simulated tool mode changes the original clinical instrument state. No fake safety scores appear in the guide.
- [ ] Procedure guide can hide and reappear from EXPERT TOOLS; neither menu obstructs the source endoscopic image.
- [ ] Mixed reality entry appears only if `immersive-ar` is supported; Quest 2 real-world passthrough is visible in grayscale.
- [ ] AR does not draw the virtual chamber or pretend to occlude real objects. The same lung remains controllable without a second engine.
- [ ] Exiting AR and returning to VR or desktop restores the original opaque scene background.
- [ ] All performance, frame, controller and endoscope issues recorded in local `pulmosimu-xr14-session-report.json`.
- [ ] Headset/browser version, stereoscopic comfort, pointer accuracy and headset heat independently assessed.

Do not merge or claim physical event readiness until all device-dependent checks pass.

## 4. What is still missing for a true clinical presentation

- Patient-specific NIfTI/DICOM volume loading in the browser, spatially verified segmentation and accurate airway centreline/mesh alignment.
- Scientifically validated airway wall contact, instrument deflection, deformation and any virtual lesion/tissue interaction.
- Demonstrator case governance, a public-safe synthetic/reference dataset and explicit provenance on every screen.
- Presenter controls, spectator output/casting, a rehearsal script, timed recovery, reset-to-start and offline/network-failure contingencies.
- Reliable WebXR physical-table hit testing, anchored objects, occlusion and potentially higher-quality colour MR on newer Quest hardware.
- Hardware performance targets and physical headset acceptance. Frame callback rates are not measured display motion-to-photon timing.

**Sources**: Meta's WebXR browser mixed-reality guide https://developers.meta.com/vr/documentation/web/webxr-mixed-reality/ and the WebXR AR module https://www.w3.org/TR/webxr-ar-module-1/.
