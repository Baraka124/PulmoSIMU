# PulmoSIMU · XR-09 Quest 2 session-health and acceptance protocol

**Branch:** `feature/mechanistic-lung-foundation`  
**Status:** XR-10 controller fixes and automated checks; physical Quest 2 verification pending  
**Canonical viewer:** original `index.html?xr=procedural&release=XR09-UNIFIED`  
**No competing viewer and no change to patient anatomy or the R3 numerical solver.**

## What is measured

The new `experimental/xr/session-health.js` records **local bounded aggregates** during the original-engine XR session:

- WebXR animation callback inter-arrival time (P50 / P95 / maximum).
- JavaScript callback CPU duration; stereo `renderer.render` call CPU duration.
- Optional endoscopic offscreen-render CPU call duration.
- Total frames, last 600 frame observations (bounded memory), and missing viewer-pose frames.
- Observed controller connect/disconnect, tracked-hand loss, viewer-pose loss and monitor-unavailable events.
- The runtime-reported headset refresh rate **if exposed**. A missed-budget count is meaningful only when the runtime provides this value.

These are **not GPU timings, compositor timing, refresh-rate certifications, reprojection measurements, haptic forces, clinical outcome evidence or patient physiology**. The observed callback rate is an estimate from callback intervals, not measured display FPS.

No raw 6DoF poses, 3D geometry, images, DICOM metadata, patient identifiers, IP addresses or recorded endoscopy video are stored. The browser does not upload a report.

## Running the headset acceptance

1. Open `https://pulmosimu-xr-preview.vercel.app/quest-short/` in **Meta Quest Browser**, on a charged Quest 2, with seated boundary and controllers paired.
2. Launch **XR-09 · Immersive procedural laboratory**. Use the real **ENTER PROCEDURAL XR** control.
3. Record whether the *original* lobar and branching airway surfaces appear stereoscopically at comfortable distance. Do not assume these procedural shapes are CT-calibrated.
4. In ANATOMY, confirm the lung is placed approximately 1.05 m from the viewer. Aim the controller laser at the real lung surface; the target marker should appear when the ray hits. Hold **either trigger or grip** while moving/turning the controller. Confirm the lung actually translates and rotates, then stays put on release. A brief unmoved trigger tap summons the lung nearer. Also test a direct near-surface grip and simultaneous two-controller resizing. Empty-space rays must never grab the lung.
5. If hand tracking is available, verify pinch, release, controller handoff, and recovery after briefly occluding a hand. The system should release stale hand grips rather than keep moving the organ.
6. In BRONCHOSCOPY, advance/retract, steer with right grip, and verify the same original scope/branch/pathology scene. Verify the live endoscopic feed; a frozen/blank/unavailable monitor fails acceptance.
7. Deliberately let a controller sleep or temporarily lose tracking in a safe seated situation. Confirm no uncontrolled scope motion or lasting object grab. Do not physically walk while testing.
8. Complete a **10-minute** seated interaction including several mode switches. Visually assess stereo comfort, world stability, labels, scope feed, latency and heat.
9. Exit XR. In the **same browser page**, choose **EXPORT XR SESSION REPORT**. Share the generated `pulmosimu-xr09-session-report.json` if available; the download is local and user initiated. The page also exposes `pulmoXRSessionReport()` for developer inspection.
10. Inspect `interactionAttempts.acceptedRay`, `acceptedNear`, `rejected`, `trigger`, `grip`, and `movedFrames` in the exported JSON. **An accepted grab without moving frames is a failed interaction test.** Attach a short human acceptance note: headset/browser version, input type, observed issues and whether each step passed. There is no automated substitute for this physical review.

## Interpreting the JSON

- `reportedHeadsetFrameRateHz`: value returned by WebXR (or `null`); not fabricated.
- `frameBudgetMs`: derived from the WebXR-reported rate when available.
- `callbackIntervals.p95Ms`: timing between observed WebXR callbacks; it can include tracking/system scheduling.
- `callbackCpu.p95Ms`: time spent in app callback on the CPU, **not** an end-to-end GPU frame cost.
- `stereoRenderCallCpu.p95Ms`: time spent calling the stereo renderer, not GPU execution.
- `endoscopeCaptureCpu.p95Ms`: optional offscreen feed call duration; includes callback frames where no capture was due.
- `intervalsOver1Point5Budgets`: diagnostic count using WebXR-reported refresh rate (null if unavailable).
- `viewerPoseMissingFrames` / `trackingAndMonitorEvents`: incident counts, not proof of recovery or completed haptic validation.

Interpret variability using the *entire* headset run, scene complexity and device firmware. CPU timings and interval statistics alone cannot prove stable 72 Hz / 90 Hz, GPU headroom or acceptable motion-to-photon latency.

## Non-negotiable scientific boundaries

- The XR anatomy is the **original procedural source**; an imported CT-derived GLB is separately labeled and requires independent geometric registration.
- The shared R3 RC solver is uncalibrated, reference-only, not coupled to deformation or CT.
- Clearance is a branch-radius tool fit check, **not wall contact pressure or haptics**.
- Hardware acceptance is a separate gate from browser/Node tests.
- Do not merge/freeze on automated CI success alone. Proceed to registered-source and measured-instrument validation only with independently reviewed geometry and physical data.
