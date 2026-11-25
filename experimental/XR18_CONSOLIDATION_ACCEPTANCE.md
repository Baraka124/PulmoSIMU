# PulmoSIMU — Consolidated Release & Independent Acceptance

**Release:** XR-18 consolidation; source branch `main`.  
**Official entry:** https://pulmosimu-xr-preview.vercel.app/  
**Single underlying procedural engine:** `index.html`.  
**Decision:** No parallel app, no speculative hardware claims, no unverified medical use.

## I. One production entry, retained technical workspaces

Vercel `vercel.json` routes the root address to the original editorial launcher in `quest-short/index.html` without duplicating its content. `/quest-short/`, `/quest.html` and `/quest4.html` redirect to `/`. The obsolete `quest.html` and `quest4.html` files have been deleted. There are three choices, all pointing to `index.html`: desktop `?entry=desktop`, VR `?entry=vr` and MR `?entry=mr`. They start the original workstation without camera permission; headset sessions require a real second click to preserve WebXR transient activation. Existing `studio.html` is retained as a functional workspace wrapper; original CT inspector, clinical validation laboratory and archived experiment source are preserved.

**Production authority:** one repository `Baraka124/PulmoSIMU`, branch `main`; one project `pulmosimu-xr-preview`. Pre-production and deployment verification must match exactly the GitHub commit hash. The obsolete `pulmovr` Vercel project and GitHub Pages publishing require dashboard-owner actions where app tools lack deletion or depublication.

## II. Four closeout tracks — evidence and limits

| Track | Software state | Device/acceptance requirement |
| --- | --- | --- |
| 1. Consolidation | Canonical root routing, historical redirects, direct desktop/VR/MR source entry, regression tests | Verify actual deployed Vercel route/status/redirect and Quest Browser entry |
| 2. Native compositor layers | Layers feature negotiation, compatibility diagnostics, original same-context endoscopy and dynamic governor remain active | **NOT IMPLEMENTED as an active compositor quad.** Three.js r128 currently renders XR through `XRWebGLLayer` and assumes its legacy `renderState.baseLayer`. WebXR compositor layers need `updateRenderState({layers:[...]})` and renderer support for XR projection layers. Replacing the legacy render state would endanger the original stereo framebuffer. Need a compatible projection-layer renderer and physical Quest optical/performance testing before activation |
| 3. Physical bronchoscope | Read-only serial input, bounded/replay-gated telemetry, three-point per-axis physical reference mapping, independently assessed measured traces, local export, explicit operator arming and stale-input stop | **No real device tested. No physical-to-virtual control enabled.** Obtain instrument, coordinate conventions, certified reference encoders, electrical safety, independent clinician/engineering review and authorized simulator-only coupling after device characterization |
| 4. Acceptance/freeze | Automated anatomy, physiology, source, XR, launcher and clinical/physical boundary regressions | **Quest run and physical measurements outstanding.** Device acceptance cannot be asserted from CI |

This represents a **software integration checkpoint**, not proof of clinical efficacy, medical device calibration, physical instrument completion or headset hardware acceptance.

## III. Native compositor engineering acceptance gate

A native quad layer would place an XR video surface into the headset compositor instead of stereoscopically resampling a flat mesh. A correct implementation must:

1. Feature-detect a **granted** `layers` feature and real `XRWebGLBinding` support on the target headset.
2. Use a renderer that correctly supplies a stereo `XRProjectionLayer` for the full scene and a transparent or opaque `XRQuadLayer` for the original endoscopic feed.
3. Present and update the quad's native layer texture from the original `camScope` source, without changing patient data or creating a second procedural camera.
4. Retain a fallback to the original WebGL render target when unavailable or if loss/recovery occurs.
5. Measure headset text contrast/readability, visual persistence, color fidelity, p50/p95 frame time, GPU cost and exits/re-entries over a continuous physical Quest test.
6. Freeze only after the actual device passes. Never mark a capability probe or successful QuadLayer allocation as active native compositor rendering.

The current `experimental/xr/display-path.js` intentionally reports `compositorQuadActive:false`. It is technically incorrect to present compositor availability as an implemented feature without a stereo renderer upgrade.

## IV. Physical bronchoscope integration acceptance gate

**Software files:** `experimental/xr/instrument-serial.js`, `instrument-bench.js`, `physical-instrument-bench.js`, `physical-calibration.js` and `experimental/validation-lab.html`.

**Measured fields:** insertion in mm, shaft roll in degrees, tip pitch and yaw in degrees. A real device must produce monotonically sequenced, individually bounded `PULMOSIMU-BENCH/1` packets. The operator supplies a `PULMOSIMU-CALIBRATION/1` dataset with **two endpoints and one independent checkpoint per axis**. Linear maps must be invertible, monotonic, in declared units and pass provisional reference residual targets. Those provisional software targets **are not clinical specifications or accredited calibration**.

**Suggested physical bench test** (protocol to be approved by equipment and clinician leads): independent reference sensor and local instrument; traceability/serial/electrical isolation recorded in laboratory logs, repeated graded insertion, clockwise and counterclockwise shaft rotation, independently controlled pitch/yaw sweeps, reversal and backlash, lost-tracking/reconnect and stale packet checks. Record device sampling rate, actual latency, residual distribution, run-to-run drift, uncertainty and operator notes.

**Live safety invariant:** read-only, no patient connection, no network upload, no automatic clinical approval, no simulated scope actuation. Configuring mappings does not arm the system. Manual arm is required and an invalid sequence, stale sample, disconnected device or out-of-range position stops the preview; operator must explicitly re-arm.

**Later coupling** would require a signed physical calibration, hardware emergency stop, dead-man hold, allowed reference-only anatomical domain, enforced units and reference transforms, feedback/latency and error limits, and independent regulatory/clinical review. That coupling is not present in the current release.

## V. Physical Quest acceptance sheet

On the latest production deployment (commit SHA and browser version recorded), use one session for each supported mode:

- **Entry:** visit the root URL, choose VR, confirm the original reference scene opens with the correct controllers, and exit/re-enter without loading a different application.
- **Anatomy:** two-handed scaling, stable pivot/rotation, controller loss, floor collision, recenter, orientation lock, 10-minute usability, movement and 6DoF visual proxy.
- **Bronchoscopy:** source endoscopic monitor readability and correct segment labels, insertion/deflection/roll, branch gating, procedural guide, precision/layout selection, video quality downgrade and recovery. Record real device frame pacing (not merely CPU submission cost).
- **Mixed reality:** real horizontal-surface hit test, explicit trigger placement, optional anchor status, reposition, unsupported hit-test fallback, tracking loss and physical drift against stationary reference. No false room occlusion or CT registration.
- **Physical-instrument laboratory:** import externally measured trace, inspect independently mapped readings, provoke stale, replayed and out-of-range packets, verify all stop conditions, and inspect clinical review evidence. Do not count synthetic CI fixtures as instrument results.

Record per test: exact URL, `main` commit, browser/headset firmware, test subject/reviewer ID, time/date, action, expected/observed result, quantitative error if relevant, screenshot/video evidence identifier and passed/blocked/failed. **Freeze** only after the corresponding external evidence is reviewed; if hardware is unavailable, mark that gate **BLOCKED—NOT TESTED**, never passed.

## VI. Finite release outcome

The present development batch closes **software routing, tested calibration preview and source-consolidation work**. Native compositor activation, calibrated real-device control and physical Quest acceptance are not technically finishable from remote software access without the renderer/device studies specified above. Keep these as exactly three bounded external acceptance gates, not an endless series of new applications or versions.
