# PulmoSIMU XR-16F — Clinical and Physical Instrument Validation Protocol

**Owner:** PulmoSIMU / Respiratory Engineering | **Version:** XR-16F | **Date:** 2026-11-24

**Purpose:** close software engineering and create traceable, repeatable **external** clinical and device verification gates. This document does **not** declare clinical or physical validation complete. Approval must be based on real clinical review and measurements of an identified physical instrument.

## A. Release boundaries

- One release authority: GitHub `main` → Vercel `pulmosimu-xr-preview`.
- One procedural source engine: `index.html` with `SEGS`, `SEG_MAP`, `scope`, `advance()`, `retract()`, `enterBranch()`.
- XR view, input, endoscopic monitor and guide are **presentation/controls over the original procedure**. Imported lung-shell CT meshes are not automatically bronchial trees.
- `PulmoXR08.assessLumenClearance` compares a reference branch radius and scope diameter. It is **not** contact mechanics, force measurement, validated airway diameter or patient-safety clearance.
- A simulated training instrument is not a physical medical device; VR haptics are not tissue force feedback.
- No patient identifiable data or private imaging leaves the browser. Keep any signed clinical review in an approved institutional document system rather than the public repository.

## B. Clinical reference-review gate (C1–C6)

Reviewer: qualified pulmonologist/bronchoscopist, with date and source evidence.

1. **C1 — Airway topology:** compare every authored reference `SEGS` parent/child to a documented bronchial atlas and verify side, generation and named junctions. Document exceptions; this is **reference-model review**, not patient registration.
2. **C2 — Landmarks:** check trachea, carina, main bronchi, lobar origins, branch naming and left/right orientation under realistic point-of-view change; capture screenshots of evidence and annotate deviations.
3. **C3 — Visual fidelity:** compare endoscopic morphology, lighting, apparent lumen and occlusion with non-identifying teaching images where available; score each factor independently. Record any artifact caused by rendering rather than anatomy.
4. **C4 — Procedure logic:** navigate from proximal to distal and back, select valid branches, test impossible transitions, document mechanism of distal tip steer, rollback and selected instrument. Separate input-control response from tissue biomechanics.
5. **C5 — Source/provenance:** verify each label and derived readout discloses original-reference versus source-verified CT; ensure CT-specific bronchoscopy remains blocked when no independently reviewed CT airway segmentation and transform exist.
6. **C6 — Physical fixture:** review an independently measured physical bench report (section C). A unit/browser test or Quest controller pose does not count as measurement of an external bronchoscope.

**Required outcome:** named clinical reviewer, role, review date, six evidence statements, limitations acknowledged, an external bench report and explicit unresolved findings. Passing the form means `READY_FOR_INDEPENDENT_CLINICAL_REVIEW`, **never** `clinically validated`.

**Independent validation still required:** clinical ground-truth data, independent reviewers, traceable source acquisition, relevant clinical performance targets, documented applicability and regulatory review before claiming clinical decision support.

## C. Physical instrument bench gate (P1–P8)

The separate laboratory entry is `/experimental/validation-lab.html`; it runs entirely in-browser.

**P1. Identify hardware.** Record instrument model, instrument/fixture ID, calibration state, independent reference sensor (encoder, ruler/tracker, optical measuring apparatus etc.), operator and collection time. A Quest controller by itself does not establish true bronchoscope tip motion.

**P2. Synchronize clocks.** Timestamp sample records in **milliseconds**; measure reference hardware and commands in a common monotonic clock where possible. Confirm timebase synchronization before claiming latency. The current validator checks timestamps and largest gaps; it does not infer physical end-to-end latency.

**P3. Insertion / retraction.** Measure commanded versus externally measured axial travel in **millimetres**, in both directions, over an instrument-appropriate travel range. Test neutral input and lost-tracking cases, not only forward motion.

**P4. Distal articulation.** Measure horizontal and vertical deflection in **degrees**, including return-to-neutral, saturation, reversed direction and repeatability. The source-reference software uses `CFG.deflMax = 32°` as a simulated display limit, **not** a clinical instrument specification.

**P5. Shaft rotation.** Measure angular rotation with an independent angular reference and handle wraparound at ±180° when computing residual errors.

**P6. Capture a trace.** JSON format `pulmosimu-instrument-trace/1`, with:
- `provenance.deviceModel`, `instrumentId`, `referenceSensor`, `operator`, `measuredAt`, `measurementSource='external-bench'`
- `units: {time:'ms', insertion:'mm', angles:'deg'}`
- `samples[]`: `tMs`, commanded and measured insertion/tipH/tipV/roll, and Boolean `tracking`.
- No patient name, patient study IDs or personal health data.

**P7. Initial engineering targets (NOT clinical tolerances):** at least 50 samples spanning at least 5 seconds, no sample gap over 250 ms, no unreported tracking loss, 95th percentile absolute insertion error ≤ 5 mm, tip H/V error ≤ 8°, and angular roll error ≤ 12°. **Targets must be reviewed and replaced** with measured device capabilities and the intended clinical training task before a physical validation claim.

**P8. Reproducibility and failure injection.** Perform independent runs for several motion amplitudes and both rotation directions. Re-test after unplugging/reconnecting instrument interface, headset/controller loss, page reload, and VR exit; confirm that loss of input stops insertion and never causes uncontrolled movement.

**Outcome language:**
- `REJECTED`: provenance or trace schema invalid.
- `BENCH_TARGET_NOT_MET`: structurally valid physical trace but preliminary engineering target missed.
- `BENCH_ENGINEERING_TARGET_MET`: submitted measured trace met preliminary checks. **No proof of authenticity, mechanical equivalence, clinical performance or device certification.**

### P9. Live packet input inspection (isolated acquisition gate)

The same `experimental/validation-lab.html` now exposes the optional read-only live-packet diagnostic. This is deliberately **separate** from the independently measured trace validator above. `experimental/xr/instrument-bench.js` exports `PulmoXRInstrumentTelemetry`, and never calls the original scope's motion/biopsy commands.

Protocol: `PULMOSIMU-BENCH/1` with a monotonically increasing non-negative integer `seq` and finite `insertionMm`, `shaftRollDeg`, `tipPitchDeg`, `tipYawDeg` fields. The generic parser rejects frames over 512 characters, missing fields, replayed sequence numbers, stale samples and out-of-range values. A sample must remain under 250 ms old to be considered fresh. Read-only zero-reference offsets are operator-entered and **not** certified sensor calibration. The software records `clinicalUse:false` and `deviceToScopeCoupled:false`; it does not request Bluetooth, serial or any other physical device pairing.

An optional read-only Web Serial intake is now available behind a user-clicked **Connect serial bench (read-only)** button. It is implemented by `experimental/xr/instrument-serial.js`, accepts newline-delimited packets at 115200 baud, and never writes to the instrument. The browser must support Web Serial and request the user's device permission. Disconnect terminates the reading loop and closes the port; a headset or device without Web Serial remains eligible for manual packet inspection. No physical device, encoder, mechanical fixture or reference measurement has yet been tested by the software developer.

This packet verifier is intended for bench protocol development and measurement-quality debugging. **It does not replace** the external-trace comparison, clinician review or headset tests.

## D. Quest XR-16 final physical acceptance matrix

| Scenario | Capture and failure criteria |
| --- | --- |
| Desktop launch without headset/camera | Opens original 2D procedural workstation; no permission request |
| XR entry and exit | Enter original scene, exit and re-enter without stale pose or missing visuals |
| Two-hand manipulation | No jumps, unexpected teleports or loss of reference scale |
| Controller loss / missing gamepad axes | No unbounded scope insertion, tip values remain finite, no runtime crash |
| Endoscopic monitor | Source pixels remain live and visible, retry after rendering fault, status matches actual frame output |
| Quality selection | Detail/Balanced/Economy actual source resolution, record CPU-cost downgrade without fabricated GPU measurements |
| Guide and source gate | Branch labels from `SEG_MAP`; no CT-specific procedure on unsupported patient shell |
| Session duration | 10 minutes continuous use; observer logs nausea/comfort, readability, recovery and scope operation |
| Performance | Record device/browser, actual headset frame interval, 95th percentile frame times, monitor CPU submission cost, thermal/guardian state; do not infer GPU timing from JS CPU call duration |

The last column of the physical acceptance record must list the exact operator, date, Quest/browser version, software commit SHA and pass/fail evidence. If a scenario is not exercised, label **NOT TESTED**, not PASS.

## E. Responsibility and stage gates

| Gate | Evidence | Status as of 2026-11-24 |
| --- | --- | --- |
| Software invariants | GitHub numerical, source integrity and Playwright browser checks | Automated; inspect latest green CI SHA |
| Clinical source inspection | Independent professional review packet + evidence | **PENDING EXTERNAL CLINICAL REVIEW** |
| Physical instrument bench | Actual measured trace and fixture metadata | **PENDING HARDWARE AND INDEPENDENT SENSOR** |
| Physical Quest XR-16 acceptance | Real headset run and local diagnostic report | **PENDING XR-16F RETEST** |
| Patient CT procedure validity | Reviewed registered CT airway graph + independent performance data | **BLOCKED** |
| Medical device / training efficacy | Device-level clinical, usability, safety and regulatory evidence | **NOT CLAIMED** |

### Closing criterion

XR-16 software can be frozen independently once its unit/browser suite passes, but **scientific, physical instrument and clinical acceptance remain explicit external gates**. This avoids indefinite new feature phases without inventing certificates from simulation-only evidence.

The independent evidence schema and local UI are source-controlled under:
- `experimental/xr/instrument-bench.js` (read-only individual telemetry packet checks)
- `experimental/xr/instrument-serial.js` (explicitly approved read-only serial acquisition; no device actuation)
- `experimental/xr/physical-instrument-bench.js`
- `experimental/xr/clinical-review-gate.js`
- `experimental/validation-lab.html`
- `experimental/tests/xr16f-physical-clinical-evidence.test.mjs`
