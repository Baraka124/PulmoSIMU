# PulmoSIMU XR-16 Final — Clinical and Physical Instrument Verification Package

**Status:** software/bench instrumentation can be built and automatically tested. **Physical instrument acceptance, clinician evaluation, anatomical source validity and clinical use remain blocked without evidence.**

**Primary software**: original reference procedure `index.html`; XR adapter `experimental/xr/procedural-immersive.js`; deterministic station `experimental/xr/bronchoscopy-station.js`.  
**Physical bench**: `experimental/instrument-bench.html` and `experimental/xr/instrument-bench.js`.  
**Evidence format**: downloadable, local-only JSON. No remotely uploaded clinical records.

## XR-16 acceptance matrix — actual status

| Domain | Software implemented / evidence | Required external study |
| --- | --- | --- |
| Source airway & branch truth | Original `SEGS`, `SEG_MAP`, scope and branch constraints; browser source tests | Clinician review against annotated reference bronchoscopy and relevant source CT |
| Navigation | Same original reference `advance/retract/enterBranch`; Quest thumbstick input | Trace predefined reference paths and log branch accuracy, unintended transitions, overshoot and operator workload |
| Visual endoscopy | Same WebGL context/camera; 800×450 default, selectable detail/economy, field-of-view selection, automatic CPU-cost fallback | Human image-quality review, lumen plausibility, visibility under motion, display latency and actual headset GPU/frame pacing |
| Scope tip | Persistent, bounded incremental controller deflection | Physical controller distal angle mapping vs calibrated angle reference; hysteresis, drift and repeatability |
| Shaft roll | Independent bounded incremental rotation | Compare commanded to actual measured shaft rotation; measurement tolerance agreed before collection |
| Insertion/retraction | Command stops when stick released, reference estimated clearance gate | Compare physical insertion encoder/optical ruler vs UI command; no unauthorized clinical use |
| Collision | Reference-radius clearance estimate; CT-specific navigation blocked | Calibrated true segment/triangle-wall contact, shaft bending and friction; not modeled yet |
| Guide | Named source junctions; buttons enabled only at actual branch | Review route semantics, instructional safety and misleading language |
| Physical input | Local JSON protocol, sequence gating, strict bounds, freshness check, zero-reference offset and export | Acquire lawful device, electrical safety, sensor characterization, coordinate transform calibration; no physical device attached |
| MR/VR recovery | Session reset, mode separation, source/provenance checks | Real Quest 2 10-min and longer testing, loss of tracked controllers, exit/re-entry, nausea/comfort checks |
| Data governance | No network access from physical bench, no patient input required | Ethics, information security and intended-use classification with hospital approval |

## Gate C1 — Reference anatomical and clinical task review

**Lead:** respiratory clinician / bronchoscopy expert; **engineer:** biomedical engineering. The clinical reviewer supplies a **de-identified** reference task/route (or a synthetic research fixture explicitly described as such) and signs an assessment independently from the implementer.

For each route, collect: task identifier, reference model source/version/hash, named target segment and actual segment ID, traversal sequence, navigation errors, anatomy visibility, guide correctness, unplanned branch selections, time to complete, observed issues and reviewer notes. Collect multiple independent routes and operators appropriate to the intended research purpose; sample size must be justified in a protocol, not invented by software.

A clinician's favorable opinion on a reference scene does not prove patient-specific segmentation accuracy. Do not label clinical suitability based on an unblinded demonstration.

## Gate C2 — Clinical UI and procedural instrument review

Use the same procedures on desktop and XR. Test the scope camera, depth cue quality, branch discrimination, guide semantics, tool-mode selection, emergency stop and state persistence. Record both expected results and observations, adverse events/near misses and discrepancies. The labels **'biopsy'**, **'measurement'** and **'annotation'** mean reference-simulator actions only; no tissue sampling or patient measurement is implied.

Review patient CT source separately: DICOM orientation and spacing, RAS/LPS chain, source hash, mesh derivation, centerline registration, branch/segment labels and blinded image-to-model agreement. Patient-specific bronchoscopy remains blocked until a reviewed, registered airway and procedure safety study exist.

## Gate P1 — Physical instrument choice and interfaces

**Candidate hardware**: a non-clinical training bronchoscope or purpose-built handle and shaft, not a patient-use instrument. The benchtop may include:
- insertion sensor (linear optical encoder or displacement sensing),
- shaft-roll encoder (absolute/incremental angular sensor),
- two-axis distal tip articulation feedback (sensor or mechanical reference),
- optional test fixture to independently measure target tip position / contact.

Capture device manufacturer, model and serial in **local laboratory records**, electrical isolation/voltage, measurement range, quantization, mounting tolerances, coordinate conventions, sampling rate, timestamp basis, firmware, calibration version, lab operator and traceable reference devices. No firmware/hardware or connected physical controller is supplied by this repo. Do not claim actual force feedback or compliant tissue mechanics.

### Read-only bench packet contract

A single UTF-8 JSON line/record, up to 512 characters, using `protocol:"PULMOSIMU-BENCH/1"`, monotonically increasing integer `seq`, and four finite numeric fields:

```json
{"protocol":"PULMOSIMU-BENCH/1","seq":1,"insertionMm":400,"shaftRollDeg":15,"tipPitchDeg":20,"tipYawDeg":-5}
```

Range gates: insertion [0,2000] mm; signed shaft roll, pitch and yaw each [-180,180]°. These are generic software **guardrails** for research packet validation, not evidence of instrument mechanical travel limits or approved clinical thresholds. Assign device-specific calibration limits after measurement. Ingesting a frame does **not** move the original scope; this prevents accidental actuation or false device-to-anatomy coupling.

The local bench rejects malformed frames, wrong protocol, nonfinite and out-of-range values, non-incrementing sequence numbers and readings older than 250 ms for calibration/readout. An operator-entered zero offset does **not** constitute traceable metrological calibration.

## Gate P2 — Calibrate, verify and establish uncertainty

Design before-data-collection acceptance limits for: encoder resolution, absolute error, repeatability, hysteresis/backlash, return-to-zero drift, timing delay/jitter, loss/recovery after unplugging, and any temperature or mechanical load dependence. Record raw references (do not rely solely on screenshots), verified sensor data, residual error and measurement uncertainty, uncertainty budget, protocol/software/firmware versions, independent sign-off.

Mandatory five-task fixture minimum for method development:
1. zero reference + return to zero,
2. 10 or more graded insertion checkpoints through permissible range,
3. clockwise/counterclockwise roll sweeps,
4. separate pitch/yaw deflection sweeps and repeated reversals,
5. deliberate input disconnect / out-of-order samples / lost tracking / reattachment.

**Measured tolerances must be agreed with clinician and instrumentation laboratory before test runs.** No arbitrary pass threshold in this code can substitute for physical verification.

## Gate P3 — Separate instrument-to-virtual-scope coupling

After P1/P2 acceptance, design an explicitly opt-in device adapter with hard stop, unit and coordinate mapping, bounded velocity, signal-loss zero velocity, replay rejection and per-device certified calibration IDs. Maintain a separate physical pose estimate, desired command and resulting virtual pose. Display differences; do not silently match them.

Do NOT activate physical-to-original-scope control while the research bench is marked `deviceToScopeCoupled:false`. The current build does not include the coupling adapter, real instrument pairing, haptic force output or hardware acceptance.

## Gate XR — Quest 2 / clinical workflow acceptance

Log device/browser versions, frame timing percentiles, camera rendering cost, 10-minute continuous use, controller handedness/recovery, two-hand manipulation, procedure monitor visibility and text legibility, complete branch task, failure/exit/reentry, MR capability detection and unsupported-feature behavior. Mark pass only after a **physical test**. CPU submission time is not GPU time.

## Evidence sign-off and release boundary

The local bench has 12 named evidence gates: `source_graph_review`, `scope_mechanical_dimensions`, `centerline_navigation_accuracy`, `distal_tip_deflection_accuracy`, `shaft_rotation_accuracy`, `insertion_depth_accuracy`, `wall_contact_and_collision`, `task_completion_reproducibility`, `endoscopic_scene_fidelity`, `clinician_review`, `hardware_tracking_recovery`, and `privacy_and_data_governance`. The software requires an explicit `pass` and a nonblank evidence ID for every gate. Those are **bookkeeping checks, not cryptographic or third-party verification of truth**; the reviewer must inspect and sign the underlying evidence.

**Release status when software CI is green:** XR-16 reference research software ready for device-specific test; physical instrument integration and clinically validated simulation **not approved**. The next milestone XR-17 is separate from independent clinical and benchtop source/device assessment.
