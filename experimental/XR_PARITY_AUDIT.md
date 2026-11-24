# PulmoSIMU XR parity audit — November 20, 2026

## Critical correction

**The XR viewer is a simplified experimental visualizer, not an immersive port of the original PNEUMO·VR application.** Previous XR-02–XR-04 work improved spatial controls, viewer positioning and reference-lobe navigation, but **did not bring over the full original procedural environment**. Treating the XR pilot as a completed advanced system would be misleading.

## Verified repository reality

- `main/index.html` (~233 kB, ~4,343 lines): original PNEUMO·VR surgical planning system using global Three.js 0.128, with two canvases, bronchoscopic navigation, procedural controls, pathology and biopsy logic, clinician UI, MediaPipe gestures, patient-model GLTF/JSON loading and DICOM-related metadata. Its own WebXR path has controller-based procedural navigation but is a separate legacy implementation.
- `feature/mechanistic-lung-foundation/index.html`: preserved original application, **not replaced** by XR.
- `experimental/vr.html`: Three.js 0.160 experimental XR viewer with **five schematic ellipsoidal lobe envelopes**, original-derived 18-node airway reference, controller grabbing, a spatial panel, a reference route marker and selection handoff to Studio. No biopsy or bronchoscopic lumen engine, full pathology tooling, vascular geometry, patient CT scene or device interaction integrated in immersive mode.
- `studio.html`: switcher between preserved full procedure, mechanistic experiment and XR. This is **UI and reference-focus interoperability only**, not a common 3D scene.

## Acceptance parity checklist

| Subsystem | Original/main | Experimental XR | Required for integrated advanced XR |
| --- | --- | --- | --- |
| Existing procedural anatomical visualization | Present | Simplified spheres / tubes | Reuse the original procedural scene, geometry and labels via a versioned renderer adapter; eliminate duplicate hand-built lobes as the primary anatomy |
| Bronchoscope motion, camera and luminal rendering | Present | Reference path marker only | Preserve actual bronchoscope operation and render it as an intentional VR procedure mode |
| Instrument modes, biopsy, bleeding, airway contact | Present in original | Absent | Instrument tool state, visuals and interactions with QA parity, not just button labels |
| Pathology / clinical scenarios | Present in original | Absent | Use same scenario state and pathology objects, with explicit source boundaries |
| Patient-derived anatomy | Existing GLTF/JSON and DICOM-related pathways | Absent | Versioned CT/mesh loader, measured units, reference-frame registration and explicit provenance before VR display |
| Vessel/micro anatomy | No validated general-purpose source proven by current repository audit | Absent | New verified model pipeline only; never claim vascular or microscopic geometry from synthetic ellipsoids |
| Quest interaction controls | Original scope navigation | Grab/scale/inspector | One mode system: anatomy handling vs procedure vs imaging layers, no conflicting controller mappings |
| Research mechanics | Experimental RC and therapy engine | Not coupled to rendered anatomy | Explicit unit-tested bridge, without claiming physics or clinical validation |
| Comfortable headset origin | Legacy VR has reported placement issues | XR-04 adapted headset-relative origin | Quest 2 physical pass; don't equate automated browser tests with headset validation |

## Engineering decision — source-of-truth first

**STOP** extending five ellipsoids as the primary XR system. They can remain a labeled fallback for low-resource testing only. Future advanced XR should adapt the **existing procedural renderer** and instrument state, not recreate their behavior from scratch.

Three **finite** stages:

1. **Parity bridge:** extract a stable model/scene/instrument contract from original `index.html`, with documented coordinate origin, units, material/mesh identifiers, live tool state and safe ownership. Ship a minimal import/export adapter with tests. Ensure full original desktop application still works.
2. **Unified high-fidelity XR:** mount an upgraded presentation of the same original anatomy and original procedural instrument engine under one WebXR lifecycle. Use corrected XR-04 headset-relative comfort framing; organize modes and spatial inspector for the original system. Reuse shared original source IDs instead of creating a competing anatomy.
3. **Imaging + device integration gate:** connect registered CT/vascular surfaces, procedural instruments, numerical model and XR only after unit/frame/source verification. Validate on real Quest 2 and preserve a desktop fallback. Freeze XR baseline after physical acceptance.

**Release gate:** No further “open Quest and enjoy the advanced system” claim until original procedural features and XR scene are demonstrably integrated, regression-tested and physically verified.

## Launch policy

The short Quest URL must expose a clear choice between **Full Procedural PulmoSIMU (original desktop-oriented experience)** and **XR interaction pilot (simplified schematic anatomy)**. This avoids confusing feature completeness with successful WebXR entry. The full procedural application can have its own original VR mode, but that mode is not yet the repaired/integrated headset experience.
