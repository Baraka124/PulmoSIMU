# PulmoSIMU — Unified Anatomical, Computational and Immersive System

**Decision record:** 2026-11-23; updated 2026-11-24  
**Canonical release:** `main` (desktop / XR-16 bronchoscopy workstation), with XR-17 modality architecture specified but **not implemented**  
**Canonical procedural source:** original `index.html`, not a copied XR anatomy scene  
**Status:** implemented integration foundation; user-observed headset entry, six-DoF manipulation and smooth two-handed scale. **Formal Quest 2 performance/device acceptance, clinical validation and true coupled physics remain pending.**

## 1. One system, multiple research capabilities

PulmoSIMU is a spatial respiratory engineering environment. It is **not** a series of independent 3D demos. The scientific scene, clinical procedure, numerical models and tracked interface have clearly defined relationships:

| Component | Canonical implementation | Relationship to shared system | Current evidence |
| --- | --- | --- | --- |
| Procedural anatomical scene | `index.html`: `mGrp`, `SEGS`, `SEG_MAP`, `pathCurves`, source GLB | One source for desktop and original-engine WebXR | Browser and source-regression tests |
| Bronchoscopy and instrument controls | `index.html`: `scope`, `advance`, `retract`, `updateScope`, `fireAction` | Exactly the same procedure state in desktop and XR, with exclusive XR-frame stepping during VR | Automated source/browser checks; physical Quest2 pending |
| Immersive scene and tracked interaction | `experimental/xr/procedural-immersive.js` | Owns XR sessions, hands/controllers, scene framing and in-world inspection, **not anatomy or organ physics** | Automated checks; physical headset pending |
| Geometry-backed grasp | `experimental/xr/surface-contact.js` | 26-direction ray queries confirm source mesh triangles within proximity; no empty AABB grabbing | Chromium sphere/corner/far-contact tests; hand-tracking behavior pending |
| Source-coframe and instrumentation | `experimental/xr/registered-geometry.js` | Source-derived CT-to-display scale, graph identity and estimated bronchoscope clearance | Validation tests; neither clinical registration nor tissue collision |
| Numerically conserved reduced-order breathing | `experimental/core/breathing-network.js` | **Single existing R3 solver** loaded via `experimental/xr/reference-physics.js` in original VR, only for original procedural reference | Unit/SI/conservation tests; uncalibrated assumptions |
| Mechanistic disease/therapy experiments | `experimental/core/mechanistic-lung.js`, `disease-engine.js`, `therapeutics-engine.js` | Research workspace retained; **NOT** bound to patient-specific or bronchoscopy state | Mechanism invariants; no patient response validation |
| Source CT loader | original `index.html` and `pipeline.py` | Co-frame check before mutating `SEGS`; GLB-authored materials preserved | Browser tests; DICOM LPS/RAS registration not yet verified |
| Studio | `studio.html` | Procedural, immersive procedural XR and numerical lab are navigational facets of one product | Browser route and focus tests |
| Unified launch | `quest-short/index.html` | Clear desktop first entry, plus original-engine VR and capability-gated MR links | Browser route and no-camera desktop checks |

### Explicit status distinctions

`SOURCE PRESENT` ≠ `COFRAMED` ≠ `MEASURED` ≠ `INDEPENDENTLY VALIDATED`.

An imported GLB and centreline points being close in the same numeric frame is necessary, not proof of image registration or anatomic accuracy. The r128 `WebGLRenderer` renders the same original scene stereoscopically, but display scale does not modify patient dimensions or validate tissue properties.

## 2. Components retired from active navigation

The older `experimental/vr.html` (XR-04–XR-06, schematic ellipsoids and duplicated reference airway geometry) is retained **only as an explicitly historical, browser-testable research fixture**. Direct visits now redirect to the canonical original-engine XR; deliberate archive inspection requires `?archive=1` and is not offered in product navigation. The files are not deleted because they preserve earlier experiments and regression coverage.

These surfaces no longer direct normal usage to that viewer:
- `studio.html`: immersive mode now uses original `index.html?xr=procedural`.
- `quest-short/index.html`: removed promoted XR-06 choice; XR-09 original engine is the principal experience.
- `quest.html`, `quest4.html`: redirect to XR-09 rather than bringing back obsolete schematic prototypes.
- `experimental/shared/workspace-route.js`: canonical XR route points to original engine.

Avoid reviving a competing app during future development. Historical route tests are maintained only as archive integrity tests, not as future product gates.

### Mechanistic laboratory source-integrity correction (23 November 2026)

The active `experimental/lab.html` numerical laboratory no longer creates five stretched sphere-based lobes, reference airway surface tubes or disease-coloured lung illustrations. Its default stage is a region-indexed quantitative observatory driven by the original R3–R5 mathematical engines. It does **not** pretend that five numerical compartments form a full anatomical mesh. Three-dimensional source inspection is available only after locally loading a co-frame-checked CT/GLB pair or manifest-backed external surfaces. Source-authored GLB materials are preserved; source centreline overlays are lines, not invented bronchial walls. Studio routes to this corrected laboratory.

The original `index.html` procedural scene remains the authoritative procedural and original-engine XR anatomy. It is an authored reference model, not a claim of patient-specific or measured anatomical fidelity. This correction does not validate the numerical physics or imported image segmentation.

## 3. Scientific fidelity and visual truth

### Geometry

- **Procedural reference:** original authored bronchial tree and lobar shapes, **not calibrated or patient specific**.
- **Synthetic GLB:** generated input for simulation/software verification, not anatomical evidence.
- **CT-derived GLB:** only source-authored mesh surfaces and materials, with a valid associated centreline graph and transform. We have explicitly **removed synthetic cartilage/vascular shader painting** from imported CT meshes because visually invented vessels were not scan-extracted structures.
- **Missing vascular, pleural or microscopic structures:** do not fabricate an apparently CT-derived layer. Add these only from explicitly sourced and validated geometry.

For genuine DICOM registration, the next source contract must capture voxel dimensions, image orientation and position, DICOM LPS frame and any RAS/display conversions as explicit 4×4 transformations with inverse roundtrips. The existing `centre_mm` plus `scale` transform alone cannot prove the complete image-to-world orientation. See 3D Slicer reference: https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html

### Physics

R3 is a **linear reduced-order airway resistance/compliance network** with Poiseuille-type branch resistances, assumed distal resistances, assumed compartment compliance, a prescribed sinusoidal pressure surrogate, conservation tests and SI-unit outputs.

It is currently useful for mechanism exploration and numerical implementation. **It is not a calibrated model of any imported CT dataset.** In the immersive environment its numerical pressure/flow output is shown only on the *procedural reference* scene with `R3 REFERENCE (ASSUMED)` labeling. Imported GLB patient geometry cannot inherit R3 pressure/flow estimates by visual association.

**Future physics integration must not run unverified heavy CFD in a headset.** Use distinct numerical engines and auditable coupling:
1. Geometry-specific centreline and local cross-sections with calibrated SI units.
2. Validated boundary conditions, viscosity, flow/pressure waveforms and tissue parameters.
3. Reduced-order 0D/1D airway flow model for interactive coupled dynamics.
4. As appropriate, offline/high-performance 3D CFD in selected airway segments and deformable tissue FEM/FSI with explicit solver tolerances.
5. Optional pulmonary perfusion/gas-exchange models tied to independently measured sources, not schematic colors.
6. Numerical solution snapshots and uncertainty metadata streamed into XR without asking headset GPUs to solve entire 3D CFD volumes.

Validation requires independently measured targets and solver verification; a smooth frame rate is not a physics-validation certificate. For respiratory CFD model context, see https://doi.org/10.1140/epje/s10189-026-00583-x

### Instrument mechanics

XR-08 estimated bronchoscope clearance derives from interpolated branch radii and the selected instrument diameter. This blocks obviously impossible traversals but **does not** represent local triangle-wall contact pressure, shaft bending stiffness, friction, lesion deformation or true haptic forces. Actual force feedback requires suitable hardware and bench validation. A pointer click alone is not an instrument.

## 4. Interaction modes, inputs, and ownership

**Anatomy mode:** distance selection/summon to an accessible work zone; controller side-grip or tracked thumb/index pinch only near actual mesh triangle surfaces; one-hand anchoring and two-hand rotate/scale. The headset camera never teleports with the object.

**Procedure mode:** the original scope/camera and branch graph; trigger and controller orientation steer source scope state; constrained progress under the source-radius model; separate actual endoscope monitor. Menu, instrument and anatomy input precedence must not overlap.

**Reference physiology:** numerical readout from R3 in the same spatial panel, explicitly uncalibrated. It does not cause fabricated lung deformation.

**Imported geometry:** preserve source materials/units/provenance; never infer a named lobe or pathology from an unknown GLB vertex ID.

Current near-contact ray sampling is **geometry-aware but approximate**. It is not continuous signed-distance contact and is not computationally feasible as a universal high-resolution tissue solver. Future source surfaces will need a spatial acceleration structure such as a BVH, with contact envelopes appropriate to instrumentation, and independent device validation.

## 4.1 Reality modes and cross-device roadmap

The canonical engineering decision for assisted/augmented/mixed/virtual reality, augmented virtuality, diminished reality, hardware fallbacks and measurable future capability gates is in [XR17_REALITY_CONTINUUM_ARCHITECTURE.md](./XR17_REALITY_CONTINUUM_ARCHITECTURE.md). **XR-17 is a proposed implementation plan**, not a new parallel headset scene and not released patient-specific modeling. The same source scene, procedure state and conservative scientific gates remain mandatory.

## 5. Finite acceptance path and freeze

### Gate U1 — Unified routing and preservation (software, implemented)
- All active XR entries point to original scene; prior demos archived.
- Original full procedural workstation and R3 laboratory remain separately testable.
- No clinical/source model state is silently transmitted by navigation.
- Complete CI: Node numerical/source tests and Chromium browser smoke.

### Gate U2 — Mesh-accurate interaction and rendering (implemented foundation; hardware pending)
- Source-mesh intersection for near-grab, 6DoF controller and optional WebXR hand poses, controlled input ownership.
- Quest 2 tests: source visibility, hand occlusion recovery, controller switching, two-hand manipulation, collision false positives and exits.
- Correct stereo headset presentation, independent live endoscopic render target and GPU frame timing.

### Gate U3 — Scientific anatomy & instrument verification (not completed)
- Registered CT frame, measured source dimensions and independently checked segmentation/topology.
- Verified branch/segment identity, lung/vessel layers, object-specific collisions and instrument dimensions.
- Benchmark source geometry and device behaviors against annotated images and physical measurements.

### Gate U4 — Coupled solver validation (future research)
- Validate reduced-order and localized 3D models under measured pressure/flow conditions.
- Characterize numerical error, mesh convergence, conservation, model-form uncertainty and parameter sensitivity.
- Couple physical-state display only once the model/source identifiers and units have been independently checked.

### Gate U5 — Quest 2 validation and baseline freeze (not completed)
- Record physical comfort/legibility, input latency, 10-minute interaction, frame budget and recovery from controller/hand loss.
- Verify that any display shown as anatomy, physics or procedural measurement has an explicit source and validation level.
- Freeze the system only after hardware and scientific acceptance. Do not call it a medical device or clinical decision tool prior to appropriate evidence and regulatory assessment.

**Current claim:** an integrated, browser-tested, source-conscious research foundation. **Not yet:** verified high-fidelity lung digital twin, validated tissue mechanics, full CFD, patient-specific simulation, or hardware-proven immersive instrument trainer.
