# PulmoSIMU R6 — Prototype Integration & Verification Gate

**Status:** R1–R5 experimental software, subject to GitHub CI. This document is a *software prototype freeze*, **not** anatomical, physiological, pharmacological, or clinical validation.

## Canonical boundaries

- The existing PNEUMO·VR advanced procedure remains implemented in root `index.html`, with its original navigation, gesture controls and bronchoscopic experience. R1–R5 do **not** rewrite it.
- The CT/DICOM processing implementation remains in `pipeline.py`, `server.py`, `synthetic.py`.
- The new modular experimental foundation remains under `experimental/` and is deliberately not linked from production navigation until a clinical/engineering review authorizes integration.
- The original airway reference is a **versioned snapshot** with 18 proximal/intermediate branches, not complete human distal lung anatomy.
- The 3D lobar envelope is schematic; it must not be described as anatomically exact.

## Verification matrix

| Check | Automated gate | Limits |
|---|---|---|
| Anatomical graph connectivity and lobe mapping | Node tests | Source procedural snapshot, not CT registration |
| Numerically finite simulated pressures, flows, tidal volumes | Node tests | Uncalibrated reduced-order linear RC approximation |
| Network mass-flow conservation | Node tests | Does not establish in-vivo validity |
| Disease localization and mechanism state isolation | Node tests | Simplified exploratory mechanisms |
| Treatment counterfactuals, zero-effect invariants, structural tumor preservation | Node tests | Assumed pathway sensitivity, not efficacy |
| Original procedure source unchanged versus `origin/main` | GitHub source-diff guard | Doesn't prove procedure works at runtime |
| Browser opens experimental lab, airway data, overlays, treatment and output changes | Chromium/Playwright smoke | Requires network access for Three.js CDN, tests functionality not image quality |
| Original procedure page basic DOM load, original view canvases present | Chromium/Playwright smoke | Not a full interactive bronchoscopic procedure regression |
| Three-dimensional anatomy and shading visual quality | **Manual pending** | Independent expert inspection required |
| Patient DICOM pipeline on real de-identified reference CT | **Not yet performed** | Segmentation dependency verification, registration and geometry units required |
| Literature-calibrated physiological/PKPD parameters | **Not yet performed** | Never use prototype output to guide treatment |

## Freeze definition

The initial **R1–R6 experimental software prototype** may be marked *engineering freeze candidate* after all automated CI jobs are green, verified browser capture inspection, and manual original-procedure regression. It is *not yet valid* to mark the platform as a scientific/clinical digital twin.

## Scientific gates before further disease expansion

1. Acquire licensed/consented or de-identified reference thoracic CT with segment/lobe contours and consistent patient-coordinate orientation, physical voxel spacing and provenance.
2. Validate segmentation/canonical anatomy mapping against imaging reference; distinguish directly imaged geometry from algorithmically inferred distal structures.
3. Quantify model sensitivity to airway radius, compliance and distal resistance assumptions; compare calculated waveforms with representative bench or clinical reference measurements.
4. Replace demonstrative phenotype/drug parameters with curated, indication-specific literature-supported evidence including appropriate uncertainty. Never infer individual efficacy from a pathway label.
5. Freeze model versions, validation datasets and regression outputs before publishing performance claims.

**After this software freeze, further work should proceed as explicitly scoped and independently testable increments, rather than an unbounded phase sequence.**

## Post-R6 engineering validation increment — anatomy provenance and sensitivity

This increment **does not reopen** the finite R1–R6 feature roadmap; it addresses two outstanding verification gates.

1. `anatomy/ct-metadata-adapter.js` interprets the original PNEUMO·VR `pipeline.py` JSON metadata format, checks positive scaling, validates a single rooted tree, normalizes arbitrary source airway IDs, and computes **metres per display-scene unit as `1 / transform.scale`**. It includes reversible scene⇄patient-mm coordinate conversion. Input patient identifiers and pathology metadata are **not propagated** into the adapter output. There is no automatic client-side patient upload or external sharing.
2. `core/sensitivity-analysis.js` runs an explicit one-factor-at-a-time ±20% numerical stress test of assumed scene scale, gas viscosity, unmeasured distal resistance and respiratory compliance, ranking the effect on modeled tidal volume. The experimental canvas exposes an **on-demand** sensitivity inspector so it does not add unnecessary visual or runtime burden.

**Important:** The source reference model is *still not converted to measured anatomy*. CT metadata can be mapped to measured SI units **only when the source patient-derived transform exists and its scale is trustworthy**. The R3 solver continues to use reference parameters until a separate, independently reviewed CT-specific region-mapping adapter is built. The R2 graph's fixed-ID lobe mapping is intentionally NOT reused for patient-derived CT IDs. This avoids silently assigning an incorrect lobe or calculating a misleading physiological result.

The sensitivity result is a mathematical stress test around assumed inputs, **not a clinical confidence interval** or validation against spirometry. Its purpose is to reveal dependence on modelling assumptions before anyone interprets simulated changes.

## Post-freeze CT geometry inspection increment (source-faithful scene)

- The research canvas now accepts a **locally selected** matching pair: `<patient>_airway.glb` and `<patient>_meta.json` from the established PNEUMO·VR CT pipeline, with size limits (GLB 100 MB, JSON 20 MB).
- The GLB is parsed in browser memory and displayed **at the original exported scene scale**, in a separate Three.js group. The schematic reference lobes and airway geometry are hidden in CT mode, not overlaid misleadingly onto the patient frame.
- A basic spatial check rejects GLB bounding boxes that are manifestly incompatible with the metadata centerlines. This is a *plausibility check only*: files from different studies can still have similar coordinates. It is not identity verification, anatomical registration validation or segmentation quality assurance.
- The CT graph overlays derived segment centerlines and provides clickable, original source branch IDs (not inferred lobe names). The centreline radii and region labels are **unverified** and not connected to the source-procedural lobe identity rules.
- `Return to reference` clears loaded scene geometry from the viewer and restores the previous schematic model. A rejected replacement leaves the previously loaded source intact.
- **R3–R5 still run only on the procedural reference airway graph**, and the inspector explicitly says their numerical results are **NOT computed from the loaded CT**. Patient-specific lung physiology, pharmacodynamics and cancer progression have not been implemented.
- A synthetic GLB/JSON browser integration test exercises local import, source provenance, numerical isolation, mismatched-pair rejection and return-to-reference. Automated validation uses synthetic geometry only, not real patient files.

**Privacy/security:** No file upload is implemented in this mode, and file content is not posted to our backend by this code. However, the experimental page currently imports Three.js modules from a third-party CDN; **do not load identifiable clinical data in the public experimental page**. Before handling identifiable or otherwise sensitive imaging data, move dependencies to vetted self-hosted assets, apply a restrictive CSP, complete privacy/security review and use only an appropriately controlled clinical deployment. For development, use non-identifiable, appropriately permitted image examples.

**Unresolved anatomical fidelity:** The current CT display is an airway segmentation surface, **not a complete lung including lobes, pleura, vascular beds, alveoli or tumor microenvironment**. Extending it to a validated whole-lung reconstruction requires curated imaging references and distinct tissue segmentations, not extra procedural rendering alone.

## Whole-lung anatomical atlas — structured topology increment

The new `anatomy/whole-lung-atlas.js` is an explicit **anatomical terminology and relationship** source. It defines five lobes, eighteen *commonly enumerated* bronchopulmonary segment units (10 right, 8 left, with S1+2 and S7+8 composite terminology on the left), three fissure identities (right oblique, right horizontal, left oblique), and six major pulmonary arterial/venous trunk landmarks. Segment naming conventions and left bronchial patterns vary between people. The data represents known typical relationships only; it is **not** a geometric lung atlas or segmentation of actual tissue.

The experimental laboratory offers a contextual atlas inspector showing a selected lobe's related segment names, fissures and named pulmonary arterial/venous connections. When CT-derived mesh inspection is active, the procedural-reference atlas inspector is hidden to prevent implying that CT branches have already been mapped to those segment IDs.

### Precise scope boundary
This feature deliberately **does not invent** 3D vessel trajectories, patient fissure locations, alveoli, vascular caliber, distal airway centerlines, realistic lobar surfaces or cancer staging. The 3D reference surfaces remain schematic. The anatomical relationship registry can later serve as the label schema for properly licensed, annotated lung CT segmentations or validated whole-organ 3D models. Before connecting any of those datasets, require image orientation, physical voxel spacing, verified patient consent/rights, robust lobe/segment and vessel correspondence, and anatomical expert review.

## Anatomical surface-atlas loading gate — licensed external meshes

The experimental laboratory now supports a **third mutually exclusive source mode**, separate from both procedural schematic lobes and CT-derived airway segmentation: a non-patient-specific surface atlas with an explicit `pulmosimu-surface-atlas-v1` JSON manifest and a corresponding multi-object GLB. This enables actual imported meshes for known lobar and pulmonary vessel structures **when the operator supplies them with appropriate rights and anatomical provenance**. It does not generate anatomy from names.

The manifest contract lives in `anatomy/surface-atlas.js`. Every registered mesh must map one GLB object name to one declared known lobe or major vessel ID, with its geometry origin, evidence note, license, rights statement, coordinate-frame name, and physical unit (`mm` or `m`). Synthetic browser fixtures use authored boxes/cylinders and are explicitly not anatomical references. Input marked `patientSpecific: true` is rejected in this research-only mode.

Example for a *single* surface object (not a real dataset):

```json
{
  "schema": "pulmosimu-surface-atlas-v1",
  "datasetId": "example-nonclinical",
  "source": "Insert verified dataset credit",
  "license": "Insert actual license",
  "rightsStatement": "Document reuse rights",
  "coordinateFrame": "source-local",
  "unit": "mm",
  "patientSpecific": false,
  "structures": [{
    "id": "RUL",
    "kind": "lobe",
    "objectName": "mesh_right_upper_lobe",
    "geometryOrigin": "segmented",
    "evidence": "Provide curation/segmentation reference",
    "sourceValidated": false
  }]
}
```

The canvas renders registered structure meshes in their own scene container, with independent lobar and vascular visibility, source details, coverage and a structure picker. It applies a **display-only fit transform** for camera framing; it does not reinterpret the GLB as registered to the procedural or CT frame. The original physical unit is retained in the manifest and inspector, not inferred from display scale. All R3–R5 numerical outputs remain based on the separate reference airway graph and are explicitly labeled **not computed from the loaded surfaces**.

A syntactically valid manifest and matching object names **cannot certify anatomical correctness or same-frame alignment**. The loader does not infer missing lung lobes, fissure surfaces, missing vessel branches or pulmonary physiology. Automated testing covers declared object coverage, missing mesh names, identity consistency, unit provenance, simulated browser import, visibility controls, rejected patient-specific replacement and return-to-reference. The code includes no upload endpoint; nevertheless, external CDN dependencies and an unreviewed public client mean **identifiable patient data must not be loaded**.

To advance from this loader to scientifically valid organ geometry, choose an appropriately licensed anatomically segmented reference CT/MRI atlas with five lobar surfaces and independently validated vessel segmentation, verify its coordinate frame and orientation, and complete a visual/quantitative morphology review. Merely importing a GLB is **not** that validation.
