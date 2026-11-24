# A2.3 — CT Source Review Workbench: software completion

**Branch:** feature/mechanistic-lung-foundation  
**Scope:** research-only, local anatomical review; no patient-specific procedure or validated fissure segmentation.

## Delivered

The Source Anatomy Laboratory at experimental/anatomy-inspector.html now carries a review strip beneath its existing three-plane CT viewer. Once the exact local NIfTI CT, GLB surface and study manifest match via source hashes and header check, the researcher can:

1. Move the shared CT crosshair using axial, coronal and sagittal views.
2. Select a proposed lobe (RUL, RML, RLL, LUL, LLL or unassigned), observation type and optional note.
3. Record a source-voxel point with computed RAS millimetre coordinates from the original NIfTI sform.
4. See point overlays in source MPR on the exact voxel slice, relocate to any observation, or remove it.
5. Explicitly export an in-memory review as JSON or import it back for the same CT/GLB pair.

Contract: experimental/anatomy/source-review.js, schema pulmosimu-ct-point-review-v1.

## Source integrity and privacy

- Every accepted review file is bound to both exact SHA-256 hashes, source dimensions and the complete 4×4 source RAS sform.
- Import rejects forged point coordinates, out-of-bounds voxels, unsupported region/finding codes, duplicate IDs, excessive note length and more than 500 marks.
- The historical A2 label-swapped CT candidate is excluded. The original imperfect A1 surface can still be inspected and flagged for review.
- CT and GLB remain unchanged. No mask editing, mesh topology changes, procedural permissions, embedded patient metadata, automatic browser storage or server upload.
- Exports contain manually entered notes, review labels, coordinates and study fingerprints only. Do not include patient identifiers in notes; recipients must treat them as research files.
- Original bronchoscopy, Quest controls, physiology and original index.html are untouched by this A2.3 pass.

## Verification

- tests/a23-ct-review.test.mjs checks source/affine binding, voxel conversion, import/export roundtrip and invalid-file rejection.
- tests/browser-smoke.mjs checks UI controls and their disabled initial state, without loading patient data.
- GitHub Actions runs existing Node and Chromium checks for this branch.

## Finite acceptance and handoff

**Software-complete point review does not mean anatomically validated segmentation.**

1. Obtain the exact source CT/GLB and independently reviewed five-lobe masks or fissure reference labels, aligned to the NIfTI sform. Manual points are observations, not masks.
2. Expert-review all source slices including apices, hilum, diaphragm, pleural boundary and midline; compare source-derived surface and reviewed masks quantitatively.
3. Generate topology and registration metrics and Quest-specific error-bounded LOD from actual input files. Reject indiscriminate polygon decimation.
4. If validated bronchial lumen and centreline become available, carry out A4 instrument collision and physical Quest 2 acceptance before any CT-specific procedure.

**Freeze rule:** do not add fictitious tissue detail, segmented fissures, vascular labels or clinically calibrated physics to substitute for absent clinical input. The bounded research reviewer is usable now; further source/hardware acceptance is external.
