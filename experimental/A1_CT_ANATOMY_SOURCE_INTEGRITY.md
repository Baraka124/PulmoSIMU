# A1 — CT Anatomical Source Integrity and Research Inspection

**Implementation:** `experimental/anatomy-inspector.html`, `experimental/anatomy/nifti-source.js`, `experimental/anatomy/inspector-app.js`; accessible through Studio > **CT anatomical source**. Source study files are **not deployed or included in this repository**.

## Contract

- Import **local** 3D uncompressed `.nii`, local `.glb` and a study-specific reconstruction `.json` manifest. Files remain in the browser session; the source inspector performs no study fetch, upload or persistent storage.
- Support NIfTI-1 with explicit **sform RAS coordinates**, voxel spacing, scalar datatype, slope/intercept and axial/coronal/sagittal windowed image views. Ambiguous 4D data or unregistered NIfTI inputs are explicitly rejected.
- Preserve the original GLB mesh coordinates, source materials and source element naming. Do not render invented vessels, false fissures or simulated disease over the imported source.
- Only activate 3D registered crosshair and source-plane clipping after all of these pass: NIfTI dimensions, sform and intensity calibration **AND exact SHA-256 hashes for the NIfTI and GLB** equal the locally selected study manifest. Even exact matching is not evidence of segmentation or clinical accuracy.
- No automatic adaptation of the original `SEGS`/`SEG_MAP` bronchoscopy engine. A bilateral lung airspace shell is **not** an endoscopically navigable airway.
- Do not silently substitute a candidate CT mesh into the original immersive procedural scene or convert a research surface into a clinical geometry.
- The complete private analysis file set, including image data and study-specific manifest, must remain outside public GitHub and Vercel deployments.

## Baseline research limitations

The first preliminary reconstruction was obtained by nonvalidated HU thresholding plus 2D connected-component selection. It captures bilateral air-filled lung regions, but can include non-lung air and miss high-attenuation or consolidated lung. A limited area of the provisional right surface appears to cross a display-side boundary and requires manual image review. Original CT slice thickness is greater through-plane than in-plane. Watertightness, reduced triangle count and similar enclosed volume **do not** demonstrate anatomical truth.

## Validation states

| Gate | Status |
| --- | --- |
| Local NIfTI file import, sform and HU handling | Implemented; browser/device checks required |
| Source GLB local import with existing materials | Implemented; browser/device checks required |
| Original image vs GLB file hash match | Implemented |
| Reproducible RAS → model coordinate transform | Manifest-driven; study-specific transform subject to independent QA |
| Review of lung airspace masks against all original CT slices | **Pending expert review** |
| Correct bilateral side labeling and segmentation leakage | **Flagged for review** |
| Validated five-lobe boundary and fissure surfaces | **Not available** |
| Segmented bronchial lumen and central airway graph | **Not available** |
| Segmented pulmonary arteries and veins | **Not available** |
| Bronchoscope registration, contact and instrument clearance | **Blocked** |
| Patient-specific physics, CFD, mechanical motion | **Blocked** |
| Physical Quest 2 acceptance for CT-derived anatomy | **Not performed** |
| Public release of the supplied CT/GLB | **Prohibited** |

## Finite implementation path

**A1 (current):** Finish source inspection, expert mask review, orientation/label audit, segmentation leakage correction, boundary/volume QA.

**A2:** Use a supported, validated segmentation workflow to obtain independently reviewed lobar masks, central airway lumen and pulmonary vessels. Maintain explicit source segmentation and anatomical uncertainty.

**A3:** After verified source airway topology/registration, connect the original bronchoscope to the new geometry with collision and instrument clearance constraints. Physically test controller interactions on Quest 2, capture performance and clinical validation evidence, then freeze the accepted scope. No procedural migration before this gate.

This source inspector requires Three.js libraries loaded from a CDN; offline graphics and instrument interoperability are **not** yet verified. It is research software, not a medical device.
