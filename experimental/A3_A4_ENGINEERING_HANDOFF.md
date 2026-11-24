# PulmoSIMU A3.1–A4 · Three-pass engineering handoff

**Status:** Source-backed research processing and fail-closed procedural integration implemented. **Clinical anatomical validation, full five-lobe/airway/vessel reconstruction, and physical Quest acceptance are not complete.**

Branch: `feature/mechanistic-lung-foundation` · original desktop/procedure and prior A2 files preserved.

## Pass I — A3.1 bilateral airspace and XR precision

The source CT (NLST 512 × 512 × 150, original 0.543 × 0.543 × 2 mm RAS sform) and original A2 candidate GLB were checked by exact SHA-256. Algorithm `experimental/anatomy/a31_disjoint_airspace.py` generates a three-label CT-native disjoint research volume, preserves the bilateral source union and creates two watertight topology-preserving surfaces.

**Actual local test result:** 27,702 conflicting input voxels, approximately 16.334 mL; zero overlapping voxels in the output; source union of 5,484,083 voxels preserved. Source-aligned GLB: 94,212 left and 96,634 right triangles (190,846 total), both watertight.

Data contract and hashes: `experimental/data/nlst-a1/a31-study-manifest.json` and `a31-source-QA.json`. Binary input and output are **not committed to normal Git history**. The original mask's residual midline leakage is **not** considered corrected.

XR15 source-preserving view interaction: original Quest controller tracking and bounded 6DoF grab; additional undo/redo, selectable yaw/pitch/roll and centre/origin view pivot. All are viewing transforms, not tissue deformation.

## Pass II — A3.2–A3.4 five lobes, airway lumen and vessels

**Built:** `experimental/anatomy/a32_a34_source_mask_qa.py` validates separately supplied source-coframed .npz files with `mask[k,j,i]` and matching `affine_ras[4,4]`.

Required actual source masks: `RUL`, `RML`, `RLL`, `LUL`, `LLL`, `airway_lumen`, `pulmonary_artery`, `pulmonary_vein`. The QA checks grid alignment, nonempty masks, 3D component count, side affiliation where appropriate, and lobar/vessel overlap.

**Built:** `experimental/anatomy/a32_a34_registered_surfaces.py` creates separate registered 3D GLB structures **only** for the masks supplied. Missing anatomy is never synthesized to satisfy a feature checklist.

**Current scan status:** no expert-reviewed five-lobe masks; no reviewed bronchial lumen/centreline; no reviewed pulmonary arterial or venous masks. The checked result `experimental/data/nlst-a1/a32-a34-source-mask-QA.json` correctly declares these missing and blocks CT-derived procedures. No automatic CT lobes/vessels are claimed.

## Pass III — A4 source inspection and XR procedure contract

**Built:** `experimental/anatomy/a4-source-gate.js`: file hash/RAS matching may permit *source inspection*; never equates this to validated anatomy or patient-specific bronchoscopy. Self-attested browser review strings cannot unlock CT-specific procedure in the current engine.

**Live CT Inspector:** local NIfTI + GLB + manifest matching; source mesh component selection and isolation; reversible opacity; source-registered axial/coronal/sagittal clipping; optional local source-mask QA JSON import with A3.1–A4 stage status. No CT bytes uploaded by this workflow.

**Original immersive engine:** loaded patient/CT geometry cannot start the original reference bronchoscopy under a patient-specific claim. Users can continue the **separate, original reference bronchoscopy** with its procedural-reference anatomy. Its existing source/scope and controller interactions remain preserved.

**NOT PROVIDED:** validated CT airway centreline, calibre/wall model, device geometry and physical Quest 2 verification. The research source cannot become a patient-registered clinical procedure by a browser UI switch.

## User data and review

Actual local research package generated in this session:
- `PulmoSIMU_A31_Disjoint_Airspace_Candidate.glb`
- `PulmoSIMU_A31_Disjoint_Source_Masks.npz`
- `PulmoSIMU_A31_Study_Manifest.json`
- `PulmoSIMU_A31_Disjoint_QA.json`
- Original `NLST_Chest_CT_150_Slices_R21(1).nii` and A2 candidate source GLB
- A3.2–A3.4 source-mask QA, source surface-builder, reviewer instructions, and synthetic unit tests.

For the actual study: open `https://pulmosimu-xr-preview.vercel.app/experimental/anatomy-inspector.html`, choose A3.1 GLB, the matching unmodified CT NIfTI and A3.1 study manifest. Use the new opacity and multiplanar source clipping. Optional structure QA can be loaded if the independently supplied masks exist.

### Exit gates

| Area | Engineering | Data / validation |
| --- | --- | --- |
| A3.1 bilateral source airspace | Implemented and computed | Mask boundary review pending |
| A3.2 five lobes | Input, QA and mesh-generation pathway implemented | **Actual reviewed five-lobe masks absent** |
| A3.3 bronchial tree | Source lumen intake and QA implemented | **Actual centreline/topology absent** |
| A3.4 arteries/veins | Independent mask intake and surface generation implemented | **Actual source vessel masks absent** |
| A4 CT procedure | Inspector and fail-closed integration contract implemented | **CT-specific bronchoscopy prohibited; Quest physical acceptance pending** |
| XR15 precision | Software implemented and testable | Physical Quest headset acceptance pending |

The A2 corrective mesh, original CT and A3.1 research assets are not yet public GitHub Release binaries. Source TCIA series correspondence/de-identification and redistribution still require independent confirmation. Do not claim clinical validation based on geometry rendering or watertight meshes alone.
