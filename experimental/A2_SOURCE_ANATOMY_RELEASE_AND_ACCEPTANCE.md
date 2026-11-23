> **SUPERSEDED RAS LATERALITY INTERPRETATION (23 Nov 2026):** This historical A2 candidate document was written with an inverted NIfTI RAS X convention. Correct convention: **+X patient RIGHT, −X patient LEFT**. The candidate created by swapping mesh labels is **INVALID and must not be published or used**. Original right GLB predominantly occupies patient right (+X 85.19% surface area), but its 14.81% opposite-hemisphere area requires expert segmentation review. See [corrected source data index](data/nlst-a1/README.md) and [A2.1 observed QA](data/nlst-a1/a2_1_observed_qa.json). Statements below about verified left/right label swapping are retained only as historical context and are not valid conclusions. CT-linked bronchoscopy remains blocked.

---

# PulmoSIMU A2 — Computed anatomy release and remaining acceptance gates

**Status:** versioned computational A2 candidate and CT-derived QA metrics complete; **clinical anatomical validation and patient-specific procedure are NOT complete**. Original PulmoSIMU bronchoscopy remains an independent reference simulation. No source file was overwritten.

## Delivered on `feature/mechanistic-lung-foundation`

| Deliverable | State |
|---|---|
| Original NLST NIfTI header / SHA-256 check | Reproducible |
| Original sphere-based procedural lung | Preserved, not relabelled patient-specific |
| Original CT-derived GLB (`5e445700…`) | Historical source, left/right **incorrectly named** |
| A2 candidate GLB (`8f8bea9a…`) | New binary generated locally; left/right **node names** corrected via RAS (no geometric edits) |
| Candidate manifest / source matrix / SHA-256 | In GitHub (`a2-candidate-manifest.json`) |
| RAS relabelling QA / voxelized comparison | In GitHub (`a2-candidate-qa.json`, `a2-candidate-voxel-qa.json`) |
| Reproducible Python relabelling and CT voxel-stencil audits | In GitHub (`experimental/anatomy/`) |
| Inspector warnings for old and new GLB | Integrated; no clinical validation implied |
| Binary public GitHub Release upload | **Not yet executed**; use checked opt-in `publish-a2-release.ps1` after source/license/de-ID audit |

## Source / candidate measurements

The candidate's **patient-left surface** is the source node originally named `right_lung_airspace`; it contains **27,498** triangles and approximately **1.774 L** enclosed volume. The candidate's **patient-right surface** is the former `left_lung_airspace`; it contains **22,690** triangles and approximately **1.473 L**. Vertex positions and all faces are preserved exactly.

The patient-left candidate retains **14.8% triangle surface area across the RAS midline**. In voxelized CT space this is **1.06% of enclosed voxels**. The candidate right has **0.132%** of enclosed voxels on the opposite side. These distinct surface/volume measures must not be conflated.

Both candidate masks overlap by **27,702 voxels, about 16.33 mL** at the CT's 0.543 × 0.543 × 2 mm spacing. These are calculated with the original image sform and explicit model-to-RAS transform.

97.4% and 97.8% of voxels enclosed by the candidate left/right meshes are in the original HU-derived airspace range. **That is a consistency check against the threshold used to create these masks—not independent proof of anatomical accuracy.**

## Clinical and engineering approval matrix

| Gate | Required evidence | Current result |
|---|---|---|
| **A2.1 Source identity** | SHA-256, sform and HU calibration match | **PASS** for inspected files |
| **A2.2 Anatomical laterality** | Correct source RAS naming and independent radiologist/clinician review | **Software candidate produced; clinical review PENDING** |
| **A2.3 Airspace mask accuracy** | Full axial/coronal/sagittal visual review, quantified overlap correction, exclusions and expert approval | **FAIL / OPEN** |
| **A3.1 Five pulmonary lobes** | Clinician-reviewed lobe masks registered to source CT | **NOT AVAILABLE** |
| **A3.2 Bronchial lumen / centreline** | Segmented airway lumen, complete topology and source registration | **NOT AVAILABLE** |
| **A3.3 Pulmonary arteries/veins** | Independently reviewed source-specific vessel masks | **NOT AVAILABLE** |
| **A4.1 Patient-specific bronchoscopy** | Validated bronchi, instrument diameter, wall contact/collision and procedure QA | **BLOCKED** |
| **A4.2 Quest hardware gate** | Stereo, performance, input accuracy, comfort, recovery, reproducibility in physical Quest 2 | **NOT PERFORMED** |
| **Public raw-data release** | Confirm exact TCIA series provenance, de-identification, license and verified GitHub Release assets | **NOT PUBLISHED** |

A binary uploaded without expert validation **must still be labelled UNVALIDATED research**. CT source provenance/redistribution permission and clinical approval are independent requirements.

## How to review

1. Download the complete A2 research kit (original CT, historical GLB, candidate GLB, candidate manifest, two QA reports).
2. Open [CT Anatomical Source Laboratory](https://pulmosimu-xr-preview.vercel.app/experimental/anatomy-inspector.html).
3. Load **the candidate GLB**, then source NIfTI, then the matching `PulmoSIMU_A2_Candidate_Study_Manifest.json`. The source matcher should accept exact hashes, while the clinical status remains pending.
4. Inspect CT MPR and mesh across cranial, middle, and caudal slices. Record any non-pulmonary air, midline violations and mask cutaways.
5. Review the original source image against an independent lung segmentation/reference label. Only a controlled geometry correction plus sign-off can close A2.3.
6. Do **not** substitute these bilateral airspace surfaces for registered endobronchial geometry.

## Reproducibility

Install in a Python environment: `numpy`, `trimesh`, `vtk`. Scripts:
- `experimental/anatomy/a2_ras_relabel.py` makes a new side-labelled GLB while verifying source hashes and topology.
- `experimental/anatomy/a2_ct_surface_voxel_qa.py` verifies source CT registration, voxelizes the candidate and computes threshold/overlap/side metrics.
- `experimental/anatomy/a2_laterality_qa.py` audits original source nomenclature and fails closed.

The scripts read local imaging data; no CT is uploaded to a server by the Inspector.

## Finite next work (external evidence required)

**A3:** provide or produce independently reviewed lobar, airway, and vessel masks aligned to source CT. Correct leakage with explicitly documented masks and compare candidate vs edited geometry quantitatively.

**A4:** only after verified airway masks exist, link the CT-specific centreline to the original procedural coordinate system, test instrument collision and diameter/clearance and measure Quest performance physically. Do not falsely mark calibrated physiology complete from code-only tests.

**Release:** once the exact original NLST series/license/de-identification check is complete, run the dry-run-first publication script and confirm the resulting GitHub Release asset URLs. Ordinary Git code commits alone do not upload the binaries.
