# PulmoSIMU NLST — Research anatomy data and source QA

**Current:** [A2.1 observed QA](./a2_1_observed_qa.json) · [A2.1 verified-pair inspector manifest](./a2_1_inspector_manifest.json) · [reproducible A2.1 segmentation pipeline](../../anatomy/a2_1_correct_nlst_airspace.py)

## Critical RAS laterality correction

An earlier A2 audit incorrectly interpreted NIfTI **RAS +X as patient LEFT**, then created an unnecessary label-swapped candidate. This was wrong. The standard NIfTI RAS+ convention is **+X = patient RIGHT; −X = patient LEFT**. See [3D Slicer coordinate systems](https://slicer.readthedocs.io/en/5.10/user_guide/coordinate_systems.html) and [NiBabel coordinates](https://nipy.org/nibabel/coordinate_systems.html).

Actual area-weighted fractions of the original GLB:

| Original surface | Patient right (+RAS X) | Patient left (−RAS X) | Interpretation |
|---|---:|---:|---|
| `right_lung_airspace` | 85.19% | 14.81% | Predominantly right; significant midline crossing requires mask review |
| `left_lung_airspace` | 1.05% | 98.95% | Predominantly left; still an unvalidated source surface |

**The original names are NOT proven reversed.** The earlier [candidate manifest](./a2-candidate-manifest.json) and associated relabelled-geometry reports are retained only as historical, *invalid* artifacts; **do not publish, load for labelled anatomy or link them to a procedure**. The old `a2_ras_relabel.py` publisher is fail-closed.

## A2.1 — Corrected conservative source-airspace review

I have resegmented the supplied 512 × 512 × 150 NLST NIfTI study using local processing:
- exact source SHA-256 and sform RAS validation, source slope/intercept,
- HU airspace candidate within a per-slice enclosed torso,
- separate, source-oriented right and left mid-thorax-seeded 6-connected 3D components,
- an explicit physical RAS midline partition to stop the A1 contralateral region leaking into the wrong-side mask,
- modest sub-voxel field smoothing before extracting the **largest watertight closed component** per side.

**These are lung airspace candidates, NOT whole lung parenchymal or lobe masks.** Midline partition may cut genuine medial lung, and connected central bronchial air may remain. Eight axial CT overlay slices are generated for review; **every source slice still requires expert review**.

| A2.1 measurement | Right | Left |
|---|---:|---:|
| Estimated thresholded airspace | 1.75629 L | 1.46991 L |
| Source GLB triangles | 233,268 | 203,536 |
| Watertight research shell | Yes | Yes |
| Cross-side source-mask voxels | 0 by construction | 0 by construction |
| Independently validated anatomy | **No** | **No** |

The corrected **436,804-triangle GLB is a high-detail desktop anatomical QA source**, **not Quest-ready**. Earlier generic quadric decimation damaged manifold topology. We reject that reduced mesh and do not place this source into the original XR/bronchoscopy engine.

### Files and publication status

| File | Location / status |
|---|---|
| A2.1 reproducible script | `experimental/anatomy/a2_1_correct_nlst_airspace.py` — **GitHub** |
| A2.1 observed metrics and validity gates | `a2_1_observed_qa.json` — **GitHub** |
| A2.1 inspector manifest for the exact CT/GLB pair | `a2_1_inspector_manifest.json` — **GitHub** |
| Original A1 study manifest | `study-manifest.json` — **GitHub** |
| Original NIfTI source CT, ~75 MiB | **Not committed or distributed** |
| A2.1 3D GLB / voxel-mask NPZ / slice preview | Generated locally, **not on GitHub** |

The [CT Source Inspector](../../anatomy-inspector.html) accepts a local CT, GLB and matching JSON manifest. SHA-256 matches verify file correspondence, **not tissue-segmentation accuracy**. The exact A2.1 manifest corresponds to the locally generated GLB; do not use the old swapped candidate.

### Provenance

NLST public chest CT collection (TCIA), [DOI 10.7937/TCIA.HMQ8-J677](https://doi.org/10.7937/TCIA.HMQ8-J677), public collection license **CC BY 4.0** subject to verifying that this particular converted NIfTI belongs to that collection and contains no identifying metadata. De-identification and original DICOM series provenance have not been independently established. A2.1 does **not** prove that these data are clinically valid or cleared for any patient-specific decisions.

## Finite next acceptance gates

1. **A2.1 reviewer gate:** inspect all 150 source CT slices, mask boundary vs CT, hilum, diaphragm, apices, and medial tissue. Resolve exclusions and quantify independent reference accuracy. The centre plane is a conservative QA intervention, not a validated segmentation technique.
2. **A2.2 segmentation:** obtain independently checked five-lobe/fissure masks and true airway lumen/centreline, with source coordinates, then investigate vessel segmentation where image contrast permits.
3. **A2.3 geometry QA:** register and validate source-specific meshes, compare reference CT, topology and triangle-budget constraints for Quest. Do not damage manifold integrity to meet a visual polygon target.
4. **A3 procedure/Quest:** only after airway validation connect the original bronchoscope to source-derived geometry; assess instrument collision and actual headset performance; freeze after review.

**No clinical use, no patient-specific bronchoscopy, no unvalidated anatomy substituted into VR.**
