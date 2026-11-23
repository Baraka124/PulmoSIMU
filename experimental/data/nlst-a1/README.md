# NLST A1 — Public Research Data Index

This folder accompanies the **PulmoSIMU CT Anatomical Source Laboratory**:
[Open the public inspector](../../anatomy-inspector.html).

## Files and status

| Data asset | Distribution status |
|---|---|
| `study-manifest.json` — source affine, HU calibration, SHA-256 and A2 quality gate | **Committed** |
| `a2-laterality-qa.json` — measured source-side naming conflict | **Committed** |
| `a2_laterality_qa.py` — reproducible fail-closed research audit | **Committed under experimental/anatomy** |
| `NLST_Chest_CT_150_Slices_R21(1).nii` — 75 MiB source volume | **Not attached to GitHub yet** |
| `PulmoSIMU_NLST_CT_Lung_Quest_Research.glb` — ~1 MiB preliminary bilateral airspace surface | **Not attached to GitHub yet** |

**This manifest is research data, not the raw CT or binary surface itself.** GitHub releases (assets) are the preferred distribution route for the binary files; do not add a 75 MiB medical image to normal Git history. Use the opt-in verification helper `publish-a1-release.ps1` from a Windows machine holding the exact two files. Neither the research inspector nor this repository uploads medical data automatically.

## Provenance and data use

Source collection: **National Lung Screening Trial (NLST)**, National Cancer Institute, hosted by **The Cancer Imaging Archive (TCIA)**, [doi:10.7937/TCIA.HMQ8-J677](https://doi.org/10.7937/TCIA.HMQ8-J677).
TCIA lists the public NLST radiology CT images under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, subject to proper citation. See https://www.cancerimagingarchive.net/collection/nlst/ and https://creativecommons.org/licenses/by/4.0/.

The NIfTI header identifies TCIA NLST and its voxel calibration / RAS transform. We have not independently re-established the NIfTI's exact original TCIA DICOM series ID, nor completed an independent de-identification audit of the original dataset lineage. **Redistribute only if the specific scan belongs to the CC BY 4.0 public NLST collection and does not contain identifying metadata.** The author of any transformation must identify its changes, avoid implying endorsement by TCIA, and retain the citation.

## A2 finding — anatomical laterality QA FAILURE

The source-image/GLB cross-sectional review has now exposed a **critical laterality-name conflict**. In the original GLB's physical RAS coordinate frame, the geometry named `right_lung_airspace` has **85.2%** of its triangle surface area in patient-left RAS X-positive coordinates. The surface named `left_lung_airspace` has **99.0%** of its surface area in patient-right RAS X-negative coordinates.

The evidence is recorded in [a2-laterality-qa.json](./a2-laterality-qa.json) and recomputed by [a2_laterality_qa.py](../../anatomy/a2_laterality_qa.py). The CT and mesh file fingerprints must match before the validator processes them. An original-source 4-slice CT/GLB overlay can be generated locally from the research files and accompanies the ChatGPT A2 review package.

**Do not interpret the original GLB's `left`/`right` names as verified patient anatomy.** The meshes themselves are kept unchanged to preserve reproducibility, and the public inspector now warns about this known-bad fingerprint even when loading an older manifest. CT-linked bronchoscopy, clinical labels and patient-specific physics remain **blocked**. A reviewer must confirm and correct the source side mapping and mask quality before any upgrade.

## Derived model and quality boundaries

`study-manifest.json` records exact CT/GLB SHA-256 digests, NIfTI sform, model-from-RAS affine, HU slope/intercept, airspace volume estimates, and optimized mesh face counts. The derived mesh comes from HU thresholding, two-dimensional component selection and smoothing. It has not been reviewed for five-lobe boundaries, pulmonary vessels, clinically correct lung masks or airway lumen. There is a flagged possible right-sided mask leakage across the centre. CT slice spacing is 2 mm.

**Not clinically validated.** Bilateral airspace geometry is not a navigable bronchial tree. The canonical PulmoSIMU bronchoscopy remains a separate authored procedural model until source registration and anatomical QA pass.

## Local binary publication (optional)

On Windows, install [GitHub CLI](https://cli.github.com/) and sign in with `gh auth login`. Download the latest private A1 study kit from your ChatGPT working session and keep the original .nii available locally. Run the checked publisher script from this directory as follows:

```powershell
powershell -File .\publish-a1-release.ps1 -SourceNifti "C:\path\NLST_Chest_CT_150_Slices_R21(1).nii" -MeshGlb "C:\path\PulmoSIMU_NLST_CT_Lung_Quest_Research.glb"
# When the dry run passes, repeat the command with -Publish at the end.
```

The script verifies both exact hashes, requires an explicit opt-in, and creates a GitHub prerelease containing the two binaries and manifest. **Do not infer that a release exists until you have checked its actual GitHub release URL and assets.**
