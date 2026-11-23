# NLST A1 — Public Research Data Index

This folder accompanies the **PulmoSIMU CT Anatomical Source Laboratory**:
[Open the public inspector](../../anatomy-inspector.html).

## Files and status

| Data asset | Distribution status |
|---|---|
| `study-manifest.json` — source affine, HU calibration, segmentation metrics, SHA-256 hashes, limitations | **Committed in this folder** |
| `NLST_Chest_CT_150_Slices_R21(1).nii` — 75 MiB source volume | **Not attached to GitHub yet** |
| `PulmoSIMU_NLST_CT_Lung_Quest_Research.glb` — ~1 MiB preliminary bilateral airspace surface | **Not attached to GitHub yet** |

**This manifest is research data, not the raw CT or binary surface itself.** GitHub releases (assets) are the preferred distribution route for the binary files; do not add a 75 MiB medical image to normal Git history. Use the opt-in verification helper `publish-a1-release.ps1` from a Windows machine holding the exact two files. Neither the research inspector nor this repository uploads medical data automatically.

## Provenance and data use

Source collection: **National Lung Screening Trial (NLST)**, National Cancer Institute, hosted by **The Cancer Imaging Archive (TCIA)**, [doi:10.7937/TCIA.HMQ8-J677](https://doi.org/10.7937/TCIA.HMQ8-J677).
TCIA lists the public NLST radiology CT images under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, subject to proper citation. See https://www.cancerimagingarchive.net/collection/nlst/ and https://creativecommons.org/licenses/by/4.0/.

The NIfTI header identifies TCIA NLST and its voxel calibration / RAS transform. We have not independently re-established the NIfTI's exact original TCIA DICOM series ID, nor completed an independent de-identification audit of the original dataset lineage. **Redistribute only if the specific scan belongs to the CC BY 4.0 public NLST collection and does not contain identifying metadata.** The author of any transformation must identify its changes, avoid implying endorsement by TCIA, and retain the citation.

## Derived model and quality boundaries

`study-manifest.json` records exact CT/GLB SHA-256 digests, NIfTI sform, model-from-RAS affine, HU slope/intercept, airspace volume estimates, and optimized mesh face counts. The derived mesh comes from HU thresholding, two-dimensional component selection and smoothing. It has not been reviewed for five-lobe boundaries, pulmonary vessels, clinically correct lung masks or airway lumen. There is a flagged possible right-sided mask leakage across the centre. CT slice spacing is 2 mm.

**Not clinically validated.** Bilateral airspace geometry is not a navigable bronchial tree. The canonical PulmoSIMU bronchoscopy remains a separate authored procedural model until source registration and anatomical QA pass.

## Local binary publication (optional)

On Windows, install [GitHub CLI](https://cli.github.com/) and sign in with `gh auth login`. Download the latest private A1 study kit from your ChatGPT working session and keep the original .nii available locally. Run the checked publisher script from this directory as follows:

```powershell
pwsh -File .\publish-a1-release.ps1 -SourceNifti "C:\path\NLST_Chest_CT_150_Slices_R21(1).nii" -MeshGlb "C:\path\PulmoSIMU_NLST_CT_Lung_Quest_Research.glb"
```

The script verifies both exact hashes, requires an explicit opt-in, and creates a GitHub prerelease containing the two binaries and manifest. **Do not infer that a release exists until you have checked its actual GitHub release URL and assets.**
