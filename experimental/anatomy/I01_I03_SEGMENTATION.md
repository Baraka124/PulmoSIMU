# I01–I03 | Segmentation evidence and topology foundation

Use the existing Source Anatomy inspector. The original procedure scene and A4 gate are unchanged.

## Offline model adapter (I01)
Run an external segmentation engine or manually segment an authorized local CT. VISTA3D, specialist airway segmentation and clinician-reviewed masks can be adapted **only** by exporting a binary NIfTI for each desired structure in the exact original CT voxel grid. No model weights, inference service, MONAI runtime, automatic label mapping, or trained airway network are shipped. Engine/version are operator claims, not authenticated provenance. For .nii.gz mask input the converter reads through nibabel; the CT supplied to the browser must be the exact uncompressed .nii hashed by the converter.

Install Python 3.11+ dependencies numpy, nibabel, scipy and scikit-image (with skeletonize method lee support). Run:
```sh
python experimental/anatomy/segmentation_pipeline.py source.nii airway.nii.gz local-airway --structure airway_lumen --engine "manual or model name" --version "actual version"
```
Never add patient files or generated masks to Git. Reject frame or grid mismatches rather than silently resample; no DICOM segmentation object support in this increment.

## I02 topology
The offline report uses six-connected components, component volumes, largest-component fraction and disconnected voxel count. For airway masks it thins only the largest face-connected component using Lee skeletonization and exports a 26-neighbor voxel graph with physical RAS coordinates and edge lengths. Junction voxels may cluster or form shortcuts/cycles, and missing distal bronchi cannot be inferred. This is not a rooted, pruned, named or validated navigation tree. Do not use the graph for bronchoscopy. Inspect topology report locally; no 3D graph viewer in this increment. Empty masks have no skeleton.

## I03 correction
Load source CT into existing inspector, then mask.json and mask.raw (wait for each read to finish). Amber overlays follow all three CT planes. Use crosshair and add/remove buttons to propose single-voxel edits; see volume and Dice agreement against input mask (not accuracy). Export a source-hash-bound sparse patch; no original data are overwritten. Apply it offline with the same original CT and binary NIfTI using the same command plus --patch proposal.json and a new output directory. The converter recalculates topology and retains parent mask and patch hashes. Reload corrected mask pair to review. Browser edit limit 10,000 voxels. No brush, multi-label editor, inference refinement, observer signature or clinical approval. Synthetic test fixtures only; real source benchmark pending.

Tests: node --test experimental/tests/i03-segmentation.test.mjs; python -m unittest discover -s experimental/anatomy -p 'test_i03*.py'. Confirm physical orientation and source masks independently before J phases.
