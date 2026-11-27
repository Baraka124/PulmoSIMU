# INF04 — First automated study worker
This is a trusted local/isolated worker entry point, NOT an enabled public upload endpoint. Source is classic DICOM directory or uncompressed .nii. Reuses H, I and V modules. One UUID study, one airway structure, atomic output directory, no overwrite. Does not anonymize data or certify anatomy. The manifest is an offline package descriptor, not a clinical authorization or a database transaction.

Core:
  python -m pip install -r requirements-reconstruction.txt
  python experimental/anatomy/study_worker.py source.nii NEW_OUT --study-id UUID --mask airway.nii.gz --engine "actual provider" --engine-version "actual version"

Optional inference in a separate environment after reviewing model terms, downloading approved weights and provisioning resources:
  python -m pip install -r requirements-inference.txt
  python experimental/anatomy/study_worker.py source.nii NEW_OUT --study-id UUID --device gpu

Pinned TotalSegmentator 2.13.0 lung_vessels task uses lung_airways.nii.gz only. The adapter does not include artery, vein or wall in airway lumen and rejects missing/misaligned/nonbinary output. No threshold fallback or implicit resampling. Verify exact class semantics and model weight provenance before actual studies. Actual inference and dependency resolution not verified here. Source mask, topology, source-space GLB, reconstruction and fidelity manifests plus checksummed package metadata. All outputs review_pending and clinical_navigation_allowed=false. Existing voxel/mesh caps remain. Files do not enter Supabase automatically; authorized job claiming, storage streaming, session binding, recovery and publication are subsequent integration work. No actual source files committed. CI test uses numerical geometry only.
