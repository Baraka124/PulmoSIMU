# SCI01 — Scientific fidelity protocol
Research framework, not a claim of validation. Do not unlock clinical navigation.
## Source and reference
Record study UUID, source SHA256, acquisition protocol, voxel spacing, affine, field of view, contrast and respiratory phase where known. Use approved independent expert annotations or measured reference data; record reviewer agreement and uncertainty. Separate development and held-out test cases by patient.
## Anatomy
Per structure: Dice, Jaccard, sensitivity, precision, absolute and relative volume error; report empty masks explicitly, not as perfect scores. Report symmetric mean surface distance and HD95 in physical mm, with sampling convention and resolution. Evaluate each lobe, airway, artery and vein independently; record missing structures and vessel identity errors.
## Topology
Airway centerline coverage and false branch length, branch detection by generation, connectivity, bifurcation location error and local lumen diameter bias. Record reference branch annotation limitations. Evaluate vascular continuity, artery/vein confusion and lobe overlap. Do not infer distal anatomy missing in the image.
## Geometry
Audit voxel-to-RAS and RAS-to-model transforms, units, handedness, mesh-mask correspondence, watertightness, self-intersections and triangle budget. Distinguish actual observed surfaces from FOV caps. Report reconstruction error separately from segmentation error.
## Physiology
Validate separately with independent measured flow, pressure, spirometry and/or dynamic imaging as appropriate. Specify assumptions, initial/boundary conditions, model parameters, conservation residuals, convergence and sensitivity. Do not claim patient-specific physiological predictions from CT alone.
## Procedure and XR
Evaluate route completeness, instrument geometry and collision error, physical phantom navigation, tracking and calibration error, latency, repeatability and observer variability. Separate device-specific and anatomical error.
## Reporting
Report distributions, per-case failure rates, subgroup and protocol effects, uncertainty and missingness. Freeze algorithms and metric conventions before held-out testing. Acceptance thresholds require clinical/scientific agreement for the specific intended use; none are asserted here. No overall score that conceals failures. Preserve the raw reference, versioned evaluation code and output hashes. Publication remains review_pending until appropriate independent review.
