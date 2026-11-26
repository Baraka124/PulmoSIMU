# K01–K03 | Paired respiratory CT registration foundation

The original CT in Source Anatomy is operator-designated inspiration. Load an independently confirmed same-subject expiration CT (the software does not establish subject identity, acquisition phase or clinical suitability). Keep data local; no patient images in Git.

K01: local baseline single-resolution SimpleITK Diffeomorphic Demons registration on physical CT coordinates. Input uncompressed NIfTI with explicit mm sform, orthogonal grid and finite CT values. Initial physical identity alignment only; large initial offsets, different lung coverage, pathology, respiratory motion near pleura and severe intensity variation need specialist registration strategies. No automatic rigid alignment, lung masks or landmark accuracy validation. SimpleITK resamples expiration to inspiration grid, computes a fixed-to-moving LPS displacement, exports sampled RAS mm vector field. Dense Jacobian diagnostics are reported, not proof of diffeomorphism, and not equivalent to browser sampled-grid Jacobians.

Run with Python 3.11+, numpy, nibabel, SimpleITK:
```sh
python experimental/anatomy/respiratory_registration.py inspiration.nii expiration.nii local-motion --stride 4 --iterations 50
```
K02: browser verifies both CT hashes, field SHA, dimensions and source affines, explicit vector convention; trilinear interpolation and analytic local Jacobian of interpolant. Field covers sampled grid only, no extrapolation; stride can omit edge voxels. Intermediate phase fraction is a linear displacement interpolation, not measured 4D breathing or time. A nonpositive sampled local Jacobian blocks mapping at that point, not a global certification.

K03: existing MPR crosshair reports mapped position, displacement and local Jacobian; nearest expiration HU is shown only at full endpoint and within expiration grid. If J route exists, map its points, compute polyline length and minimum node Jacobian. Neither between-node topology nor clearance is revalidated. No dynamic GLB, patient endoscopic renderer, scope or physical interaction update. No moving-phase MPR image displayed yet.

Wait for each local file load to finish. Load inspiration CT, expiration CT, field.json, field.raw; use phase slider. Changing inspiration invalidates the pair. Clear pair releases field and expiration volume. Original procedure and reference XR unchanged.

Node tests: node --test experimental/tests/k03-respiratory-field.test.mjs. Real paired scans, independent landmarks, TRE, overlap, folding, inverse consistency, physiological plausibility and respiratory phase review required before further dynamic procedural coupling.
