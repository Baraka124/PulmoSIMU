# PulmoSIMU — Mechanistic Lung Laboratory (R1)

This is an **additive experimental extension** of [PulmoSIMU](https://github.com/Baraka124/PulmoSIMU).

## Preservation boundary

The original `index.html` (PNEUMO·VR surgical planning and bronchoscopy) and the established Python CT / synthetic-data pipelines **have not been modified**. The new research experiment lives at `experimental/lab.html`, and uses its own importable core in `experimental/core/mechanistic-lung.js`.

## Explore

Run a static local HTTP server from the repository root (for example `python -m http.server 8080`) and visit `http://localhost:8080/experimental/lab.html`. The visual workspace uses Three.js through a public CDN and requires network access. To test the numerical core, from `experimental/` run `npm test` (Node.js 20+).

## What R1 actually models

- Five lobar compartments (RUL, RML, RLL, LUL, LLL) with illustrative weights, *not* patient-derived volumes.
- Deterministic dimensionless state variables: bronchoconstriction, inflammation, mucus, remodeling, emphysema, infection and local tumor burden.
- Relative airway caliber; a fourth-power resistance **proxy**; regional compliance **proxy**; normalized ventilation weights; an aggregate heterogeneity index.
- Mechanism-level therapy attenuation with **illustrative** exponential activation over time. Not pharmacokinetics, pharmacodynamics, biomarker prediction or treatment efficacy.
- Explicit regional targeting of a tumor placeholder, avoiding false global tumor effects.
- A selectable, rotatable bilateral schematic with disease and mechanism overlays.

## Critical limitations

The viewer uses schematic ellipsoid lobe geometry and simple visual branches, **not anatomically accurate full lungs, CT-based segmentation, airflow CFD, gas exchange, or measured treatment effects**. Never interpret output as a diagnostic or therapeutic recommendation. Parameters are demonstrative and need literature provenance, unit-based physics, sensitivity analysis, calibration and experimental validation.

## Next integration gate

1. Preserve PNEUMO·VR's entire original procedural behavior and CT loader with regression tests.
2. Define a versioned anatomy graph adapter to reuse its airway tree and CT-derived centerline metadata.
3. Replace this R1 schematic with medically grounded lobar/segmental geometry while maintaining controls.
4. Implement independently verified physical respiratory solvers, then validated disease mechanism modules.
5. Merge the experimental page into a shared laboratory shell only after the preservation gate passes.

There is no automatic navigation link from the existing surgical UI: intentional to avoid unintended regressions.
