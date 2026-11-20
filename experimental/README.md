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

## R2 — anatomical canvas expansion (development branch)

- `anatomy/reference-airways.json` is a **versioned snapshot of the 18 airway branches from the existing `index.html` `SEGS` array**, without editing that source.
- `anatomy/airway-graph.js` validates IDs, coordinates, positive radii, root-connected acyclic topology, and supplies airway-to-lobe mapping and ancestor navigation.
- The experimental canvas replaces its earlier hand-drawn demonstration tubes with tapered 3D meshes constructed from these original airway centerlines. The airway graph remains in its **original coordinate orientation**: right branches at positive X and left branches at negative X.
- Lobar overlays remain schematic ellipsoids, not patient-derived surface anatomy, and branch geometry is visually scaled to the demonstration canvas. The source radii are procedural scene units, **not validated measurements**.
- Stage controls now toggle the airway graph, isolate a selected lobe and reset the camera. Picking works on airway branches and on visible lobar surfaces. No original PNEUMO·VR controls were modified.
- Run `cd experimental && npm test` to check numerical and anatomical graph invariants. **Tests have been authored but not yet executed through this GitHub connection.** Manual browser review and comparison against the original procedure are still required.

### R2 limits and next physical gate

This does not claim complete bronchial generations, alveolar tissue, 3D vascular anatomy, tumor morphology, accurate bronchoscopy geometry or physiologically validated values. The next gate is to load real reference CT-derived geometry with known units, confirm airway orientation/registration and replace schematic lobar surfaces with appropriately sourced anatomy. The original procedural application remains the canonical functionality.

## R3 — Reduced-order airway physiology (development branch)

- `core/breathing-network.js` implements a deterministic **linear resistance–compliance tree**, indexed by the same 18 branches in the versioned PulmoSIMU source reference.
- Branch resistance is a laminar Poiseuille approximation from assumed converted source lengths/radii (scene units × 0.1 m), air viscosity and a disease-dependent relative caliber. Terminal compartments include unmeasured distal resistance and regional compliance, with SI units internally.
- The simulator solves the coupled tree using a backward-implicit terminal compliance step, producing pressure [Pa / cmH2O], net volume [m³ / L], airway and regional flow [m³/s / L/s], an approximate tidal volume, and numerical conservation residual.
- The canvas includes model tidal volume, peak inspiratory flow, compartment count, a volume-cycle trace, and research pressure/frequency inputs.
- Tests cover continuity, flow conservation, approximate breathing dynamics, resistance sensitivity, regional tumor effects, and therapy-linked resistance changes. GitHub Actions executes `npm test` on branch pushes and pull requests.

### Important R3 scientific limitations

**This is a mathematically testable reduced-order model, not a validated clinical simulator.** The reference centerline scale is assumed (not calibrated CT geometry). Distal airway resistance, tissue compliance, local narrowing, driving waveform and regional weights are demonstration parameters. It does not reproduce nonlinear airway collapse, full airway generations, forced expiratory spirometry (FEV1/FVC), perfusion, gas exchange, whole-lung CFD, or patient-specific outcomes. In this model, 'driving pressure' is a mathematical analogue of a transpulmonary input rather than a measured airway or pleural pressure. Simulated bronchodilator effects are exploratory state perturbations, not efficacy predictions.

### R3 verification and merge gate

GitHub Actions should pass all R1–R3 numerical tests; then visually smoke-test the 3D laboratory and the unmodified PNEUMO·VR procedure before merging. Quantitative physiological validity requires independently curated geometry, measurement data and uncertainty analysis. This branch stays a draft review until that gate is satisfied.

## R4 — Spatial disease mechanisms

`core/disease-engine.js` provides a disease registry and spatial mechanism composer for asthma, COPD, bronchiectasis and lung cancer, with healthy reference. Focus and intensity can be changed without touching the original procedural UI. In the R4 experimental viewer, selecting a lobe also updates the focus; the sidebar offers an inflammation adjustment, mucus and tumor overlays, and the R3 breathing network receives the resulting regional resistance and compliance changes.

- **Asthma:** exploratory diffuse bronchoconstriction, mucus and inflammation perturbations.
- **COPD:** illustrative basal-weighted emphysema, remodeling, mucus and inflammation.
- **Bronchiectasis:** focal inflammation, mucus and infection burden with attenuated background burden; the simulation does not represent bronchial dilation or ciliary transport.
- **Lung cancer:** one localized tumor-occupancy placeholder, movable to one lobe; it is neither tumor growth nor an oncological treatment model.
- **Mechanistic propagation:** the existing R1 state → relative regional caliber/compliance → R3 numerical airway resistance and simulated breathing. No independent claim of clinical outcome accuracy follows from mathematical conservation.

`tests/disease-engine.test.mjs` checks localization, mechanistic isolation, downstream resistance effects and invalid inputs. Parameters and spatial weights are deliberately marked as **not calibrated**, and therapy adjustments are illustrative rather than validated pharmacological response curves. Disease-specific anatomical realism, overlap phenotypes, actual measured regional perfusion, individual biomarkers and patient CT registration remain future validation gates.

## R5 — Therapeutic Interaction Laboratory

The **new independent** `core/therapeutics-engine.js` implements a treatment experiment against an untreated counterfactual, avoiding double application of the older R1 therapy multipliers. It has explicit pathway metadata, a selectable research phenotype, elapsed simulated days and a dimensionless sensitivity factor.

Supported illustrative interventions: bronchodilator, ICS, anti-IgE, anti-IL-5, anti-IL-5R, anti-IL-4R and anti-TSLP. Endotype demonstrations: unspecified, Type 2-high, eosinophilic, allergic and Type 2-low. A response factor of zero or elapsed time of zero MUST produce no therapeutic difference. Biologic examples influence inflammation and/or mucus state only; they **do not remove tumors, reverse emphysema or produce immediate bronchodilation**.

The experimental 3D laboratory includes a phenotype selector, pathway-response factor, treatment timeline and side-by-side numerical counterfactual deltas (inflammation, resistance index, ventilation index, model tidal volume). Both control and treatment states are passed separately through the R3 airway network using the same assumptions and driving pressure.

**Scope limits:** The mechanism coefficients, activation time constants, endotype scores and response factors are invented exploratory teaching/research parameters, **not empirical drug PK/PD, clinical response likelihood or treatment eligibility rules**. They should not be used for medication selection. Time-dependent disease progression, dose, drug distribution, adverse events, treatment combinations, biological feedback and interactions are not modeled. Source-grounded model parameters and clinical calibration belong to the subsequent validation program.

Verification: `tests/therapeutics-engine.test.mjs` verifies zero-effect invariants, endotype sensitivity, absence of inappropriate tumor effect, targeted pathway specificity, model determinism and downstream physical-network coupling. `npm test` is executed by the GitHub workflow, but browser visual inspection remains a separate gate.
