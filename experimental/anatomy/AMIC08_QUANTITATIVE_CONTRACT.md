# AMIC07E / AMIC08 — Quantitative paired respiratory imaging

## Scientific scope
Research prototype, noncontrast CT, matched respiratory phases, independently reviewed parenchymal masks. Source HU is not filtered or clipped for quantification. Registration uses a clipped copy only. Distinct phase identity, acquisition protocol, lung inflation and calibration require review. No patient data in source control.

## Coordinate and volume convention
Fixed inspiration x, moving expiration T(x)=x+u(x), RAS millimetres. F=I+grad(u), J=det(F)=dV_exp/dV_insp. A positive J below 1 indicates contraction. J<=0 excluded, not repaired or hidden. Reference-mask fractions are not whole-lung estimates when using ROIs.

Two-compartment approximation: tissue fraction t=(HU+1000)/1000, air fraction 1-t, valid only in [-1000,0] HU. No clipping or silent extrapolation. Water-equivalent tissue includes blood and does not measure actual tissue mass. Tissue-volume residual r=J*t_exp(T(x))-t_insp(x). Under local tissue-volume conservation r=0. The integral of r times reference voxel volume is the mapped-minus-reference tissue-equivalent volume. Air-volume change per reference volume is (1-t_insp)-J*(1-t_exp). These are model estimates, not measured ventilation, flow, or gas exchange.

Current release evaluates a mass-conservation objective but does not optimize the registration under that objective. Future optimizer must include spatial regularization, positive-J safeguards, correspondence/landmark evidence, mask support and independent evaluation. Do not infer physiological validity from a lower residual alone.

## PRM research convention
Fixed inspiration grid. Valid common lung support with positive Jacobian. Normal pattern: -950 <= I < -810 and E >= -856; non-emphysematous air trapping: same I range and E < -856; emphysema pattern: I < -950 and E < -856; higher inspiratory attenuation: I >= -810. Other pairs or either HU outside [-1000,0] are indeterminate. 0=unassessed. This explicit variant is not a certified reproduction of a commercial PRM product. Report coverage and all class denominators. Airway lumen exclusion and mask quality are operator-attested.

## Validation
- Analytic rigid transform, affine stretch, and mass-conserving compression; verify direction and units.
- Repeatability, landmark target registration error, inverse consistency, positive-J fraction, quantitative sensitivity to acquisition and reconstruction.
- Compare masks against independently reviewed engineering references, and PRM against a documented reference implementation on matched protocols.
- Compare regional functional estimates to independent measurements where available. Do not interpret CT surrogates as direct spirometry or regional ventilation measurements.
- Record human review time and processing time separately; no superiority claim until measured.

## References
- Galban et al., CT-based biomarker, https://pmc.ncbi.nlm.nih.gov/articles/PMC3493851/
- PRM variability, https://pmc.ncbi.nlm.nih.gov/articles/PMC4643661/
- PRM in transplantation, https://pmc.ncbi.nlm.nih.gov/articles/PMC5387704/
- Registration-based regional lung function, https://journals.physiology.org/doi/full/10.1152/japplphysiol.00113.2013
- Multiscale MDCT breathing model, https://pmc.ncbi.nlm.nih.gov/articles/PMC3685439/
