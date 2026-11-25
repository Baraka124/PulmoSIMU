# HIL-01 | Physical counterpart — preliminary engineering brief
This is a passive, nonclinical handle concept. The dimensions are design assumptions except the previously recorded generic scope shaft/optical reference. No print, assembly, measurement, calibration or haptic actuation has been performed.
## Assembly architecture
- M01 two-piece printed control body, screw clearance and tube/USB ports (parametric OpenSCAD).
- M02 lever coupon and preliminary lever geometry; pivot brackets, stops and Hall sensor bracket require hardware selection.
- M03 flexible commercial tube, nominal 4.1 mm diameter and 600 mm reference working length; the printer does not make the long precision shaft.
- M04 removable low-voltage sensor electronics bay; board and wire routing pending.
- M05 desktop insertion/rotation measurement unit: independent encoders, low-friction guide, load-cell fixture; separate CAD phase.
- M06 passive resistance fixture: not yet designed or authorized for actuation.
- M07 printed airway phantom / silicone mold derived from reviewed reference geometry later.
- M08 calibration rig and measurement procedure.
## Manufacturing gate
Print only the coupon after checking dimensions in CAD and slicer. Select actual components before revising shell mounting posts, lever pivot, sensor clearances, tube retention, fasteners and assembly tolerances. Use PETG prototype material; no sterilization or patient contact. Check split fit and pinch points before handling. Use separate bench enclosure and manual quick release before future haptic experiments.
## Digital link
Existing PULMOSIMU-BENCH/1 read-only serial interface remains the starting protocol. Local acquisition and future network bridge are separate engineering gates. The browser must not command physical movement. Report measurements and uncertainty, no patient data.
Files: design-v0.1.json and handle-v0.1.scad. STEP/STL/3MF are not committed manufacturing outputs. Local OpenSCAD geometry export and trimesh inspection of the four draft parts succeeded: left and right shell each one watertight connected component (30.93 × 53.88 × 165.41 mm tessellated bounds); lever one watertight component (17 × 38.5 × 6 mm); coupon one watertight component (42 × 22 × 5 mm). This only checks numerical geometry; no slicer, material, tolerance, hardware fit, print or load validation.
