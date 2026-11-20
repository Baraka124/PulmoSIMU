# XR-06 — Anatomical workbench and sectional visualization

**Implementation:** `experimental/vr.html` + `experimental/xr/anatomy-workbench.js`  
**Source:** original PNEUMO·VR-derived `experimental/anatomy/reference-airways.json`, schema `pulmosimu-airway-reference-r2`  
**Release:** GitHub development branch only; dedicated Vercel Quest preview, not original `main`.

## Why this milestone exists

XR-05 enabled moving, summoning and grasping a complete *reference* lung assembly. XR-06 turns that spatial interaction into **anatomical inspection**: a user can choose a specific named branch from the real procedural reference graph, follow its source-defined ancestral airway path, and manipulate a virtual clipping plane to look through the displayed material without navigating the headset camera through solid geometry.

This is a visualization and navigation feature. **It does not validate patient-specific anatomical surface accuracy, calibrated airway diameters, vessel structures or tissue mechanics.**

## New workbench operations

- **EXPLORATION:** side-grip manipulation, 6DoF tracking, hand pinch when supported, hover and distant summon remain available
- **AIRWAY GUIDE:** choose a branch, and the route marker now follows the **source-defined chain** to that exact selected branch rather than a generic terminal path; thumbstick advances the marker without camera locomotion
- **NEXT BRANCH:** cycles through the real graph nodes assigned to the selected lobe, reports the graph's branch name and generation, and preserves canonical lobe focus
- **SECTION VIEW:** virtual section plane uses the current anatomical display-group transform, orientation and bounded local depth. The **left thumbstick** moves the section plane; desktop slider offers the same control. Three.js material clipping is switched on only for section mode and removed when leaving it
- **SOURCE PROVENANCE:** each mesh is tagged with its original airway or schematic-lobe source, and the inspector does not show invented physical measurements
- **RECENTER:** preserves the selected lobe; restores exploration mode and clears section clipping

### Critical invariants

1. Controller inputs are mode-specific: the left thumbstick does not simultaneously move an airway route, sweep a section and advance a bronchoscope.
2. All clipping is **reversible and visual**; no geometry mutation, cutting, volume deletion or physics changes.
3. Route selection is tied to the canonical source branch IDs and graph hierarchy; anatomical transformations never change source coordinates.
4. The richer original PNEUMO·VR `index.html` is preserved. This build does not falsely represent a complete XR port of the original instrument engine.
5. The five lobar *envelopes* remain schematic. The section plane can reveal the underlying **reference geometry**, not CT-derived tissue layers or blood vessels.

## Validation

Automated:
- `tests/anatomy-workbench.test.mjs` checks branching, region transitions, source provenance and clipping fractions
- `tests/browser-smoke.mjs` checks actual browser controls, airway selection and that Three.js materials receive/release clipping planes
- `tests/webxr-viewer.test.mjs` verifies spatial mode wiring and source-specific scene integration
- Original procedural browser smoke and all existing numerical research tests remain mandatory

Physical Quest 2 acceptance after battery recharge:
1. Confirm `XR-06` in the viewer.
2. Enter XR; verify recenter, distant summon and reachable side-grip manipulation.
3. Choose AIRWAY GUIDE; change the branch with NEXT BRANCH and move the route marker.
4. Choose SECTION VIEW; sweep the left thumbstick to move the section plane; confirm visibly different cut-through views and no involuntary camera movement.
5. Return to EXPLORE; verify lobe geometry is restored.
6. Record the session (Meta system Camera → Record), observing frame stutter, controller accuracy and visual legibility.

**Exit criterion:** Working, intelligible section and navigation on Quest 2 without crashes. This is **not** the final PulmoSIMU medical-digital-twin freeze. The next science gate is to integrate verified anatomical assets into the same workbench using proven registration, dimensions and provenance before reporting actual biological metrics.
