# PulmoSIMU Spatial Systems · XR-05 technical baseline

**Status:** Spatial-engine foundation implemented on `feature/mechanistic-lung-foundation`. This is not a physically or clinically validated immersive digital twin. **Do not merge into main or replace the original procedural experience until the headset and source-parity gates pass.**

## Design principle

The experience is centered on **anatomically identifiable objects in 3D space**, not a 2D pointer UI. Each interaction has a deterministic state and safe source/coordinate semantics.

### Current implementation

| Layer | Source of truth | Behavior available | Limits |
| --- | --- | --- | --- |
| Reference anatomy | `anatomy/reference-airways.json` and `anatomy/airway-graph.js` | Select original-derived branch/lobe IDs; travel through Studio canonical focus | 18 proximal procedural-reference branches, uncalibrated dimensions |
| Display surfaces | `vr.html` schematic lobes | Identify and isolate selected lobe | Deliberately not patient-specific lobar surfaces |
| Source identity | `anatomy/object-contract.js` | Stable source ID per mesh, forbids unsupported length/volume/diameter claims | Requires independently measured geometry for future metrics |
| Spatial intent | `xr/spatial-intent.js` | Near/far arbitration, pinch hysteresis, workzone target, smooth finite summon | 3D visual positioning is not tissue mechanics |
| XR controller and hand input | `xr/controller-rig.js` | Controller ray for distant selection and summon; controller side grips for *near* grabs; optional tracked-hand index/thumb pinch for near grabs; two-hand scale/rotation | Hardware input and controller/hand handoff must be physically tested |
| UX and accessibility | world-space inspector and five-step tutorial | Contextual panel, recenter, separation of exploratory and airway guide input, return to Studio | No fully simulated medical instrument |
| Performance diagnostic | `frameMetrics` rolling 90 frames | Estimated FPS, 95th percentile frame time, slower frame count in `pulmoXRDebug()` | Not a device-level GPU timing or clinical performance metric |

### Interaction rules

1. **Far-field:** user points at a distant anatomical mesh, then presses index trigger. The selected canonical region is updated and the *entire reference assembly* smoothly approaches a near working zone; no headset/camera teleport. Spatial menus capture input before anatomy does.
2. **Near-field:** grip begins only when the gripping hand is within a small threshold of the actual *displayed mesh bounds*. The object then follows an attached hand transform until release. Squeeze at a distance cannot mysteriously drag the distant lung.
3. **Two-handed:** if both tracked inputs have eligible proximity contacts, hand separation scales the shared assembly and connecting-axis orientation rotates it. Release falls back to a fresh one-hand anchor to avoid jumps.
4. **Tracked bare hands:** request optional WebXR `hand-tracking`. Use `XRFrame.getJointPose` for thumb/index tips and wrist. Only *near-field pinch* directly manipulates anatomy; thresholds have hysteresis to avoid flicker. Controller and hand input share per-hand ownership and can never drive the same hand at once. Input disappearance cancels a grasp.
5. **Anatomical selection:** nearest indexed region can be selected by contact, and raycasting also maps to the same original reference IDs. **No individual lobes are detached or deformed** while geometry is schematic.
6. **Recenter:** sends the reference assembly back to its distant display position, preserving the selected canonical lobe; session exit restores the desktop view. No forced locomotion.
7. **Physiology:** spatial distances use *display metres* only. They cannot be used as airway diameters, airway resistance, compliance, or patient-specific mechanics.

### Known engineering limitations

- The current VR visual is still simplified; an advanced anatomical shell requires a validated geometry-source adapter to the original procedural scene or CT-registered surface. XR-05 is specifically the **interaction architecture milestone**, not high-fidelity anatomical completion.
- World mesh bounds approximate contact on transparent ellipsoids and reference airway tubes. Need actual collision primitives/surface BVH for precise geometry-backed contact and near-grab affordances, particularly with imported CT surfaces.
- Browser test environment cannot reproduce Quest 2 optical comfort, controller jitter, hand occlusion, GPU load or headset thermal behavior.
- Haptics are optional controller vibration, **not** measured tissue resistance.
- Mechanics and anatomy source coordinates must only be coupled after actual registration, scaling and numerical validation. Original `index.html` remains preserved.

## Finite gates to the integrated release

**Gate A · Input and ownership (XR-05 code / physical test pending).** Controller near-grab, far selection and summon, pinch near-grab, two-hand transfer and safe panel priority; rolling performance measurements visible in debug; no mode/input conflicts. Tests: `tests/spatial-intent.test.mjs`, `tests/spatial-interaction.test.mjs`, `tests/object-contract.test.mjs`, and full browser smoke.

**Gate B · Primary anatomy and physics fidelity.** Stop presenting schematic ellipsoids as main anatomy: use original engine's model-data source or calibrated and registered CT/GLTF source. Add explicit measurement provenance, mesh-derived colliders and subject-specific registration validation. Preserve original bronchial IDs and instrument/tool state.

**Gate C · Integrated procedures.** Use *original* bronchoscopy camera/instrument/pathology logic in headset; constrain scope movement to airway geometry and clearly separate exploratory manipulation from operating an instrument. Validate contact and response physics against independently documented targets rather than aesthetic animation.

**Gate D · Quest 2 acceptance and freeze.** Seated comfort, readability, no involuntary user motion, 6DoF registration, recoverable controller/hand transitions, near/far affordances, instrument controls, target performance and regression test across 10-minute continuous sessions. Freeze when signed off by an actual headset tester, not just GitHub CI.

## How to test XR-05

In the dedicated headset preview URL, use **Enter VR**. Look for `PULMOSIMU XR-05`. From the distant display zone, point at a lobe and press **index trigger** once to *summon* the model; do not squeeze the side grip at a distance. Once within reach, try **side grip** to hold, drag and release; then two grips for scaling. If hand tracking activates, use an index/thumb pinch *near* the object, not in empty space. Select **Recenter** to restore the distant position. Pinch/menu affordances are preliminary and require Quest 2 physical validation.

Read user-facing `XR_PARITY_AUDIT.md`, `INTEGRATED_DEVELOPMENT.md` and `QUEST2_READY.md` for the broader integration and evidence gates.
