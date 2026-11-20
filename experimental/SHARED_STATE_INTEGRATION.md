# Shared anatomy and workspace-state contract — first integration gate

## Implemented

PulmoSIMU now has a narrow, versioned **reference-anatomy focus** shared across Studio's procedural and mechanistic workspaces:

- `experimental/shared/reference-focus.js` defines the protocol `pulmosimu.reference-focus.v1`, strict five-lobe identifiers and the canonical provenance tag `original-procedural-SEGS-snapshot-r2`.
- `procedural-workspace.html` reads the active branch *from the unchanged original PNEUMO·VR scope state*, derives its lobe only against the procedural reference airway hierarchy, and publishes **branch ID + lobar focus**.
- `studio.html` routes only accepted, same-origin messages from its **currently active** iframe, reissues monotonically increasing revisions, and transfers the latest reference focus when users switch workspaces.
- `experimental/lab.html` receives the lobe focus in reference mode, updates its selected lobe, and can publish a user-selected lobe back through Studio.
- Invalid source identifiers, unknown lobes, malformed versions, unexpected origins, or stale revisions are rejected. The page checks both `event.origin` and `event.source`.
- The original `index.html` is unchanged. No biomechanical, pharmacological, clinical, camera, gesture, biopsy, image, personal or patient data crosses the bridge.

## What is genuinely integrated

**Shared navigation state, not shared numerical physics.** On the reference anatomy, both workspaces can describe the same five lobes and transfer the current region. A procedural branch can be interpreted as its reference lobe in the mechanism view. Switching workspaces preserves the last chosen lobe. No procedural scope is teleported or overwritten when a selection arrives from the research viewer.

The branch mapping is only valid for the **original procedural SEGS source / reference-airways snapshot**. Patient CT segment IDs and imported 3D surface-atlas IDs are deliberately excluded. While the mechanistic viewer displays CT or external geometry, reference-focus messages are ignored and no state is published. This avoids false anatomical registration.

The `revision` field rejects stale workspace messages. It is a temporary in-memory Studio counter; there is no persistent user session, backend storage or cross-origin synchronization.

## Still separate (critical)

The original procedural engine and the experimental numerical R3–R5 solver do **not** share measured SI geometry, regional segmentation, tissue constraints, real-time pressure/flow state, procedure annotations, or CT registration. The original procedural mesh and the R2 experimental snapshot are derived from the same definitions but are still distinct instantiated objects. They must not be described as a single physiological engine or clinically calibrated digital twin.

## Next scientific integration gate

1. Build one serializable, checksummed anatomy graph with explicit revision, physical units and verified original-to-reference branch correspondence.
2. Introduce an anatomy-source adapter used by both engines; preserve existing procedural rendering and scope behavior behind regression tests.
3. Map patient CT airway segments to validated lobar/segmental anatomy only through controlled registration (never source ID guessing).
4. Link reduced-order mechanics to shared geometry **only after** physical-scale and distal-boundary validation.
5. Keep surgical navigation, measurements and disease-treatment experiments independent unless their data semantics are explicitly verified.

## Verification

Automated Node tests validate protocol/version/ID validation and stale-message behavior. Chromium tests exercise lobe selection through Studio, switch into procedural focus and back, and confirm that reference selection is retained. These are interaction/regression tests, not image-processing or physiologic validation.
