# PulmoSIMU · Quest 2 headset pilot — XR-03

**Branch:** `feature/mechanistic-lung-foundation`  
**Entry:** `/quest.html` (redirects to the integrated `/experimental/vr.html`)  
**Environment:** dedicated Vercel `pulmosimu-xr-preview` deployment, **not** the legacy GitHub Pages `main` application.

## Before putting on the headset

- Charge and pair the two Quest 2 Touch controllers, connect Quest Browser to Wi-Fi and set a seated play boundary. Be aware of your physical surroundings.
- Open the release URL in Meta Quest Browser. Once the reference-lung model appears on the 2D browser page, select **Enter VR**. WebXR requires an explicit click/tap; do not auto-enter without user interaction.
- You should see the lungs **in front of you**, not attached to your head, and a separate 3D guide panel to the right. Do not physically walk around. If placement is uncomfortable, recenter.
- If this is a first browser session, the guide opens with **five** short steps. Point at **NEXT STEP** with either controller, press the index trigger, or choose **EXPLORE NOW**. You can reopen the instructions with **FIRST STEPS**.

## Controls

| Control | In-headset effect | Safety |
| --- | --- | --- |
| Head movement | Look around naturally | No automatic headset motion |
| Either controller, **index trigger** | Select a region or a spatial panel action | May vibrate briefly if headset browser supports haptics |
| Either controller, **side grip** (middle finger) | Hold, position and rotate the anatomical reference | Release to stop manipulation |
| **Both side grips** | Rotate, reposition and pinch-scale the model | Bounded display size, soft comfort distance clamp |
| **A** (right controller) | Hide/show the spatial inspector | Restore it even if hidden |
| **B** (right controller) | Change anatomy exploration ↔ airway guide | Does not change bronchoscope camera |
| **X** (left controller) | Recenter model and spatial panel | Preserves selected lobe, resets exploration only |
| Airway Guide, **left thumbstick up/down** | Move highlighted marker along selected lobe's reference route | The camera does not travel along the airway |
| Spatial panel **ISOLATE LOBE** | Dim other lobe envelopes | All original airway data stays in memory |
| Spatial panel **PROCEDURES** | End immersive session and open Studio procedures | No unexpected procedure action |
| Exit VR via Quest system UI | Leave immersive renderer | Desktop browser workspace remains |

**Controller handedness:** Gamepad buttons map to the tracked *actual* left/right controller, not the order in which they connect. If one controller disconnects, its grip state is released.

## Integrated development

The 3D reference selection can flow through **Studio → Procedures / Mechanisms / XR** via `pulmosimu.reference-focus.v1`. Only the five canonical lobe IDs and reference branch mapping are allowed. When using standalone Quest Browser, navigation back to Studio carries the selected lobe through a validated URL `?mode=...&focus=...` as a **navigation hint only**. No CT data, synthetic/patient volume, imaging, therapy, respiratory mechanics, or scope control is imported into this XR scene.

Source fidelity: `experimental/anatomy/reference-airways.json` is the versioned original 18-branch procedural reference. Lobar outlines are **schematic reference envelopes**, not true CT-segmented pleural surfaces or anatomical fissures. Scale is **visual display only** and not calibrated to physical lung dimensions. The airway route marker is illustrative path navigation, **not** real bronchoscopy or a physiological solver. No clinical validation claimed.

## Release checklist

Automated software gates:

- `cd experimental && npm test` (network/disease/therapy/atlas/geometry/Focus/XR controls tests)
- Chromium browser smoke checks 2D reference viewer, standalone Quest redirect, focus preservation on recenter, Studio handoffs and untouched legacy procedure
- Vercel deployment returns the expected XR viewer over secure HTTPS on current commit

**Physical headset acceptance remains mandatory** before saying the full interaction is hardware-verified:

1. First-use five-step panel readable in both eyes, controllable by index trigger
2. Either side grip picks up the model; release stops movement
3. Both grips rotate/scale without jump, tracking loss, unintended camera drift or clipping
4. Point/trigger highlights and selects lobe; A restores the panel; B selects airway guide
5. Left thumbstick moves highlighted airway route marker only
6. X recenters while retaining selected lobe
7. Enter Procedures returns to Studio gracefully after exiting VR
8. No nausea, unexpected forced travel, excessive controller lag, blurry labels or rapid Quest 2 thermal degradation over a short seated session

**Do not declare clinical, anatomical or physical precision from an automated pass.** A user wearing the headset must confirm the physical checklist.
