# PulmoSIMU — Quest 2 WebXR pilot (Q1)

## Purpose and boundary
An **additive, opt-in, standalone** immersive anatomy inspection page in `experimental/vr.html`, entered directly from PulmoSIMU Studio. It uses the same versioned procedural reference airway graph already used by the mechanistic research laboratory (`experimental/anatomy/reference-airways.json`). The original `index.html` virtual bronchoscopy and gesture/procedure code are untouched.

This first pilot is a **controller-operated reference anatomy inspection**, NOT VR bronchoscopy, clinical decision support, realistic distal airways, or a verified 1:1 human lung.

## User workflows

- **Desktop**: orbit with the mouse, scroll to zoom, click airway/lobar elements to focus, reset view, return to Studio. The VR button reports unsupported browser status if immersive WebXR is not available.
- **Quest 2**: use Meta Quest Browser on a securely hosted HTTPS page, select **Enter VR**, look around the anatomy, point with a Touch controller and press its trigger to select reference lobe/airway regions. Leave immersive mode with the browser's Exit VR control or headset's usual system UI.
- The 3D anatomical label is placed **inside** the Three.js scene, so the selected label remains visible in immersive mode rather than relying only on 2D webpage panels.
- The model is stationary and close to the user; there is no teleportation, simulated intervention, forced motion, passthrough requirement or user data collection.

## Anatomy and units
- Uses `reference-airways.json` through the validated source graph adapter. The lobar envelopes are **schematic ellipsoids** and not based on measured pleura, lobar/fissural surfaces or registered CT.
- Procedural scene units are **not calibrated SI dimensions**. A display-only transform enlarges the model to a practical inspection size. The model is therefore **not life-size**; don't infer lengths, vessel positions or disease volumes from the VR visual.
- New objects are not registered to patient CT or imported vessel surfaces. This pilot doesn't transfer physiology, treatment, camera or procedural scope state into or out of VR.

## First-time Quest 2 setup (user-led)
1. Charge the headset and have both Touch controllers ready.
2. Install and sign into the **Meta Horizon mobile app** on a phone, then follow its pairing guidance and the headset's on-screen setup. Pair controllers as requested.
3. Connect the headset to Wi-Fi and complete any requested system updates.
4. Configure a clear boundary/play area with good lighting. Start seated or standing still; keep obstacles and pets outside the area.
5. Open Meta Quest Browser, visit the **deployed HTTPS** URL for the VR pilot, and select **Enter VR**. A public GitHub source-file URL is not a working immersive host.
6. Confirm: headset tracking, both triggers, lobe selection, in-world labels, visual readability and successful Exit VR.

For the initial test, keep sessions brief and stop if discomfort appears.

## Release and verification gate
- Node tests: scripts parse, WebXR button and controller events exist, the source is the canonical procedural airway JSON and remains nonclinical.
- Chromium smoke: tests page load, mesh/reference branch initialization, user-visible selection and desktop fallback. **Chromium CI does not provide physical Quest 2 XR hardware**.
- **Hardware pending:** confirm Meta Quest Browser immersive-vr support on the actual Quest 2, controller aiming, stereo scale, legible labels, device thermal/performance budget, guardian safety and clean VR exit.
- **Publishing pending:** this work is only on `feature/mechanistic-lung-foundation` in a draft PR, not yet deployed on the repository's main GitHub Pages website. A separate trusted HTTPS preview deployment or user-approved merge/deploy is needed before entering VR from the headset.

This is Q1, a bounded proof of working XR presentation and controller interaction, with a **physical headset acceptance gate**, not an indefinite feature roadmap.
