# PulmoSIMU

**One research platform. One authoritative code branch: `main`.**

PulmoSIMU is an integrated pulmonary research and spatial procedure laboratory built on the original procedural engine. It combines an interactive reference airway model, bronchoscopy, experimental respiratory computation, source-linked CT review, desktop workstation and headset experiences. It is not yet a clinically validated medical simulation or patient-specific digital twin.

## Canonical entries

- **One canonical PulmoSIMU entrance:** https://pulmosimu.vercel.app/ (the old `/quest-short/`, `/quest.html` and `/quest4.html` shortcuts redirect here)
- **Original desktop procedural workstation (no Quest or webcam required):** https://pulmosimu.vercel.app/index.html?entry=desktop
- **CT source / MPR inspector:** https://pulmosimu.vercel.app/experimental/anatomy-inspector.html
- **Clinical and physical verification lab (local-only):** https://pulmosimu.vercel.app/experimental/validation-lab.html

Historical Vercel aliases are not release targets; use only the canonical entrance above. The single production project is **`pulmosimu`**. The earlier `pulmovr` project is obsolete and subject to permanent deletion in the Vercel dashboard.

## Authoritative implementation

- `vercel.json` routes `/` to the retained original launcher at `quest-short/index.html`, with the legacy Quest shortcuts redirected to `/`. The launcher no longer carries XR09/XR14 query labels.
- `index.html`: the single original airway, clinical reference procedure, scope state, desktop controls and WebXR adapter installation. Desktop/VR/MR choices open this same engine directly without a webcam prompt.
- `experimental/xr/procedural-immersive.js`: original-engine VR/MR session lifecycle, controller/hand tracking, spatial UI, original scope camera feed and procedure station.
- `experimental/xr/xr-capabilities.js`: XR-17A capability registry distinguishing desktop availability, VR/MR support, requested optional session features and verified grants. MR requests optional hit-test and anchors without assuming availability.
- `experimental/xr/mr-placement.js`: XR-17B original-scene spatial MR placement from actual WebXR hit-test results, explicitly confirmed surface placement, optional in-session anchor pose correction and manual unregistered preview. No physical depth occlusion, cross-session persistence or patient CT registration.
- `experimental/xr/instrument-replay.js`: XR-17C local, read-only external trace comparison and timeline. The clinical reference scope is never driven by imported measurements.
- `experimental/xr/physical-calibration.js`: fail-closed, operator-armed three-point physical-sensor reference mapping. No coupling to the original scope and no metrological or clinical certification.
- `experimental/xr/tracked-tool-proxy.js`: XR-17C optional right WebXR controller 6DoF visual marker, strictly illustrative and uncalibrated.
- `experimental/xr/display-path.js`: XR-17C compatibility evidence for WebXR compositor layers; the original same-renderer endoscopic feed remains the active display.
- `experimental/xr/bronchoscopy-station.js`: bounded XR-16F reference instrument controls, dynamic source-video CPU safeguard, configurable monitor quality and spatial layouts.
- `experimental/xr/physical-instrument-bench.js`, `clinical-review-gate.js` and `experimental/validation-lab.html`: fail-closed external evidence intake. Does **not** certify devices, clinical fidelity or physical Quest performance.
- `experimental/anatomy-inspector.html`: local CT / NIfTI / GLB/manifest source review and multiplanar section visualization, separate from procedural validity claims.
- `experimental/core/`: reduced-order computational research models, **not** automatically coupled to patient CT.
- `experimental/tests/`: numerical, source, controller, browser and procedure regression tests.
- `experimental/UNIFIED_SIMULATION_ARCHITECTURE.md`: core source and scientific boundaries.
- `experimental/XR16F_CLINICAL_PHYSICAL_INSTRUMENT_ACCEPTANCE.md`: explicit review and physical bench measurements still required before hardware/clinical claims.
- `experimental/XR17_REALITY_CONTINUUM_ARCHITECTURE.md`: XR-17A, XR-17B and XR-17C software integration and finite acceptance gates. Real Quest/device acceptance and clinical validity remain unverified.
- `experimental/XR18_CONSOLIDATION_ACCEPTANCE.md`: release inventory, native-compositor renderer limitation, physical-instrument test plan and Quest acceptance sign-off.

## Development and release policy

1. All changes are integrated in `main`; do not release from a feature branch or spin up a competing Vercel project.
2. Preserve the original procedural engine and CT source provenance. Never equate CT lung-shell meshes with validated bronchial pathways or attach uncalibrated physiology to a patient study.
3. Run the existing GitHub Actions model and browser checks before deploying the exact `main` SHA. The Vercel automatic Git integration is **not yet independently confirmed**; manual API-linked builds currently require checking their source SHA.
4. Keep patient or private CT source files local; regression tests use synthetic fixtures.
5. Preserve archived experiments as source history/test fixtures, not public flagship applications. Delete archival assets only after dependency and regression review.
6. Physical Quest interaction and performance must be verified separately from browser and source-code tests.

## Current release

**Desktop/Quest software baseline:** XR-16F. **Canonical entry + calibration preview:** XR-18 software changes integrated. **Capability-negotiated runtime:** XR-17A implemented. **Spatial MR surface placement:** XR-17B implemented in software. **Advanced procedure/input research:** XR-17C read-only trace review, tracked-controller visual proxy and compositor capability assessment implemented in software. Physical headset acceptance and clinically calibrated instrument coupling remain pending. The physical-instrument evidence reader and clinical review checklist are software, not measured proof. Hardware-validated device claims, stable physical MR anchoring, depth-based room occlusion and verified patient-specific endoscopy are **not available**.

**Cleanup status:** The duplicate development branch was deleted after proving it contained no unique commits. The obsolete `pulmovr` Vercel project remains paused and requires its owner's Delete Project confirmation; GitHub Pages depublication also requires repository settings because the automation token received HTTP 403. Historical aliases are redirected to the canonical production host by the release routing configuration. GitHub automatic deployment in the remaining Vercel project must still be verified in the Vercel Git settings.

## XR-19 renderer and simulator input integration

The original workstation now pins Three.js r147 for WebXR projection-layer support. The native quad adapter copies the original endoscopy render target to a granted XR quad layer; the same-context mesh remains the fallback. Native pixels are only reported active after a successful GPU copy. Actual Quest acceptance remains required. The original workstation also offers optional local serial setup and measured calibration import, manually armed simulator-only control and a left-grip dead-man. No physical commands, patient-specific control, or clinical certification are provided. Device sampling, axis orientation, latency and calibration must be independently measured before acceptance. Use the official root domain only.

## XR-20 physical acceptance recovery

Corrected the original clearance gate's undefined diameter variable, added explicit forward/retract controls to the spatial guide, hand pinch menu actions and remote/near anatomical grabs, closer default anatomy, MR local-space entry and explicit manual placement when real hit testing is unavailable. Native compositor activation is disabled for this compatibility test release; source endoscopy remains available. Physical retesting is mandatory.

## XR-21 inspection and optics

Reduced reference mucosal specular glare and normalized optical falloff and scope offset to model display scale. Three light profiles, original source route breadcrumb and cumulative reference path distance, all source junction choices, and reversible individual-lobe inspection using the existing five reference meshes. Reassembly is automatic before bronchoscopy and on XR exit. This is a visual reference exploration mode, not physical tissue separation or clinical anatomy validation.

## XR-22 continuous reference lumen

A bounded implicit union of the existing tapered centerline pieces provides one continuous endoscopic viewing surface, removing overlap walls at bifurcations and trifurcations. Original source graph, radii, scope kinematics and external anatomical meshes are unchanged. Shared optical rendering across desktop and XR, with automatic fallback to original tubes on construction/budget error. Terminal closures are numerical reference boundaries, not inferred distal airways. Headset rendering and orifice appearance require physical acceptance.

## XR-23 reference video bronchoscope

A generic dimensionally referenced flexible video scope adds a shaped control body, lever, suction valve, channel port, flexible 600 mm display shaft and optical tip (4.2 mm distal OD, 4.1 mm insertion OD, 2 mm channel reference, 110 degree forward view; Olympus BF-P190 published specification https://olympusmedical.asia/products/pulmonology/bronchoscopy/endoscopes/video-bronchoscopes/bf-p190/index-en.html). Housing geometry is illustrative, not manufacturer CAD. The original camera now uses a roll-coupled distal frame and aspect-correct horizontal field-of-view convention. In VR, the right controller holds the instrument; a left-grip relative-motion clutch feeds/retracts the original scope, while original sticks remain available. No physical actuation, measured shaft mechanics, haptics or clinical validation.

Manufacturer reference also specifies 2–50 mm depth of field; this release records it but does not simulate measured lens MTF, focus blur or sensor response. Published field of view is modeled as horizontal, an explicit simulation convention. https://content.medical.olympusamerica.com/products/diagnostic-bronchoscope-bf-p190

## XR24–26 device research pass

XR24 adds a depth-texture-driven focus approximation (reference 2–50 mm), bounded radial distortion and static luminance response in the XR monitor, with original image fallback on non-WebGL2 devices. XR25 displays an inserted shaft constrained to the original source route, and reports source-field geometric margin and estimated curvature. It does not solve forces or tissue deformation; the original navigation remains authoritative. XR26 adds a 2 mm reference working-channel visual mechanism for forceps, brush and suction indication. Deployment and jaw movement are bounded; no actual biopsy, fluid removal or clinical sampling is performed. Headset acceptance and bench calibration remain outstanding.

## XR-27 / HIL-01 parallel start

Local 20 Hz bounded procedure observation records reference segment bins, within-segment travel, movement-speed summaries and estimated shaft-state sample counts. No visual surface coverage, clinical score, contact force or raw trajectory export. HIL-01 preliminary passive training handle concept is stored in hardware/hil-01 with parametric OpenSCAD, dimensional design manifest and manufacturing gates. Draft CAD exported and checked locally for single connected watertight meshes; no print, slicer or hardware fit validation.

## XR-28 / HIL-02
Bounded discrete elastic centerline relaxation with fixed endpoints, assumed stretch/bend/anchor weights and source-field compliant-wall penalty. Candidate shape is observational and fallback preserves original navigation; no calibrated force, torsional dynamics, or tissue deformation solution. Shaft recomputation capped at nominal 10 Hz. HIL-02 contains separate preliminary sensor-carrier CAD, interface manifest and uncompiled raw-only ESP32-C3 acquisition sketch. Hardware remains unassembled and uncalibrated.

## XR-29 / HIL-03
R3 net-volume-driven bounded affine reference motion uses the existing 18-branch solver, an assumed baseline 2.5 L and a reversible tracheal-anchor deformation frame. The same reference airway, route, shaft visualization and tip camera frame move together; original source coordinates and procedural navigation remain unchanged. This is one-way geometry motion, not 4D CT, measured regional mechanics or updated CFD/RC resistance. The HIL-03 local angle sweep CLI fits raw ADC against independent references and reports holdout residual and hysteresis; a draft passive indexing fixture accompanies it. No BENCH/1 export, hardware actuation, or actual bench calibration.

## XR-30 / HIL-04
Optional side-by-side radial interface view uses the reference lumen's signed field in the scope transverse plane, with a 2 Hz update budget and idealized interface echo only under explicitly selected fluid coupling. No tissue acoustic map, diagnosis, clinical EBUS fidelity or patient data. HIL-04 adds a preliminary passive fixture CAD and dissipative resistance numerical analysis, with no hardware commands or actuation. Hardware design and validation remain incomplete.

## XR-31 / HIL-05–06
Three-landmark rigid registration CLI with independent check landmarks; local XR pose continuity observation without invented uncertainty. Numerical geometric contact risk is dimensionless, never a force or actuator command. HIL-06 documents the remaining independent headset and physical acceptance gates. No actual physical registration, sensor fusion, live bridge, calibrated forces or hardware acceptance.

## XR-32 | Integrated quality and freeze candidate
Preserves XR31/HIL05 and all prior modules. Branch selection takes priority over accessory actuation; use the explicit working-channel action for accessory actuation at junctions. Procedure mode requires the original reference model. The native compositor is restricted to raw-source display; depth-processed optics use the same-context mesh, and raw mesh fallback is explicit when processing is off/unavailable. Imaging panel labels and optical status are corrected. Added route-policy tests and a WebGL2 offscreen optical pixel check; independent Quest and physical HIL-06 acceptance remain necessary. No new hardware control or clinical claims.

## A–G integrated enhancement foundation (AG01)
Source-preserving lobar isolation; deliberate bounded reference-space steps and 30-degree turns (VR anatomy only); near-contact multi-finger grasp heuristic using actual joints; aggregate MR candidate stability diagnostics; sampled accessory clearance using the existing source field and deformed reference frame; R3 pressure-flow integral and conservation evidence. Hardware G adds an offline measured multi-axis calibration assessment, without BENCH/1 output or actuator commands. No new anatomy, regional biomechanics, physical MR occlusion, tissue mechanics, true sampling/suction, live sensor fusion or validated physical device. These are first increments in each block, not full A01–G03 completion. Physical headset and bench acceptance remain open.

## AG02 — depth pass
Reference selected-lobe visual clipping with cloned, restored materials (no cut-volume reconstruction); bounded virtual viewpoint and obstacle-aware step path checks for the authored plinth, menu side selection; palm-based closed-hand proxy; MR candidate stability gate and latched large anchor jump; explicit simulator-only deployment interlock and periodic source-field clearance; R3 regional volume integrals on solver time. Offline G data synchronization only, no manufactured hardware or live control. Headset and bench tests remain required.

## AG03 — integrated pre-headset checkpoint
Adds independently movable and resizable spatial panels with deliberate grip and bounded position; real-frame environment capability observations (no room reconstruction); dimensionless contact overlap evidence; regional R3-flow-derived display markers (not regional tissue deformation); HIL-09 offline evidence manifest and read-only readiness checks. Fixes AG02 stale regression assertion while retaining deployment interlock. Physical headset acceptance, higher-fidelity tissue/optics, bidirectional mechanics and physical sensor manufacture remain outstanding.

## UI01 — unified external and spatial interface
Shared graphite/porcelain presentation from launcher, original boot and workstation to desktop spatial-session dock; world-space controls and guide use measured text, quiet state highlights and the original normalized hit targets. Endoscope chrome remains outside actual source pixels. No authentication service was added: entry is the existing research launcher, not a password-protected login. Original interaction, data, source gates and AG03 simulation are retained. Browser rendering tests cover launcher responsive layout and canvas primitives; Quest optical readability still requires physical assessment.

## UI02 — physical spatial interface
The main console and procedure guide now use curved, depth-tested spatial substrates and separately extruded atlas-mapped controls. Actual mesh hits—not a flat full-canvas button approximation—select the controls. A compact command rail restores the full console, switches mode or recenters after the menu is hidden. The video monitor has a physical chassis, depth-tested source image and separate information strips. All original action IDs, source gates, calibration restrictions and render target capture remain. Physical headset appearance, occlusion, and ergonomics require acceptance; no claim of native OS interface or measured display quality.

## UI03 — spatial architecture and unobstructed start
Virtual-reality entry defaults to the compact command rail rather than the full console; MR still begins with explicit placement controls. Authored matte room surfaces, quiet architectural lighting cues and wall signage replace the grid/sphere presentation in studio mode; focus retains a minimal context. Monitor geometry blocks underlying pointer actions; compact rail has an exit and MR recenter routes to placement options rather than bypassing placement. Actual Quest ergonomics, room appearance and physical display readability remain untested. No LUNGx changes.

## UI04 — contextual spatial interaction
Mode-dependent command rail; independent monitor shelf for image focus, guide visibility and station settings; full-console minimize and explicit enlarged reading mode; raised-key press travel and source-preserving procedure interlocks. Sample nominal glyph angular size from actual world scale, distance and viewing incidence, using a 24 arcminute design target (not headset-validated acuity). The monitor remains live during layout changes; no image processing or source changes. MR placement remains explicit. Physical acceptance is pending.

## UI05 — industrial design
Semantic vector symbols accompany sufficiently large physical control faces; small controls retain text. The authored VR laboratory adds instanced acoustic fins, restrained floor joints and baked wall-wash gradients without new scene lights. Legacy decorative grid/datum/ring cues are suppressed in studio. Monitor housing has a rounded beveled rear chassis, instanced vents, mounting plate and state-driven status indicator; no source video changes. Plinth gets a machined skirt without enlarging its original collision footprint. These are display-only changes, not hardware manufacturing or real-room reconstruction. Physical visual quality remains to be evaluated on Quest.

## UI06 — spatial composition checkpoint
A bounded viewer-relative angular-envelope layout heuristic estimates overlap with the current source-anatomy bounds and other interface surfaces, including the monitor control shelf. It chooses among authored poses only on explicit layout events, protects pinned windows, preserves normal/focus station presets and provides Arrange Workspace under Window Position. No continuous head-following, organ movement, physical collision claim or clinical change. Exported evidence records estimated overlaps and layout events. Actual stereo visibility, readability, comfort and Quest performance remain physical acceptance requirements. This is the final planned UI01–UI06 pass before measured headset feedback.

## H01–H03 — patient-study foundation

Existing Source Anatomy inspector now supports local versioned study manifest export/verification, explicit mm voxel/RAS/LPS transforms and exact optional reconstruction linkage. Offline classic single-series DICOM conversion is provided separately; no patient data is included. No CT-specific procedure authorization or segmentation validation. See [H01–H03 foundation](experimental/anatomy/H01_H03_PATIENT_STUDY.md).

## I01–I03 — source-linked segmentation foundation

Offline adapter for externally segmented binary NIfTI structures, exact source/affine/hash binding, six-connected component QA and preliminary airway voxel skeleton graph; local MPR mask overlays, sparse correction and source-bound patch export. No bundled segmentation inference, validated branch graph or patient navigation. See [I01–I03](experimental/anatomy/I01_I03_SEGMENTATION.md).

## J01–J03 — source-derived route rehearsal foundation

Local airway skeleton binding, voxel-occupancy/edge verification, explicit entry/target and conservative geometric clearance-constrained shortest path; source CT crosshair route rehearsal with RAS position and forward direction. Original procedure remains independent. No patient endoscopic renderer or clinical navigation. See [J foundation](experimental/anatomy/J01_J03_AIRWAY_ROUTE.md).

## K01–K03 — respiratory deformation foundation

Offline paired CT baseline registration (SimpleITK), exact source-bound sampled displacement field and local Jacobian checks; original CT crosshair and J route point mapping with phase interpolation and geometric length comparison. No measured 4D motion, dynamic lumen surface, moving clearance or endoscopic coupling. See [K foundation](experimental/anatomy/K01_K03_RESPIRATORY_MOTION.md).

## L01–L03 — flexible shaft mechanics foundation

Source-linked fixed-endpoint discrete elastic shaft relaxation, assumed nodal wall penalty and friction diagnostic; independent expiration lumen contact recomputation using K-mapped geometry and fixed inspiration rest length. No bidirectional tissue mechanics or measured device properties. See [L foundation](experimental/anatomy/L01_L03_SHAFT_MECHANICS.md).

## M01–M03 — source optical foundation

Explicit voxel-lumen raycast endoscopic preview at J route node; optional depth-gated optical point kernels and local unregistered still comparison. No trained 3DGS, real mucosa reconstruction or XR binding. See [M foundation](experimental/anatomy/M01_M03_SOURCE_OPTICS.md).

## N01–N03 — computational research assistance

Deterministic source-linked route evidence review, bounded optical gradient-similarity search of nearby route poses, local still/video-frame capture and traceable evidence JSON. No real-time registration or clinical localization. See [N foundation](experimental/anatomy/N01_N03_RESEARCH_INTELLIGENCE.md).

## O01–O03 — integrated source research release candidate

Local read-only source evidence transfer into existing XR console, separate external bench status, H–O regression runner and release gates. No multi-user network, live patient-specific XR or clinical/device acceptance. See [O release candidate](experimental/O01_O03_RELEASE_CANDIDATE.md).

## P01–P03 — embodied reference XR

Articulated tracked glove surfaces, estimated VR operator posture and experimental deliberate two-hand reference scope control. See [P foundation](experimental/P01_P03_EMBODIMENT.md).
