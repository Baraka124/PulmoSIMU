# UI11 | Software freeze and acceptance gate

Controller commands now require current finite grip/target-ray poses and an observed neutral frame after connection, tracking loss or mode change. A held stick or grip does not automatically reacquire on recovery. No polling of hand gamepads; hand pinch and closed-hand grasp require observed opening after tracking loss. Trigger events and polling share edge state; armed calibration uses only the frame-polled capture path. Existing controller movement mapping is preserved (anatomy stick transforms organ, workspace menu steps/turns user; procedure stick controls scope). Current runtime local-floor is respected if explicitly granted; otherwise existing 1.18 m estimate is retained and labelled. No physical floor certification. EBUS control animation updated; movement during armed calibration blocked. Existing source and safety gates remain. No new clinical/hardware capability.

Finite software acceptance: node experimental/tests/run-release-checks.mjs. Contains H–U and UI07–UI11 isolated tests. Syntax and focused gate checks are not browser integration tests. Required acceptance before calling this a working headset release:
1. Desktop: launch all canonical destinations, import actual local source CT/GLB/mask/field, verify original overlays and route/shaft/optics exports.
2. Quest VR: enter and exit twice, verify room/floor, menu ray, neutral recovery, step/turn, bimanual organ manipulation, hand tracking, patient station, EBUS monitors and controls, no stale motion after input loss. Record and replay a reference procedure. Check readability, frame timing and overheating.
3. Quest MR: hit-test placement, optional features, no unsupported depth or anchor claim, session-local registration and reset. Real floor and tracking capability may vary.
4. Hardware: only non-actuating simulator input, calibration residuals, dead-man, disconnect and stale sample tests; no force actuation.
5. Deploy exact reviewed main SHA on the existing Vercel project, verify release provenance and smoke test before promoting. No separate project/branch or unrelated features.

Status: software candidate, not physical acceptance. Clinical navigation, accurate full body tracking, real ultrasound, remote multi-headset collaboration and active force feedback remain outside this accepted scope.

## Release hardening after X03

J planning now uses a per-search cache and a cumulative maximum of 5,000,000 voxel-neighborhood iterations, alongside the existing per-node and visited-node limits. Budget exhaustion is not a proof of route infeasibility. Source inspector literal control bindings are checked for missing or duplicate IDs in the release suite. Still synchronous; workstation profiling and an asynchronous worker remain future optimization if needed.

## Acceptance evidence — 2026-11-27

At source commit `039b8049b4cf7706511f2703e4ef16835345bb95`, 67 checked-in focused test bodies passed in an adapted JavaScript harness using exact fetched sources (H–X, UI07–UI11, excluding the full local dependency audit). All four nonempty classic inline entry scripts compiled; all literal inspector control IDs resolved uniquely. Two additional integrated numerical fixtures exercise H/I/J/L/M/N/O handoff, source consistency, and edited-mask rejection. No clinical data is used in test fixtures. Native Node run, all entrypoint dependency checks, browser rendering, authorized CT dataset, actual Quest and hardware tests remain pending. The native release workflow uses the existing release runner, with read-only repository permissions and no deployment credentials. Do not infer acceptance from this record alone.
