# HIL-06A | Quest 2 physical acceptance protocol
Status: NOT EXECUTED. This is a test procedure, not a signed result.
Canonical entry: https://pulmosimu.vercel.app/ — record the exact production commit/deployment before beginning. No physical instrument is needed for this gate. Keep calibrated serial input disarmed.

## Preparation
- Charge headset and controllers. Use a clear play area and normal Guardian. Stop immediately if uncomfortable.
- Record test date, headset OS, browser version, production commit and VR or MR mode. Do not use patient data.
- Start with the original reference anatomy. Run VR first, then MR separately. Exit between modes.
- Use PASS / FAIL / NOT AVAILABLE / NOT TESTED; never convert unsupported capability to PASS.
- For each failure record step, expected, observed, recovery, and screenshot or short video if practical.

## Test sequence
| ID | Action | Observable acceptance criterion |
|---|---|---|
| Q01 | Open canonical site, enter reference anatomy, launch immersive VR | Original scene visible in stereo; spatial panel accessible; no black screen |
| Q02 | Recenter (left X); hide/restore panel (left Y) | Anatomy returns within comfortable reach; panel responds |
| Q03 | Grip/trigger lung; move, rotate; grab with both hands and scale | Predictable movement, release stops manipulation, no jumps |
| Q04 | Expert Tools > Lobe Inspection; separate lungs/lobes, select and move an individual lobe, restore | Distinct structures separable and inspectable; reassembly restores original view |
| Q05 | Enter Bronchoscopy; hold right grip to steer; use left stick for insertion and roll | Live feed changes with tip direction and travel; release stops commanded motion |
| Q06 | Reach an airway junction; use triggers to choose branches and procedure guide for available choices | Orientation and branch identity agree with displayed route; no unexplained wall clipping |
| Q07 | Procedure Station > Virtual Scope; hold left grip and move left controller relative to right, then release | Virtual housing follows right hand; clutch changes insertion only while held; sticks remain usable |
| Q08 | Procedure Station > Lighting and Endoscope View; inspect near wall and bifurcation | Legible orifices, acceptable glare and lighting, correct orientation; record objectionable appearance |
| Q09 | Working Channel > Select Accessory > Extend > Actuate > Retract | Visible virtual forceps/brush extends and retracts; jaws move when applicable; suction is indicator only; no unintended original biopsy event |
| Q10 | With accessory selected, choose a branch at junction; use panel Actuate for accessory | Branch selection works and does not unexpectedly actuate tool |
| Q11 | Toggle optical response in Working Channel | On/off visibly changes source feed on supported WebGL2; raw fallback is identified if unsupported |
| Q12 | Working Channel > Imaging Lab > Radial View; cycle coupling/range and toggle off | Distinct side panel updates and disappears; original optical feed restored |
| Q13 | Imaging Lab > Respiratory Motion; enable, then disable | Reference camera/anatomy move together; disable restores original geometry |
| Q14 | In a stationary safe posture, briefly set down one controller; reacquire it | No unintended motion, recoverability, no stale command |
| Q15 | Observe a comfortable 10-minute VR session; exit; export XR session report from desktop | No persistent monitor loss, major tracking interruption or unacceptable discomfort; report saves |
| Q16 | Enter MR if supported; use real surface placement; otherwise explicitly unregistered preview | Passthrough visible where supported; fallback clearly labeled; no fabricated anchor claim |
| Q17 | Exit MR and return to original desktop view | Original camera, anatomy and controls usable; no retained accessory or motion overlay |

## Failure classification
Critical: inability to enter/exit, uncontrolled movement, unavailable live feed, branch selection blocked, unexpected source change, severe discomfort.
Major: unstable grab/scale, poor orientation, excessive glare, accessory or imaging restoration failure.
Minor: readability or cosmetic issues not blocking the workflow.
A critical or major unresolved failure blocks software freeze. Unsupported optional MR/hand tracking must be recorded separately, not silently accepted.
Hand tracking without controllers is an optional separate test; do not infer successful hand interaction from controller tests.
GPU timing and motion-to-photon latency are not proven by the exported CPU diagnostics.

## Record template
- Exact production commit:
- Deployment:
- Headset OS / browser:
- Date / test operator:
- VR results Q01–Q15:
- MR results Q16–Q17:
- Optional hand tracking:
- Critical / major / minor findings:
- Report file:
- Review decision: PENDING / RETEST / ACCEPT SOFTWARE BASELINE
- Physical instrument gate: separate HIL-06B; not completed.

## AG01–AG03 extension — physical test additions
- Q18: isolate a lobe, enable sectioning, sweep axis and restore; verify source materials/visibility restored on procedure entry.
- Q19: use virtual steps, turn and viewpoint; verify virtual plinth blocked, no physical room safety claims.
- Q20: enter window placement, select a visible panel, hold controller grip and move, release, resize and reset. In procedure confirm scope does not advance while dragging a window.
- Q21: if hand tracking available, distinguish pinch from closed-palm grasp and verify tracking loss releases it.
- Q22: in MR confirm stable surface; inspect capability status and test reposition; do not deliberately create unsafe tracking loss or fabricate anchoring.
- Q23: deploy accessory, verify scope motion and branch change inhibited, retract and resume; record clearance alerts.
- Q24: switch regional markers on/off, verify they remain separate from the optical feed; report numeric region values, not measured physiology.
- Q25: export session report and verify all AG evidence; physical HIL record inventory is separate. No physical device connected for this run.

## UI02–UI03 spatial interface extension
- Q26: in VR verify compact Controls / mode / Recenter / Exit rail is visible at entry, not covering the lung; open full console, use left Y to collapse and reopen. MR still starts with placement panel.
- Q27: inspect the console and guide obliquely; confirm curved display, individual raised controls, correct nearest-hit selection, disabled controls non-actionable. Verify readability on Quest 2, at normal distance and resized.
- Q28: enter procedure, verify monitor chassis, video and separate status strips; pointing at the monitor must not accidentally advance or actuate a procedure. Confirm native optical pixels unchanged and panels excluded from endoscopic feed.
- Q29: inspect studio architecture, floor and wall orientation; switch focus and back, then MR. Authored room must not obscure passthrough. Record comfort, glare, clipping and any blocked controls.

## UI04 extension
- Q30: switch from anatomy to procedure; confirm the contextual rail changes, restore/minimize guide and open working channel directly. Exit must remain accessible. Return to anatomy and MR placement.
- Q31: open the main console, enlarge the reading mode, minimize, restore and check that the pinned windows are deliberately repositioned only when requested. Inspect physical key travel and hover feedback. Export angular-size estimates; assess actual readability separately.
- Q32: use the endoscope shelf to focus/restore the video, show/hide guide and open station settings. Verify live feed continues and no click passes through the monitor. With accessory deployed or calibrated control armed, guide movement targets must be disabled.
