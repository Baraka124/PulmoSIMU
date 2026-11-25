# HIL-02 — passive sensor integration, draft v0.2
This is a component-interface study, not a completed handle assembly. CAD files have not been rendered, sliced, printed or fit-checked. Start with the coupon. Use measured board, magnet, sensor, fastener and shaft dimensions before revising the CAD.
- sensor-carriers-v0.2.scad: provisional electronics tray, lever sensor pivot bracket, shaft guide insert, dimensional coupon. These are separate parts; assembly alignment to HIL-01 is not verified.
- interfaces-v0.2.json: geometry and sensor requirements. 22 x 34 mm is a design envelope, not a claim about any particular board.
- firmware/raw-acquisition.ino: provisional ESP32-C3 Arduino raw ADC and switch telemetry at nominal 50 Hz. Not compiled or flashed. Check pinout and 3.3 V compatibility with actual electronics. Use an analog-output Hall sensor for this sketch; a digital angle sensor requires its own driver.
- External insertion and rotation encoders, load cell and mounting geometry remain to be designed. No motor, brake, powered movement or real medical instrument.
## Calibration boundary
RAW/0.1 is intentionally rejected by the existing BENCH/1 calibrated parser. Do not fabricate missing insertion, roll, pitch or yaw measurements. Record reference angle using an independent jig, test repeatability and hysteresis, then define calibrated mapping. Future HIL-03 bridge may publish BENCH/1 only after required channels exist and are verified. Browser remains read-only. No patient contact.
