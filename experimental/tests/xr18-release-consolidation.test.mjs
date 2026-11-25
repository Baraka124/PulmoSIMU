import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const src=path=>readFileSync(new URL(path,import.meta.url),'utf8');
const sandbox={window:{}};
for(const file of ['../xr/instrument-bench.js','../xr/physical-calibration.js'])runInNewContext(src(file),sandbox);
const parser=sandbox.window.PulmoXRInstrumentTelemetry.parseFrame;
const make=()=>sandbox.window.PulmoXRPhysicalCalibration.create({parseFrame:parser});
const fixture=()=>({
 protocol:'PULMOSIMU-CALIBRATION/1',calibrationId:'TEST_ONLY_DO_NOT_USE',
 axes:{
  insertionMm:{unit:'mm',points:[{reading:0,reference:2},{reading:1000,reference:1002},{reading:500,reference:502}]},
  shaftRollDeg:{unit:'deg',points:[{reading:-150,reference:-148},{reading:150,reference:152},{reading:0,reference:2}]},
  tipPitchDeg:{unit:'deg',points:[{reading:-60,reference:-58},{reading:60,reference:62},{reading:0,reference:2}]},
  tipYawDeg:{unit:'deg',points:[{reading:-60,reference:-60},{reading:60,reference:60},{reading:0,reference:0}]}
 }
});
const sample=(seq=1,over={})=>({
 protocol:'PULMOSIMU-BENCH/1',seq,insertionMm:200,shaftRollDeg:10,
 tipPitchDeg:5,tipYawDeg:-5,...over
});
test('XR18 mapping requires independent three-point reference data',()=>{
 const c=make();
 assert.throws(()=>c.arm(),/CALIBRATION_REQUIRED/);
 assert.equal(c.snapshot(0).state,'NO_CALIBRATION');
 const bad=fixture();delete bad.axes.tipYawDeg;
 assert.throws(()=>c.configure(bad),/THREE_REFERENCES_REQUIRED_tipYawDeg/);
 const unsupported=fixture();unsupported.axes.insertionMm.points[2].reference=800;
 assert.throws(()=>c.configure(unsupported),/RESIDUAL_EXCEEDS_PROVISIONAL_TARGET_insertionMm/);
 const cfg=fixture();cfg.axes.tipPitchDeg.points[1].reading=-50;
 assert.throws(()=>c.configure(cfg),/NONINVERTIBLE/);
 const good=c.configure(fixture());
 assert.equal(good.calibrationId,'TEST_ONLY_DO_NOT_USE');
 assert.equal(good.clinicalApproval,false);
 assert.equal(good.deviceToScopeCoupled,false);
 assert.equal(good.measurementAccuracyCertified,false);
 assert.equal(good.state,'AWAITING_INPUT');
});
test('XR18 read-only preview is fail-closed and requires manual rearm',()=>{
 const c=make();c.configure(fixture());
 assert.equal(c.ingest(sample(1),1000).reason,'MANUAL_ARM_REQUIRED');
 c.arm();
 const x=c.ingest(sample(1),1000);
 assert.equal(x.accepted,true);
 assert.equal(x.corrected.insertionMm,202);
 assert.equal(x.corrected.shaftRollDeg,12);
 assert.equal(x.corrected.tipPitchDeg,7);
 assert.equal(x.drivesVirtualScope,false);
 assert.equal(c.snapshot(1100).state,'BENCH_MAPPING_ONLY');
 assert.equal(c.ingest(sample(1),1150).reason,'REPLAYED_SAMPLE');
 assert.equal(c.snapshot(1160).state,'SAFETY_STOP');
 assert.equal(c.ingest(sample(2),1170).reason,'MANUAL_ARM_REQUIRED');
 c.arm();
 assert.equal(c.ingest(sample(2),1180).accepted,true);
 assert.equal(c.snapshot(1480).state,'SAFETY_STOP');
 assert.equal(c.snapshot(1480).lastError,'STALE_SAMPLE');
 assert.equal(c.snapshot(1480).corrected,null);
 c.arm();
 assert.equal(c.ingest(sample(3,{insertionMm:1500}),1500).reason,'OUTSIDE_CALIBRATED_RANGE_insertionMm');
 assert.equal(c.snapshot(1510).state,'SAFETY_STOP');
 c.clear();assert.equal(c.snapshot(2000).state,'NO_CALIBRATION');
});
test('XR18 invalid frame and mismatched units never actuate the simulator',()=>{
 const c=make();c.configure(fixture());c.arm();
 assert.equal(c.ingest('malformed packet',100).accepted,false);
 assert.equal(c.snapshot(150).state,'SAFETY_STOP');
 const wrong=fixture();wrong.axes.tipYawDeg.unit='mm';
 assert.throws(()=>c.configure(wrong),/AXIS_THREE_REFERENCES_REQUIRED_tipYawDeg/);
 const source=src('../xr/physical-calibration.js');
 for(const forbidden of ['scope.velocity','scope.deflH','scope.deflV','scope.roll','fireAction(', 'fetch(', 'WebSocket(', 'getWriter(', 'requestPort('])
  assert.equal(source.includes(forbidden),false,'Unexpected actuator/output: '+forbidden);
 const clinical=readFileSync(new URL('../validation-lab.html',import.meta.url),'utf8');
 assert.ok(clinical.includes('physical-calibration.js?v=XR18-FAIL-CLOSED-CALIBRATION'));
 for(const id of ['configureMapping','armMapping','stopMapping','mapPacket'])
  assert.ok(clinical.includes('id="'+id+'"'));
});
test('Consolidated Vercel root serves launcher and redirects all legacy shortcuts',()=>{
 const config=JSON.parse(readFileSync(new URL('../../vercel.json',import.meta.url),'utf8'));
 assert.ok(config.routes.some(r=>r.src==='^/quest-short/?$'&&r.status===308&&r.headers.Location==='/'));
 assert.ok(config.routes.some(r=>r.src==='^/(quest|quest4)\\.html$'&&r.status===308));
 assert.ok(config.routes.some(r=>r.src==='^/$'&&r.dest==='/quest-short/index.html'));
 assert.equal(config.routes.at(-1).handle,'filesystem');
 const launcher=readFileSync(new URL('../../quest-short/index.html',import.meta.url),'utf8');
 for(const value of ['vr','mr','desktop'])assert.ok(launcher.includes('../index.html?entry='+value));
 assert.ok(!launcher.includes('XR09-UNIFIED'));
 assert.ok(!launcher.includes('interaction=XR14'));
 const engine=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
 assert.ok(engine.includes("['desktop','vr','mr'].includes(entryMode)"));
 assert.ok(engine.includes("openDesktopWorkstation({requestCamera:false})"));
});
