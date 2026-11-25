import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const load=path=>readFileSync(new URL(path,import.meta.url),'utf8');
const ctx={window:{}};runInNewContext(load('../xr/instrument-bench.js'),ctx);
const API=ctx.window.PulmoXRInstrumentTelemetry;
const packet=(seq=0,o={})=>JSON.stringify({protocol:'PULMOSIMU-BENCH/1',seq,
 insertionMm:43,shaftRollDeg:18,tipPitchDeg:20,tipYawDeg:-5,...o});
test('XR16-FINAL rejects malformed, nonfinite, out-of-range and stale telemetry',()=>{
 for(const value of ['{}','not json',packet(1,{tipYawDeg:1e8}),packet(1,{insertionMm:-4}),
 packet(1,{tipPitchDeg:null}),JSON.stringify({protocol:'WRONG',seq:1,insertionMm:4,
 shaftRollDeg:0,tipPitchDeg:0,tipYawDeg:0})])assert.throws(()=>API.parseFrame(value));
 const b=API.create();
 assert.equal(b.ingest(packet(1),1000).accepted,true);
 assert.equal(b.ingest(packet(1),1010).reason,'STALE_OR_REPLAYED_SEQUENCE');
 assert.equal(b.ingest(packet(0),1020).accepted,false);
 assert.equal(b.ingest(packet(2,{tipYawDeg:999}),1030).accepted,false);
 assert.equal(b.ingest(packet(2),1040).accepted,true);
 assert.equal(b.snapshot(1100).status,'UNCALIBRATED');
 assert.equal(b.snapshot(1400).status,'STALE_INPUT');
 assert.equal(b.snapshot(1100).deviceToScopeCoupled,false);
 assert.equal(b.snapshot(1100).clinicalUse,false);
});
test('XR16-FINAL physical offsets require fresh data and explicit reference',()=>{
 const b=API.create();
 assert.equal(b.calibrate({},100).ok,false);
 b.ingest(packet(4),1000);
 assert.equal(b.calibrate({insertionMm:0},1020).ok,false);
 const z={insertionMm:40,shaftRollDeg:18,tipPitchDeg:10,tipYawDeg:0};
 assert.equal(b.calibrate(z,1100).ok,true);
 const snapshot=b.snapshot(1120);
 assert.equal(snapshot.status,'BENCH_ONLY');
 assert.equal(snapshot.corrected.insertionMm,40);
 assert.equal(snapshot.corrected.tipPitchDeg,10);
 assert.equal(snapshot.corrected.tipYawDeg,0);
 assert.equal(b.snapshot(1400).status,'STALE_INPUT');
 b.reset();assert.equal(b.snapshot(1450).status,'NO_INPUT');
});
test('XR16F uses the existing measured-trace and clinical reviewer gates as single authority',()=>{
 const ctx={window:{}};
 runInNewContext(load('../xr/clinical-review-gate.js'),ctx);
 const reviewer=ctx.window.PulmoXRClinicalReview;
 assert.equal(reviewer.assess({}).clinicalValidated,false);
 assert.equal(reviewer.assess({}).readyForIndependentReview,false);
 const strict=readFileSync(new URL('../xr/physical-instrument-bench.js',import.meta.url),'utf8');
 assert.ok(strict.includes('BENCH_ENGINEERING_TARGET_MET'));
 assert.ok(strict.includes('clinicalValidated:false'));
 assert.equal('acceptance' in API,false,'Packet intake must not offer competing approval paths');
});
test('XR16-FINAL module is source-isolated, never streams or actuates the scope',()=>{
 const source=load('../xr/instrument-bench.js');
 const xr=load('../xr/procedural-immersive.js');
 const root=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
 const bench=readFileSync(new URL('../validation-lab.html',import.meta.url),'utf8');
 assert.ok(root.includes('instrument-bench.js?v=XR16-FINAL-ISOLATED-BENCH'));
 assert.ok(xr.includes('const physicalBench=global.PulmoXRInstrumentTelemetry?.create?.()||null'));
 assert.ok(xr.includes("station.set('video',recommended)"));
 assert.ok(xr.includes('physicalInstrumentBench:physicalBench?.snapshot'));
 for(const keyword of ['fetch(', 'WebSocket(', 'navigator.serial.requestPort(', 'navigator.bluetooth.requestDevice('])
   assert.ok(!source.includes(keyword),'Bench must not perform remote IO or hardware pairing: '+keyword);
 assert.ok(!source.includes('advance(')&&!source.includes('fireAction(')&&!source.includes('scope.velocity'));
 assert.ok(bench.includes('PulmoXRClinicalReview'));
 assert.ok(bench.includes('id="inspectPacket"'));
});
