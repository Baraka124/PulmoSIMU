import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

const source=readFileSync(new URL('../xr/session-health.js',import.meta.url),'utf8');
function create(capacity=20){
  const window={};
  runInNewContext(source,{window});
  let now=0;
  return {
    health:window.PulmoXRSessionHealth.createSessionHealth({clock:()=>now,capacity}),
    advance:delta=>{now+=delta}
  };
}
test('local XR session measurements use bounded CPU-side observations, not fabricated GPU FPS',()=>{
  const {health,advance}=create();
  health.begin(72);
  for(let i=0;i<12;i++){
    health.startFrame(i*13.9);
    advance(2);
    health.endFrame({renderMs:1.3,monitorMs:i%3===0?1.2:0});
    advance(11.9);
  }
  const report=health.end();
  assert.equal(report.reportedHeadsetFrameRateHz,72);
  assert.equal(report.framesObserved,12);
  assert.equal(report.framesRetained,12);
  assert.ok(report.callbackIntervals.p95Ms>=13);
  assert.ok(report.callbackCpu.p95Ms>=2);
  assert.equal(report.intervalsOver1Point5Budgets,0);
  assert.match(report.classification,/not GPU or clinical validation/);
  assert.ok(!('patientId' in report));
  assert.equal(report.active,false);
});
test('long sessions retain bounded samples while preserving total frame and pose loss counts',()=>{
  const {health,advance}=create(10);
  health.begin(90);
  for(let i=0;i<80;i++){
    health.startFrame(i*12);
    advance(3);
    health.endFrame({poseAvailable:i%5!==0});
  }
  const report=health.end();
  assert.equal(report.framesRetained,10);
  assert.equal(report.framesObserved,80);
  assert.equal(report.viewerPoseMissingFrames,16);
  assert.equal(report.intervalsOver1Point5Budgets,0);
});
test('tracking, controller and monitor loss is explicitly counted and reset per session',()=>{
  const {health,advance}=create();
  assert.equal(health.event('unexpected patient/mesh state'),false);
  health.begin();
  health.event('viewerPoseLost');health.event('handTrackingLost');
  health.event('controllerDisconnected');health.event('monitorUnavailable');
  health.startFrame(10);advance(6);health.endFrame({poseAvailable:false});
  const snapshot=health.end();
  assert.equal(snapshot.viewerPoseMissingFrames,1);
  assert.equal(snapshot.trackingAndMonitorEvents.controllerDisconnected,1);
  assert.equal(snapshot.trackingAndMonitorEvents.monitorUnavailable,1);
  assert.equal(snapshot.reportedHeadsetFrameRateHz,null);
  assert.equal(snapshot.intervalsOver1Point5Budgets,null);
  health.begin(72);
  assert.equal(health.snapshot().framesObserved,0);
  assert.equal(health.snapshot().trackingAndMonitorEvents.handTrackingLost,0);
});
test('XR integration exposes a user-initiated local report and does not upload health records',()=>{
  const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
  const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
  assert.ok(main.includes('/xr/session-health.js?v=XR09-PROFILE'));
  assert.ok(xr.includes('global.PulmoXRSessionHealth.createSessionHealth()'));
  assert.ok(xr.includes('health.startFrame(time)'));
  assert.ok(xr.includes('health.endFrame('));
  assert.ok(xr.includes('health.event(\'controllerDisconnected\')'));
  assert.ok(xr.includes('health.event(\'handTrackingLost\')'));
  assert.ok(xr.includes('health.event(\'viewerPoseLost\')'));
  assert.ok(xr.includes('health.snapshot()'));
  assert.ok(xr.includes('URL.createObjectURL(blob)'));
  assert.ok(!source.includes('fetch('));
});
