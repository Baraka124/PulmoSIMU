/* XR17C tracked right-controller proxy is illustrative and non-actuating. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const code=readFileSync(new URL('../xr/tracked-tool-proxy.js',import.meta.url),'utf8');
const ctx={window:{}};runInNewContext(code,ctx);
const p=ctx.window.PulmoXRTrackedToolProxy.create();
test('XR17C actual 6DoF controller proxy requires explicit opt in',()=>{
 assert.equal(p.snapshot().enabled,false);
 assert.equal(p.observe({rightGripTracked:true,procedure:true,immersiveVR:true}),false);
 p.set(true);
 assert.equal(p.observe({rightGripTracked:true,procedure:true,immersiveVR:true}),true);
 assert.equal(p.snapshot().tracking,true);
 assert.equal(p.snapshot().observedFrames,1);
 assert.equal(p.snapshot().sensorCalibration,false);
 assert.equal(p.snapshot().physicalBronchoscopeConnected,false);
 assert.equal(p.snapshot().drivesScope,false);
 assert.equal(p.snapshot().realDeviceValidation,false);
});
test('XR17C proxy hides on tracking loss, procedure exit and session reset',()=>{
 assert.equal(p.observe({rightGripTracked:false,procedure:true,immersiveVR:true}),false);
 assert.equal(p.snapshot().trackingLossTransitions,1);
 assert.equal(p.observe({rightGripTracked:true,procedure:false,immersiveVR:true}),false);
 assert.equal(p.observe({rightGripTracked:true,procedure:true,immersiveVR:false}),false);
 assert.equal(p.observe({rightGripTracked:true,procedure:true,immersiveVR:true}),true);
 p.reset();
 assert.equal(p.snapshot().tracking,false);
 assert.equal(p.snapshot().enabled,false);
 assert.equal(p.snapshot().observedFrames,0);
});
test('XR17C 6DoF proxy geometry exists outside patient anatomy and source bronchoscope',()=>{
 const runtime=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
 const source=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
 assert.ok(source.includes('tracked-tool-proxy.js?v=XR17C-6DOF-PROXY'));
 for(const token of [
  "toolProxyOverlay.name='xr17c-6dof-controller-only-reference-marker'",
  "new THREE.CylinderGeometry(",
  "toolProxy.observe({",
  "Boolean(controllerSources[i]?.gripSpace)",
  "if(toolProxyOverlay.parent!==target)target.add(toolProxyOverlay)",
  "toolProxyOverlay.visible=valid",
  "if(action==='tool-proxy')",
  "XR17CTrackedControllerProxy:toolProxy.snapshot()",
  "toolProxy.reset()",
  "toolProxyOverlay.visible=false"
 ])assert.ok(runtime.includes(token),'Proxy missing: '+token);
 assert.ok(!code.includes('scope.'));
 assert.ok(!code.includes('fetch('));
 assert.ok(!code.includes('WebSocket('));
});
