/* XR17A deterministic capability negotiation and user-gesture session contracts. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const src=readFileSync(new URL('../xr/xr-capabilities.js',import.meta.url),'utf8');
const context={window:{}};runInNewContext(src,context);
const M=context.window.PulmoXRCapabilities;
test('XR17A desktop always available; missing WebXR fails closed',async()=>{
 const r=M.create();
 const v=await r.probe();
 assert.equal(v.desktop,'available');
 assert.equal(v.webxrAvailable,false);
 assert.equal(v.sessions.vr,'unavailable');
 assert.equal(v.sessions.mr,'unavailable');
 assert.equal(v.worldAnchor,'NOT_IMPLEMENTED');
 assert.equal(v.physicalOcclusion,'NOT_IMPLEMENTED');
 assert.throws(()=>r.request('vr'),/WEBXR_NOT_AVAILABLE/);
 assert.equal(r.snapshot().sessionActive,false);
});
test('XR17A supports VR without inventing MR or a room mesh',async()=>{
 const calls=[];
 const xr={
  isSessionSupported:async(mode)=>mode==='immersive-vr',
  requestSession:(mode,options)=>{
   calls.push({mode,options});
   return Promise.resolve({enabledFeatures:['local-floor','hand-tracking']});
  }
 };
 const r=M.create({xr});
 const available=await r.probe();
 assert.equal(available.sessions.vr,'supported');
 assert.equal(available.sessions.mr,'unsupported');
 assert.throws(()=>r.request('mr'),/IMMERSIVE_AR_UNSUPPORTED/);
 const session=await r.request('vr');
 assert.equal(calls.length,1);
 assert.equal(calls[0].mode,'immersive-vr');
 assert.deepEqual(Array.from(calls[0].options.optionalFeatures),
  ['local-floor','bounded-floor','hand-tracking']);
 let v=r.bind(session,'vr');
 assert.equal(v.activeMode,'vr');
 assert.equal(v.actualFeatures['local-floor'],'enabled');
 assert.equal(v.actualFeatures['bounded-floor'],'not-granted');
 assert.equal(v.actualFeatures['hand-tracking'],'enabled');
 assert.equal(v.actualFeatures['anchors'],'not-requested');
 assert.equal(v.actualFeatures['depth-sensing'],'not-requested');
 assert.equal(v.worldAnchor,'NOT_IMPLEMENTED');
 assert.equal(v.patientSpecificBronchoscopy,'NOT_VALIDATED');
 v=r.end();
 assert.equal(v.activeMode,'desktop');
 assert.equal(v.sessionActive,false);
 assert.equal(v.lastSession.mode,'vr');
});
test('XR17A MR preview never fabricates spatial anchors or optional grants',async()=>{
 const xr={
  isSessionSupported:async()=>true,
  requestSession:()=>Promise.resolve({ /* no enabledFeatures exposed */ })
 };
 const r=M.create({xr});await r.probe();
 const s=await r.request('ar');
 const v=r.bind(s,'ar');
 assert.equal(v.activeMode,'mr');
 assert.equal(v.actualFeatures['local-floor'],'not-reported');
 assert.equal(v.actualFeatures['anchors'],'not-requested');
 assert.equal(v.spatialRegistration,'NOT_IMPLEMENTED');
 assert.equal(v.physicalOcclusion,'NOT_IMPLEMENTED');
 assert.equal(v.featureReporting,'runtime-did-not-report-enabled-features');
});
test('XR17A failed session leaves original 2D context usable',async()=>{
 const r=M.create({xr:{
  isSessionSupported:async()=>true,
  requestSession:()=>Promise.reject(new Error('Denied by browser'))
 }});
 await r.probe();
 await assert.rejects(()=>r.request('vr'),/Denied/);
 assert.equal(r.snapshot().sessionActive,false);
 assert.equal(r.snapshot().desktop,'available');
 assert.match(r.snapshot().lastError,/Denied/);
 assert.throws(()=>r.bind(null,'vr'),/UNRECOGNIZED_XR_SESSION/);
});
test('XR17A original engine owns anatomy and procedures across presentation modes',()=>{
 const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
 const page=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
 const launch=readFileSync(new URL('../../quest-short/index.html',import.meta.url),'utf8');
 assert.ok(page.includes('xr-capabilities.js?v=XR17A-MODE-NEGOTIATION'));
 assert.ok(xr.includes('const xrCapabilities=global.PulmoXRCapabilities.create'));
 assert.ok(xr.includes('xrCapabilities.request(presentation)'));
 assert.ok(xr.includes('xrCapabilities.bind(session,presentation)'));
 assert.ok(xr.includes('xrCapabilities.end()'));
 assert.ok(xr.includes('xrCapabilityEvidence'));
 assert.ok(launch.includes('id="vr-capability"'));
 assert.ok(launch.includes('id="mr-capability"'));
 assert.ok(launch.includes('window.pulmoXRLauncherStatus=()=>runtime.snapshot()'));
 assert.ok(xr.includes('advance(distance);else retract(-distance)'));
 assert.ok(xr.includes("getMeshMode()==='patient'"));
 assert.ok(!src.includes('Mesh(')&&!src.includes('SEG_MAP')&&!src.includes('fetch('));
});
