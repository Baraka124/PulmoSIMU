/* XR17B deterministic source placement tests — no browser device claims. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source=readFileSync(new URL('../xr/mr-placement.js',import.meta.url),'utf8');
const context={window:{}};runInNewContext(source,context);
const MR=context.window.PulmoXRMRPlacement;
const position=(x,y,z)=>({position:{x,y,z}});
const makeSession=({anchor=false}={})=>{
 let cancelled=0,anchorDeleted=0;
 const hit={
  getPose:()=>({transform:position(.8,0,-.75)}),
  ...(anchor?{createAnchor:()=>Promise.resolve({anchorSpace:{name:'surface-anchor'},
    delete:()=>{anchorDeleted++}})}:{})
 };
 const session={
  requestReferenceSpace:async(type)=>{
   assert.equal(type,'viewer');return {kind:'viewer'}
  },
  requestHitTestSource:async(options)=>{
   assert.equal(options.space.kind,'viewer');
   return {cancel:()=>{cancelled++}}
  }
 };
 const frame={
  getHitTestResults:()=>[hit],
  getPose:()=>({transform:position(.82,.01,-.74)})
 };
 return {session,frame,ref:{kind:'local-floor'},events:()=>({cancelled,anchorDeleted})};
};
test('XR17B missing hit test fails closed and cannot place',async()=>{
 const m=MR.create();
 assert.equal(m.snapshot().state,'idle');
 const bad=await m.start({requestReferenceSpace:async()=>({})});
 assert.equal(bad.state,'unsupported');
 assert.equal(bad.error,'HIT_TEST_API_UNAVAILABLE');
 assert.equal(m.confirm().accepted,false);
 assert.equal(m.snapshot().userConfirmed,false);
 assert.equal(m.preview(),true);
 assert.equal(m.snapshot().unregisteredPreview,true);
 assert.equal(m.snapshot().patientCTRegistration,false);
 assert.equal(m.snapshot().physicalDepthOcclusion,false);
});
test('XR17B real hit test needs actual result and explicit user confirmation',async()=>{
 const e=makeSession();
 const m=MR.create();
 assert.equal((await m.start(e.session)).state,'scanning');
 assert.equal(m.confirm().accepted,false);
 const a=m.scan(e.frame,e.ref);
 assert.equal(a.kind,'candidate');
 assert.equal(a.point.x,.8);
 assert.equal(m.snapshot().hitAvailable,true);
 assert.equal(m.snapshot().userConfirmed,false);
 assert.equal(m.confirm().accepted,true);
 const p=m.scan(e.frame,e.ref);
 assert.equal(p.kind,'placed');
 assert.equal(p.point.z,-.75);
 assert.equal(m.snapshot().state,'placed');
 assert.equal(m.snapshot().anchorStatus,'not-anchored');
 assert.equal(m.snapshot().placementCount,1);
 assert.equal(m.scan(e.frame,e.ref),null,'placed unanchored model stays fixed');
 assert.equal(m.reposition(),true);
 assert.equal(m.snapshot().state,'scanning');
 assert.equal(m.snapshot().userConfirmed,false);
 m.reset();
 assert.equal(e.events().cancelled,1);
 assert.equal(m.snapshot().state,'idle');
});
test('XR17B optional genuine anchor applies only frame-supplied anchor deltas',async()=>{
 const e=makeSession({anchor:true});
 const m=MR.create();
 await m.start(e.session);
 m.scan(e.frame,e.ref);
 m.confirm();
 const placed=m.scan(e.frame,e.ref);
 assert.equal(placed.kind,'placed');
 await Promise.resolve();
 assert.equal(m.snapshot().anchorStatus,'anchor-acquired');
 const delta=m.scan(e.frame,e.ref);
 assert.equal(delta.kind,'anchor-delta');
 assert.ok(Math.abs(delta.delta.x-.02)<1e-12);
 assert.ok(Math.abs(delta.delta.y-.01)<1e-12);
 assert.ok(Math.abs(delta.delta.z-.01)<1e-12);
 assert.equal(m.snapshot().anchorStatus,'tracking');
 m.reposition();
 assert.equal(e.events().anchorDeleted,1);
 assert.equal(m.snapshot().anchorStatus,'none');
});
test('XR17B actual XR hit-test failure loses target, never fabricates placement',async()=>{
 const e=makeSession();
 const m=MR.create();await m.start(e.session);
 assert.equal(m.scan({getHitTestResults:()=>[]},e.ref),null);
 assert.equal(m.confirm().accepted,false);
 assert.equal(m.snapshot().hitAvailable,false);
 m.scan(e.frame,e.ref);
 assert.equal(m.confirm().accepted,true);
 m.reset();
 assert.equal(m.scan(e.frame,e.ref),null);
 assert.equal(m.snapshot().userConfirmed,false);
});
test('XR17B original engine owns scene, MR placement has no network or CT routing',()=>{
 const index=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
 const runtime=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
 const caps=readFileSync(new URL('../xr/xr-capabilities.js',import.meta.url),'utf8');
 assert.ok(index.includes('mr-placement.js?v=XR17B-SURFACE-PLACEMENT'));
 assert.ok(caps.includes("'hit-test','anchors'"));
 for(const token of [
  "const mrPlacement=global.PulmoXRMRPlacement.create()",
  "new THREE.RingGeometry(.075,.092,48)",
  "mrPlacement.start(rBro.xr.getSession())",
  "mrPlacement.scan(frame,rBro.xr.getReferenceSpace())",
  "placeOnMRSurface(hit.point)",
  "mrPlacement.reposition()",
  "mrPlacement.preview()",
  "mrPlacementEvidence:mrPlacement.snapshot()",
  "mGrp.visible=false",
  "if(!placed&&xrPresentation!=='ar')",
  "getMeshMode()==='patient'",
  "enterBranch(current.branches[index].index)"
 ])assert.ok(runtime.includes(token),'Missing source boundary or MR integration: '+token);
 for(const no of ['fetch(', 'WebSocket(', 'getUserMedia(', 'SEG_MAP', 'new THREE.Mesh('])
  assert.equal(source.includes(no),false,'MR must not create scientific scene or upload: '+no);
});
