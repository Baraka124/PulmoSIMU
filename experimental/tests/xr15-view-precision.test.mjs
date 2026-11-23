import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source=readFileSync(new URL('../xr/view-history.js',import.meta.url),'utf8');
const runtime=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
function history(){const window={};runInNewContext(source,{window});return window.PulmoXRViewHistory.create(3)}
const pose=(x=0,s=1)=>({position:[x,0,0],quaternion:[0,0,0,1],scale:s});
test('XR15 reversible anatomical view stores one transaction per actual manipulation',()=>{
 const h=history();
 assert.equal(h.begin(pose(0)),true);
 assert.equal(h.begin(pose(0)),false);
 assert.equal(h.finish(pose(1)),true);
 assert.equal(h.status().undo,1);
 const back=h.undo(pose(1));
 assert.equal(back.position[0],0);
 assert.equal(h.status().redo,1);
 const next=h.redo(back);
 assert.equal(next.position[0],1);
 h.clear();
 assert.equal(h.status().undo,0);
 assert.equal(h.status().redo,0);
});
test('XR15 rejects no-change snapshots and clears redo after fresh movement',()=>{
 const h=history();
 h.begin(pose(1));assert.equal(h.finish(pose(1)),false);
 h.begin(pose(0));h.finish(pose(2));
 h.undo(pose(2));assert.equal(h.status().redo,1);
 h.begin(pose(0));h.finish(pose(3));
 assert.equal(h.status().redo,0);
});
test('XR15 runtime preserves original meshes and exposes functional controls',()=>{
 for(const t of ['viewHistory.begin','viewHistory.finish','viewHistory.undo','viewHistory.redo',
  "pivotMode='center'","axisMode='yaw'","{id:'undo'","{id:'redo'","{id:'pivot'","{id:'axis'",
  "a.id==='pivot'","a.id==='axis'","finishViewChange()","beginViewChange()","attempts.thumbstickFrames++",
  "viewHistory:viewHistory.status()","axisList=['yaw','pitch','roll']"]){
   assert.ok(runtime.includes(t),'Missing source-preserving precision interaction: '+t)
 }
 assert.ok(main.includes('view-history.js?v=XR15-UNDO'));
 assert.ok(main.includes('procedural-immersive.js?v=XR15-VIEW-CONTROLS'));
 assert.ok(!source.includes('geometry.attributes.position.setXYZ'));
 assert.ok(!source.includes('mesh.geometry'));
});
