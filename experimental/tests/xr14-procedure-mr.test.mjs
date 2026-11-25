import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const guideSource=readFileSync(new URL('../xr/procedure-guide.js',import.meta.url),'utf8');
const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
const entry=readFileSync(new URL('../../quest-short/index.html',import.meta.url),'utf8');
function modelGuide(args){
 const window={};runInNewContext(guideSource,{window});
 return window.PulmoXRProcedureGuide.describe(args);
}
const originalBranches={
  0:{id:0,name:'Trachea',gen:0,children:[1,4]},
  1:{id:1,name:'L Main Bronchus',gen:1,children:[]},
  4:{id:4,name:'R Main Bronchus',gen:1,children:[]}
};
test('XR14 guide uses only actual source branch names and current scope state',()=>{
 const state=modelGuide({segmentId:0,segment:originalBranches[0],segments:originalBranches,
  position:.66,atBranch:false,clinicalMode:'nav',monitorStatus:'live'});
 assert.equal(state.location,'Trachea');
 assert.equal(state.generation,0);
 assert.equal(state.progressPercent,66);
 assert.equal(state.tool,'NAVIGATION');
 assert.equal(state.monitor,'LIVE SOURCE VIEW');
 assert.equal(state.branches.length,2);
 assert.equal(state.branches[0].label,'L Main Bronchus');
 assert.equal(state.branches[1].label,'R Main Bronchus');
 assert.equal(state.canChooseBranch,false);
 assert.equal(state.branches[0].enabled,false);
 assert.match(state.provenance,/NOT PATIENT-SPECIFIC/);
});
test('XR14 guide only enables existing junction branches',()=>{
 const state=modelGuide({segmentId:0,segment:originalBranches[0],segments:originalBranches,
  position:.98,atBranch:true,clinicalMode:'biopsy'});
 assert.equal(state.heading,'AIRWAY JUNCTION');
 assert.equal(state.canChooseBranch,true);
 assert.equal(state.branches[0].enabled,true);
 assert.equal(state.branches[0].index,0);
 assert.equal(state.branches[1].index,1);
 assert.equal(state.tool,'BIOPSY SIMULATION');
 const terminal=modelGuide({segmentId:1,segment:originalBranches[1],segments:originalBranches,
  position:.97,atBranch:false,clinicalMode:'nav'});
 assert.equal(terminal.branches.length,0);
 assert.equal(terminal.heading,'TERMINAL REFERENCE SEGMENT');
});
test('XR14 guide does not reinterpret modeled clearance as verified patient safety',()=>{
 const a=modelGuide({segmentId:0,segment:originalBranches[0],segments:originalBranches,
  clearance:{canAdvance:true},monitorStatus:'unavailable'});
 assert.match(a.clearance,/REFERENCE.*ESTIMATE/);
 assert.equal(a.monitor,'SOURCE MONITOR UNAVAILABLE');
 assert.ok(!a.clearance.includes('SAFE'));
 const blocked=modelGuide({segmentId:0,segment:originalBranches[0],segments:originalBranches,
  clearance:{canAdvance:false}});
 assert.match(blocked.clearance,/BLOCKING/);
 const empty=modelGuide({segmentId:null,segment:null,segments:{}});
 assert.equal(empty.branches.length,0);
 assert.equal(empty.heading,'NO SOURCE AIRWAY');
});
test('XR14 spatial guide drives original enterBranch and instruments, not duplicate simulations',()=>{
 for(const token of [
   'global.PulmoXRProcedureGuide.describe({',
   "enterBranch(current.branches[index].index)",
   "setClinMode(modes[(modes.indexOf(getClinMode())+1)%modes.length])",
   "guidePlane.name='xr14-original-procedure-guide'",
   "guidePlane.visible=next==='procedure'&&guideVisible",
   "guideHitboxes",
   "const guideWasVisible=guidePlane?.visible||false",
   "if(guidePlane)guidePlane.visible=guideWasVisible",
   'paintGuide();',
   'panelIntersection(r)',
   'guideActions++'
 ]) assert.ok(xr.includes(token),'Missing source/procedure integration '+token);
 assert.ok(main.includes('procedure-guide.js?v=XR14-SOURCE-GUIDE'));
 assert.ok(!guideSource.includes('fetch('),'Guidance has no network access');
 assert.ok(!guideSource.includes('Math.random('),'Guidance never invents clinical routes');
});
test('XR14 real MR entry is browser-feature gated and restores opaque desktop/VR',()=>{
 assert.ok(main.includes('alpha:true,powerPreference:'));
 for(const token of [
  "xrCapabilities.request(presentation)",
  "xrCapabilities.bind(session,presentation)",
  "arButton.style.display=q.mr==='supported'?'block':'none'",
  "scene.background=null;scene.fog=null",
  "rBro.setClearColor(0x000000,0)",
  "room.visible=xrPresentation!=='ar'",
  "scene.background=backgroundBeforeXR;scene.fog=fogBeforeXR",
  "if(clearBeforeXR)rBro.setClearColor(clearBeforeXR,clearAlphaBeforeXR)",
  "xrCapabilities.end()",
  "xrPresentation='vr'"
 ]) assert.ok(xr.includes(token),'Missing passthrough safety/restore '+token);
 assert.ok(entry.includes('Mixed Reality · Quest 2 passthrough preview'));
 assert.ok(entry.includes('grayscale'));
 assert.ok(entry.includes('not verified tabletop anchoring'));
});
