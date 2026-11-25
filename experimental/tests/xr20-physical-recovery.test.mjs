import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const html=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
test('actual original advance and branch clearance evaluate the declared diameter',()=>{
 const root={};runInNewContext(readFileSync(new URL('../xr/registered-geometry.js',import.meta.url),'utf8'),{window:root});
 const start=html.indexOf('function advance(dt){'),end=html.indexOf('function showBranchSel()',start);
 const branch=html.slice(html.indexOf('function enterBranch(childIdx){'),html.indexOf('const raycaster=new THREE.Raycaster()'));
 const scope={segId:0,t:.1,velocity:1,history:[]};
 const ctx={scope,SEG_MAP:{0:{r0:.098,r1:.09,children:[1]},1:{r0:.08,r1:.06,children:[]}},scopeDiamMm:3.8,mmPerUnit:100,rBro:{xr:{isPresenting:true}},window:{PulmoXR08:root.PulmoXR08},getScopeResistance:()=>0,updateScope(){},showBranchSel(){},hideBranchSel(){},showToast(){},console};
 runInNewContext(html.slice(start,end)+branch+';advance(.1);',ctx);
 assert.ok(scope.t>.1,'actual advancement must occur');
 runInNewContext(branch+';enterBranch(0);',ctx);
 assert.equal(scope.segId,1);assert.equal(scope.t,.01);assert.equal(scope.history.length,1);
});
test('XR20 retains reachable interaction and explicit fallback routes',()=>{
 for(const s of ["function frameGroup(distance=.76,width=.52)","confirmOrManualMR()","scope-forward","scope-back","nativeLayersOptIn=false","instrument.inputAxis","floorGranted","if(pressed&&!handPinched[i])","updateHandVisual(i,source,frame,space)","ctx.getMRHiddenObjects?.()"])assert.ok(xr.includes(s),s);
});
