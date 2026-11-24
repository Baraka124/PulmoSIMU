/* XR-16 bronchoscopy workstation: numerical invariants and original-engine gates. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

const source=readFileSync(new URL('../xr/bronchoscopy-station.js',import.meta.url),'utf8');
const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const original=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
const sandbox={window:{}};
runInNewContext(source,sandbox);
const S=sandbox.window.PulmoXRBronchoscopyStation;

test('XR16 source-only station provides bounded three-layout spatial workbench',()=>{
 const s=S.create();
 assert.equal(s.snapshot().layout,'stacked');
 assert.ok(s.snapshot().pose.monitor.depth>1);
 assert.equal(s.next('layout').layout,'side');
 assert.equal(s.next('layout').layout,'focus');
 assert.equal(s.next('layout').layout,'stacked');
 assert.equal(s.snapshot().capture.width,800);
 assert.equal(s.next('video').capture.width,1024);
 assert.equal(s.next('video').capture.width,640);
 assert.equal(s.next('video').capture.width,800);
 assert.equal(s.next('optics').fovDegrees,110);
 assert.equal(s.next('optics').fovDegrees,88);
 assert.equal(s.next('optics').fovDegrees,100);
 assert.equal(s.next('control').insertionMultiplier,.42);
 assert.equal(s.reset().insertionMultiplier,1);
 assert.equal(s.snapshot().layout,'stacked');
 for(const layout of Object.values(S.LAYOUTS)){
   for(const target of [layout.monitor,layout.guide])
     assert.ok(target.scale>=.8&&target.scale<=1.4&&target.depth>0.9&&target.depth<1.5);
 }
 for(const quality of Object.values(S.VIDEO)){
   assert.equal(quality.width/quality.height,16/9);
   assert.ok(quality.intervalMs>=33&&quality.intervalMs<=90);
 }
});

test('XR16 tip deflection holds at neutral and is bounded under prolonged input',()=>{
 let h=0;
 for(let k=0;k<300;k++)h=S.tipStep(h,1,.016);
 assert.equal(h,1);
 for(let k=0;k<80;k++)h=S.tipStep(h,0,.016);
 assert.equal(h,1,'a released tip control holds rather than unexpectedly recentres');
 for(let k=0;k<400;k++)h=S.tipStep(h,-1,.016);
 assert.equal(h,-1);
 assert.equal(S.tipStep(.5,1,0),.5);
 assert.equal(S.tipStep(0,1,99),.0696);
});

test('XR16 insertion, shaft rotation and wrist articulation are independent',()=>{
 assert.equal(S.velocity(0,14),0);
 assert.equal(S.velocity(.09,14),0);
 assert.ok(S.velocity(-1,14)>0);
 assert.ok(S.velocity(1,14)<0);
 assert.ok(Math.abs(S.velocity(-1,14,.42))<Math.abs(S.velocity(-1,14)));
 let roll=0;
 for(let k=0;k<100;k++)roll=S.rollStep(roll,1,.016);
 assert.ok(roll>0&&roll<=Math.PI);
 assert.equal(S.rollStep(roll,0,.016),roll);
 assert.ok(S.rateLimit(0,1,.016)<=.03200001);
 assert.equal(S.rateLimit(0,1,.016),.032);
});

test('XR16 uses original clinical engine and independent world-space displays',()=>{
 for(const token of [
  'const station=global.PulmoXRBronchoscopyStation.create()',
  'scope.velocity=instrument.velocity(y,CFG.scopeSpd,station.snapshot().insertionMultiplier)',
  'scope.deflH=instrument.tipStep(scope.deflH,x,dt)',
  'scope.deflV=instrument.tipStep(scope.deflV,-y,dt)',
  'scope.roll=instrument.rollStep(scope.roll,x,dt)',
  'scopeTarget.setSize(settings.capture.width,settings.capture.height)',
  'camScope.fov=settings.fovDegrees;camScope.updateProjectionMatrix()',
  "monitorRoot.name='xr16-independent-endoscopy-display'",
  "action==='view'&&mode==='procedure'",
  "menuPage='station';actionHitbox=panelLayouts.station",
  'const pose=station.snapshot().pose',
  "function place(object,spec)",
  'place(scopeScreen,pose.monitor)',
  'place(guidePlane,pose.guide)',
  "monitorSource:'same-renderer-offscreen'",
  'position:scope.t,atBranch:scope.atBranch',
  'enterBranch(current.branches[index].index)',
  "getMeshMode()==='patient'",
  'camScope.fov=originalScopeFov'
 ])assert.ok(xr.includes(token),'XR-16 implementation missing: '+token);
 assert.ok(original.includes('bronchoscopy-station.js?v=XR16-INSTRUMENT-STATION'));
 assert.ok(original.includes('procedural-immersive.js?v=XR16-BRONCHOSCOPY-ENGINEERING'));
 assert.ok(original.includes('vec3 vD=normalize(cameraPosition-vWP)'),
  'Mucosa illumination must follow the actual camera, not the scene origin');
 assert.ok(!source.includes('fetch('),'Station must not fetch or transmit CT data');
});
