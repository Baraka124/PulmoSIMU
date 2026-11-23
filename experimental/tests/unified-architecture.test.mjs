import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const read=file=>readFileSync(new URL('../../'+file,import.meta.url),'utf8');
const studio=read('studio.html');
const main=read('index.html');
const gateway=read('quest-short/index.html');
const entrances=['quest.html','quest4.html'].map(read);
const originalXR=read('experimental/xr/procedural-immersive.js');
const legacy=read('experimental/vr.html');

test('unified official Quest/Studio routes point only to original procedural WebXR',()=>{
 assert.ok(studio.includes("src:'./index.html?xr=procedural&release=XR09-UNIFIED'"));
 assert.ok(studio.includes("new URL('./index.html?xr=procedural&release=XR09-UNIFIED'"));
 assert.ok(!studio.includes("src:'./experimental/vr.html'"));
 assert.ok(gateway.includes('XR-09 · Immersive procedural laboratory'));
 assert.ok(gateway.includes('release=XR09-UNIFIED'));
 assert.ok(!gateway.includes('experimental/vr.html'),'Schematic viewer is archived, not prominently linked');
 for(const page of entrances){
  assert.ok(page.includes("url=new URL('./index.html',location.href)"));
  assert.ok(!page.includes("new URL('./experimental/vr.html'"));
  assert.ok(page.includes('XR09-UNIFIED'));
 }
 assert.ok(legacy.includes('XR-06'),'Historical experiments are still source-preserved');
});
test('procedural origin owns one scene plus shared R3 reduced-order solver',()=>{
 assert.ok(main.includes('window.PulmoImmersiveXR?.install('));
 assert.ok(main.includes('loadXRReferencePhysics()'));
 assert.ok(main.includes("reference-physics.js"));
 assert.ok(originalXR.includes('global.PulmoXRReferencePhysics.step(dt)'));
 assert.ok(originalXR.includes("getMeshMode()==='reference'"));
 assert.ok(!originalXR.includes("new THREE.CatmullRomCurve3("));
});
test('XR source context cannot quietly redefine real procedure or patient state',()=>{
 assert.ok(main.includes('window.pulmoActiveReferenceRegion'));
 assert.ok(main.includes("value.channel!=='pulmosimu.reference-focus.v1'"));
 assert.ok(main.includes("getReferenceFocus:()=>meshMode==='reference'"));
 assert.ok(!main.includes('const PATIENT_F='));
 assert.ok(main.includes('if(!child.material)child.material=neutralFallback'));
});

test('immersive XR never uses decorative desktop breathing as physical deformation',()=>{
 assert.ok(main.includes('const illustrativeTime=rBro.xr.isPresenting?0:time;'));
 assert.ok(main.includes('if(!rBro.xr.isPresenting){\n    // Desktop-only illustrative breathing'));
 assert.ok(main.includes('// Stable source geometry: no invented pulsation/deformation in XR.'));
 assert.ok(main.includes('rUpperL.scale.setScalar(1);rMidL.scale.setScalar(1);'));
});
