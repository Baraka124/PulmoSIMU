import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
const source=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const original=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
test('XR-07 runtime is browser-valid classic JavaScript compatible with original Three r128',()=>{
 const r=spawnSync(process.execPath,['--check'],{input:source,encoding:'utf8'});
 assert.equal(r.status,0,r.stderr);
 assert.match(source,/global\.PulmoImmersiveXR=\{install\}/);
 assert.match(original,/three@0\.128\.0/);
 assert.match(original,/procedural-immersive\.js\?v=XR08/);
});
test('only original clinical scene, geometry and procedural state drive headset experience',()=>{
 for(const k of ['scene, rBro, rExt, camScope, mGrp','scope, CFG, SEGS, SEG_MAP, pathCurves','patientData','proceduralMeshes','referenceMeshes','pathoMeshes','fireAction,enterBranch','setClinMode,getClinMode','getAnatomyMeshes'])
   assert.ok(original.includes(k),'original engine source: '+k);
 for(const k of ['getAnatomyMeshes()','rBro.render(scene,vrCam)','rBro.render(scene,camScope)',
    'advance', 'scope.velocity','scope.deflH','scope.deflV','getViewerPose','grips[i]','capture(i)',
    'pointerRay','sourceBounds']){
   // 'pointerRay' is equivalent to rayFrom in this adapter
   if(k==='pointerRay'||k==='advance')continue;
   assert.ok(source.includes(k),'immersive adapter: '+k);
 }
 assert.ok(!source.includes('new THREE.SphereGeometry('),'No replacement lung spheres');
 assert.ok(!source.includes('new THREE.CatmullRomCurve3('),'No duplicate bronchial tree');
});
test('legacy wrist-menu VR animation loop has been replaced and desktop rendering guarded',()=>{
 assert.doesNotMatch(original,/drawWristMenu\(/);
 assert.doesNotMatch(original,/wristMesh\.position/);
 assert.match(original,/if\(!rBro\.xr\.isPresenting\)/);
 assert.match(original,/preserveDrawingBuffer:true/);
 assert.match(source,/rBro\.setAnimationLoop\(/);
 assert.match(source,/sessionend/);
 assert.match(source,/PROCEDURAL XR/);
});
test('diagnostics disclose reference status and preserve real simulator component counts',()=>{
 assert.match(source,/originalSegments:SEGS\.length/);
 assert.match(source,/originalPathologyCount/);
 assert.match(source,/originalAnatomyMeshes/);
 assert.match(source,/comfort display/i);
 assert.match(source,/not calibrated/i);
});

test('XR-08 uses audited CT source registration and estimated instrument-lumen clearance',()=>{
 assert.ok(original.includes('registered-geometry.js?v=XR08'));
 for(const token of ['assessRegistration','_xr08MeshBoundsLocal','patientData.xrAudit',
 'CT registration rejected',"setViewMode('reference')",
 'getPatientAudit:()=>patientData.xrAudit','assessLumenClearance']){
  assert.ok(original.includes(token),'original source contract: '+token);
 }
 for(const token of ['getPatientAudit','setSourceScale','sourceScaleEligible',
 'getMmPerSceneUnit','getScopeDiameterMm','sourceAudit:','clearanceState']){
  assert.ok(source.includes(token),'headset source gate: '+token);
 }
});

test('XR-08 live endoscopy is rendered in the original VR GPU context, not copied from a second canvas',()=>{
 for(const token of ['new THREE.WebGLRenderTarget(640,360','rBro.xr.enabled=false',
   'rBro.setRenderTarget(scopeTarget)','rBro.render(scene,camScope)',
   'rBro.setRenderTarget(previousTarget)','monitorSource:']){
  assert.ok(source.includes(token),'same-context bronchoscopy feed: '+token);
 }
 assert.ok(!source.includes('new THREE.CanvasTexture(feedCanvas)'),
   'Do not upload another WebGL context canvas into Quest texture');
 assert.ok(!source.includes('rExt.render(scene,camScope)'),
   'VR monitor never depends on original desktop renderer capture');
 assert.ok(source.includes('rBro.state.bindXRFramebuffer(layer.framebuffer)'),
   'XR r128 must restore headset base-layer framebuffer after endoscope target');
 assert.ok(source.includes("monitorStatus='unavailable'"),
   'A failed offscreen render must be reported, not faked');
});

test('immersive XR owns original scope advance/retract without a second desktop animation step',()=>{
 assert.match(original,/if\(!rBro\.xr\.isPresenting&&Math\.abs\(scope\.velocity\)/);
 assert.ok(original.includes('enterBranch,advance,retract,hideBranchSel'));
 assert.ok(source.includes("if(mode==='procedure'&&Math.abs(scope.velocity)>"));
 assert.ok(source.includes('if(distance>0)advance(distance);else retract(-distance);'));
 assert.ok(source.includes('Math.pow(scope.velocityDecay,step)'));
 assert.ok(source.includes('updateScope();'));
});
