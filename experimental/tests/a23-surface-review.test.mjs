import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const root={};
runInNewContext(readFileSync(new URL('../anatomy/lobe-territories.js',import.meta.url),'utf8'),{window:root});
runInNewContext(readFileSync(new URL('../anatomy/ct-review.js',import.meta.url),'utf8'),{window:root});
const api=root.PulmoCTReview;
const manifest={source_sha256:'a'.repeat(64),mesh_sha256:'b'.repeat(64),
 model_from_ras_mm:[[.001,0,0,0],[0,0,.001,.39],[0,-.001,0,.122],[0,0,0,1]]};
const sample={type:'right-oblique',ras_mm:[102.1234,68.5,-399.75],note:'review point'};
test('source-locked RAS surface annotation round trip',()=>{
 const data=api.session(manifest,{rightMajor:12,selected:'RUL'},[sample]);
 assert.equal(data.schema,'pulmosimu.ct-lobe-review/1');
 assert.equal(data.status,'RESEARCH_HYPOTHESES_ONLY');
 assert.deepEqual(Array.from(data.annotations[0].ras_mm),[102.12,68.5,-399.75]);
 const loaded=api.parse(JSON.stringify(data),manifest);
 assert.equal(loaded.atlas.rightMajor,12);
 assert.equal(loaded.annotations.length,1);
});
test('different source CT or mesh cannot absorb surface annotations',()=>{
 const data=api.session(manifest,{},[sample]);
 assert.throws(()=>api.parse(data,{...manifest,source_sha256:'c'.repeat(64)}),/hash mismatch/);
 assert.throws(()=>api.parse(data,{...manifest,mesh_sha256:'d'.repeat(64)}),/hash mismatch/);
});
test('unsupported records fail closed',()=>{
 assert.throws(()=>api.point({type:'malignancy',ras_mm:[0,0,0]}),/Unknown/);
 assert.throws(()=>api.point({type:'uncertain',ras_mm:[NaN,0,0]}),/Invalid RAS/);
 assert.throws(()=>api.session(manifest,{},Array(501).fill(sample)),/Maximum 500/);
 const data=JSON.parse(JSON.stringify(api.session(manifest,{},[sample])));
 data.atlas.rightMajor=999;
 assert.throws(()=>api.parse(data,manifest),/outside supported range/);
 data.atlas.rightMajor=0;data.status='CLINICALLY_VALIDATED';
 assert.throws(()=>api.parse(data,manifest),/Unsupported review schema/);
});
test('original procedural engine and A2.2 remain available',()=>{
 const page=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
 for(const expected of [
  'lobe-territories.js?v=A22-FISSURE-ATLAS',
  'ct-review.js?v=A23-RAS-REVIEW',
  'id="ct-point-type"',
  'window.importCTReviewFile=async input=>',
  'window.exportCTReview=()=>',
  'window.clearCTReview=()=>',
  'window.resetCTReviewAtlas=()=>',
  'ray.intersectObject(patientData.gltf,true)',
  'mGrp.worldToLocal(hits[0].point.clone())',
  'closeCTTerritoryAtlas();',
  "setViewMode('reference')",
  'PNEUMO·VR'
 ])assert.ok(page.includes(expected),expected);
});
