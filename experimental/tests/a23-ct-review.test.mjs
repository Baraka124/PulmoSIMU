import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source=readFileSync(new URL('../anatomy/source-review.js',import.meta.url),'utf8');
const ui=readFileSync(new URL('../anatomy-inspector.html',import.meta.url),'utf8');
const app=readFileSync(new URL('../anatomy/inspector-app.js',import.meta.url),'utf8');
const page={};
runInNewContext(source,{window:page});
const review=page.PulmoCTReview;
const QA={
 source_sha256:'a'.repeat(64),mesh_sha256:'b'.repeat(64),
 source_shape_vox:[40,50,60],
 sform_RAS_mm:[[.6,0,0,12],[0,.6,0,-95],[0,0,1.5,-500],[0,0,0,1]],
 model_from_ras_mm:[[.001,0,0,0],[0,.001,0,0],[0,0,.001,0],[0,0,0,1]]
};
const study=()=>review.identity({ctSha:QA.source_sha256,meshSha:QA.mesh_sha256,qa:QA,headerValid:true});
const mark=(overrides={})=>review.newMark(study(),[10,12,30],'RUL','fissure-proposal','Review needed','marker-1',overrides);
test('A2.3 source identity requires exact CT+GLB SHA, valid source sform and image header',()=>{
 assert.ok(study());
 assert.equal(review.identity({ctSha:QA.source_sha256,meshSha:QA.mesh_sha256,qa:QA,headerValid:false}),null);
 assert.equal(review.identity({ctSha:'c'.repeat(64),meshSha:QA.mesh_sha256,qa:QA,headerValid:true}),null);
 assert.equal(review.identity({ctSha:QA.source_sha256,meshSha:'c'.repeat(64),qa:QA,headerValid:true}),null);
 assert.equal(review.identity({ctSha:QA.source_sha256,meshSha:QA.mesh_sha256,
  qa:{...QA,sform_RAS_mm:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,1]]},headerValid:true}),null);
});
test('A2.3 manual fiducials are in CT voxel and RAS mm coordinates; data is not segmentation',()=>{
 const m=review.newMark(study(),[10,12,30],'RUL','fissure-proposal','Review needed','marker-1');
 assert.equal(m.region,'RUL');
 assert.equal(m.finding,'fissure-proposal');
 assert.equal(m.ras_mm.join(','),'18,-87.8,-455');
 assert.throws(()=>review.newMark(study(),[41,0,0],'RUL','fissure-proposal','bad','marker-2'),/outside/);
 assert.throws(()=>review.newMark(study(),[10,0,0],'INVALID','fissure-proposal','bad','marker-3'),/Unknown/);
 assert.throws(()=>review.newMark(study(),[10,0,0],'RUL','fissure-proposal','X'.repeat(281),'marker-4'),/Invalid/);
 const json=review.exportReview(study(),[m]);
 assert.equal(json.schema,'pulmosimu-ct-point-review-v1');
 assert.equal(json.source_type,'manual-point-review');
 assert.match(json.interpretation,/NOT CLINICAL/);
 assert.equal(review.importReview(JSON.parse(JSON.stringify(json)),study())[0].note,'Review needed');
});
test('A2.3 imported JSON is fail-closed across source geometry and coordinate forgery',()=>{
 const s=study(),m=review.newMark(s,[10,12,30],'RUL','boundary-concern','','marker-1');
 const bundle=review.exportReview(s,[m]);
 const from=obj=>JSON.parse(JSON.stringify(obj));
 const wrongCT=from(bundle);wrongCT.study.source_sha256='c'.repeat(64);
 assert.throws(()=>review.importReview(wrongCT,s),/different exact/);
 const wrongMesh=from(bundle);wrongMesh.study.mesh_sha256='c'.repeat(64);
 assert.throws(()=>review.importReview(wrongMesh,s),/different exact/);
 const wrongAffine=from(bundle);wrongAffine.study.sform_RAS_mm[0][3]+=5;
 assert.throws(()=>review.importReview(wrongAffine,s),/different exact/);
 const wrongVoxel=from(bundle);wrongVoxel.annotations[0].voxel=[999,12,30];
 assert.throws(()=>review.importReview(wrongVoxel,s),/outside/);
 const forgedRAS=from(bundle);forgedRAS.annotations[0].ras_mm=[0,0,0];
 assert.throws(()=>review.importReview(forgedRAS,s),/RAS coordinates/);
 const duplicate=from(bundle);duplicate.annotations.push(duplicate.annotations[0]);
 assert.throws(()=>review.importReview(duplicate,s),/duplicate/);
 const huge=from(bundle);huge.annotations=Array(501).fill(huge.annotations[0]);
 assert.throws(()=>review.importReview(huge,s),/different exact/);
});
test('A2.3 review stays separate from original CT data, reference bronchoscopy and physics',()=>{
 for(const key of ['id="review-add"','id="review-list"','id="review-import"','id="review-export"',
  'source-review.js?v=A23-REVIEW'])assert.ok(ui.includes(key),key);
 for(const key of ['function reviewIdentity()','function drawReviewPoints(',
  "window.PulmoCTReview.newMark(study,state.cross.slice()",
  'window.PulmoCTReview.importReview(JSON.parse(await file.text()),study)',
  'state.reviewMarks=marks','resetReview();state.qa=data;applyQa();renderSlices();'])
   assert.ok(app.includes(key),key);
 assert.ok(!app.includes('bronchoscopyAllowed=true'));
 assert.ok(!source.includes('localStorage'));
 assert.ok(!source.includes('fetch('));
});
