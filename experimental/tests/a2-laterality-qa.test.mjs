import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const manifest=JSON.parse(readFileSync(new URL('../data/nlst-a1/study-manifest.json',import.meta.url)));
const qa=JSON.parse(readFileSync(new URL('../data/nlst-a1/a2-laterality-qa.json',import.meta.url)));
const inspector=readFileSync(new URL('../anatomy/inspector-app.js',import.meta.url),'utf8');
const script=readFileSync(new URL('../anatomy/a2_laterality_qa.py',import.meta.url),'utf8');

test('A2 demonstrates reproducible RAS side-name conflict for both original GLB nodes',()=>{
 assert.equal(qa.source_sha256,manifest.source_sha256);
 assert.equal(qa.mesh_sha256,manifest.mesh_sha256);
 assert.equal(qa.overall_gate,'FAIL_RESEARCH_SEGMENTATION_LATERALITY');
 assert.equal(qa.results.length,2);
 const right=qa.results.find(x=>x.stored_label==='right');
 const left=qa.results.find(x=>x.stored_label==='left');
 assert.ok(right.area_fraction_opposite_name>.85);
 assert.ok(left.area_fraction_opposite_name>.98);
 assert.ok(right.area_weighted_RAS_x_mm>0);
 assert.ok(left.area_weighted_RAS_x_mm<0);
 assert.equal(qa.bronchoscopy_to_CT,'BLOCKED');
 assert.equal(qa.source_glb_renamed,false);
});
test('A2 failure is enforced in the public source manifest and browser even with older A1 manifests',()=>{
 assert.equal(manifest.quality_gates.laterality_labels,'FAIL_A2_SOURCE_RAS_CONFLICT');
 assert.equal(manifest.quality_gates.ct_linked_anatomy,'BLOCKED_UNTIL_LATERALITY_AND_SEGMENTATION_QA');
 assert.equal(manifest.anatomical_laterality_check.clinical_gate,'FAIL');
 assert.ok(inspector.includes('const A2_FLAGGED_GLBSHA='));
 assert.ok(inspector.includes('A2 LATERALITY QUALITY FAILURE'));
 assert.ok(inspector.includes('a2LateralityConflict()?'));
 assert.ok(inspector.includes("'registration').dataset.state=a2LateralityConflict()?'error'"));
});
test('A2 reproducibility tool checks source SHA and image sform before geometric QA',()=>{
 for(const term of ['sha256','source_sform','model_from_ras_mm',
   'trimesh.load','RAS','right','left','raise SystemExit(2)',
   '--report-only']){
   assert.ok(script.includes(term),'Missing source-integrity gate: '+term);
 }
});
