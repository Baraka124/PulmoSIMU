import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const manifest=JSON.parse(readFileSync(new URL('../data/nlst-a1/study-manifest.json',import.meta.url)));
const qa=JSON.parse(readFileSync(new URL('../data/nlst-a1/a2-laterality-qa.json',import.meta.url)));
const inspector=readFileSync(new URL('../anatomy/inspector-app.js',import.meta.url),'utf8');
const script=readFileSync(new URL('../anatomy/a2_laterality_qa.py',import.meta.url),'utf8');

test('A2 canonical NIfTI RAS+ orientation is patient RIGHT at +X, correcting historical sign inversion',()=>{
 assert.match(qa.laterality_convention,/X>0.*RIGHT/i);
 assert.match(manifest.anatomical_laterality_check.reference_frame,/\+X patient RIGHT/);
 assert.equal(manifest.quality_gates.laterality_labels,'REVIEW_REQUIRED_OLD_RIGHT_SURFACE_CROSSES_MIDLINE');
 assert.equal(qa.critical_label_reversal,false);
 assert.match(qa.overall_gate,/BILATERAL_LABEL_REVERSAL_NOT_SUPPORTED/);
 assert.equal(qa.source_sha256,manifest.source_sha256);
 assert.equal(qa.mesh_sha256,manifest.mesh_sha256);
});
test('A2 original right surface crosses midline, but both original labels are not proven reversed',()=>{
 const right=qa.results.find(x=>x.stored_label==='right');
 const left=qa.results.find(x=>x.stored_label==='left');
 assert.ok(right.area_fraction_patient_right>.85);
 assert.ok(right.area_fraction_opposite_name>.14&&right.area_fraction_opposite_name<.16);
 assert.ok(left.area_fraction_patient_left>.98);
 assert.ok(left.area_fraction_opposite_name<.02);
 assert.equal(right.midline_crossing_requires_review,true);
 assert.equal(left.anatomical_label_reversal_supported,false);
 assert.equal(qa.bronchoscopy_to_CT,'BLOCKED');
});
test('A2 original and invalid candidate surfaces remain fail-closed in CT inspector',()=>{
 assert.ok(inspector.includes('const A2_FLAGGED_GLBSHA='));
 assert.ok(inspector.includes('A2 GEOMETRY REVIEW REQUIRED'));
 assert.ok(inspector.includes('A2 CANDIDATE INVALID'));
 assert.ok(inspector.includes('a2LateralityConflict()?'));
 assert.ok(inspector.includes("registration').dataset.state=a2LateralityConflict()?'error'"));
});
test('A2 audit Python script uses right-positive RAS and checks independent source transforms',()=>{
 for(const t of ['source_sform','model_from_ras_mm','trimesh.load','sha256',
   'NIfTI sform RAS: +X PATIENT RIGHT', 'wrong = (1 - pos) if name == "right"',
   '--report-only'])assert.ok(script.includes(t),'Missing audit invariant '+t);
 assert.ok(!script.includes('patient left X-positive'));
});
