import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const src=(p)=>readFileSync(new URL(p,import.meta.url),'utf8');
const original=JSON.parse(src('../data/nlst-a1/study-manifest.json'));
const candidate=JSON.parse(src('../data/nlst-a1/a2-candidate-manifest.json'));
const qa=JSON.parse(src('../data/nlst-a1/a2-candidate-qa.json'));
const voxel=JSON.parse(src('../data/nlst-a1/a2-candidate-voxel-qa.json'));
const inspector=src('../anatomy/inspector-app.js');
const engine=src('../xr/procedural-immersive.js');
const deprecated=src('../anatomy/a2_ras_relabel.py');
const voxelScript=src('../anatomy/a2_ct_surface_voxel_qa.py');

test('Archived sign-inverted A2 candidate is explicitly invalid and cannot be promoted',()=>{
 assert.equal(candidate.source_sha256,original.source_sha256);
 assert.notEqual(candidate.mesh_sha256,original.mesh_sha256);
 assert.equal(candidate.mesh_sha256,'8f8bea9a95fc21a44ca10244c91c249d2836539bf119973bcb18c614c9886d2f');
 assert.equal(candidate.quality_gates.laterality_labels,'INVALID_RAS_SIGN_SWAP_DO_NOT_USE');
 assert.equal(candidate.quality_gates.ct_linked_anatomy,'BLOCKED_INVALID_CANDIDATE_LABELS');
 assert.equal(candidate.distribution_authorized,false);
 assert.equal(candidate.binary_objects_in_git,false);
 assert.equal(candidate.anatomical_laterality_check.clinical_gate,'BLOCKED');
 assert.ok(deprecated.includes('raise RuntimeError("BLOCKED: this historical candidate'));
});
test('Historical geometry and voxel measurements retained with explicit correction flags',()=>{
 assert.equal(qa.source_geometry_original_unchanged,true);
 assert.equal(qa.laterality_name_conflict_corrected_in_candidate,false);
 assert.equal(qa.historical_laterality_swap_invalid,true);
 assert.equal(qa.findings.reduce((n,x)=>n+x.faces,0),50188);
 assert.equal(qa.CT_linked_bronchoscopy,'BLOCKED');
 assert.equal(qa.clinical_use,false);
 assert.equal(voxel.overlap_voxels,27702);
 assert.ok(voxel.overlap_volume_ml>16&&voxel.overlap_volume_ml<17);
 assert.equal(voxel.mask_approval,'REJECTED_INVALID_SIDE_RELABELLING_AND_PENDING_CLINICAL_REVIEW');
 assert.ok(voxel.measurements.every(x=>!x.side_label_valid));
 assert.match(voxelScript,/x_ras > 0 if side == "left" else x_ras < 0/);
});
test('The source browser warns against invalid A2 candidate and never couples it to procedure',()=>{
 assert.ok(inspector.includes('A2_CANDIDATE_SHA='));
 assert.ok(inspector.includes('A2 CANDIDATE INVALID'));
 assert.ok(inspector.includes('16.3 mL'));
 assert.ok(inspector.includes('A2 CANDIDATE · INVALID RAS LABELS'));
 assert.ok(inspector.includes('CT-linked bronchoscopy'));
 assert.ok(!inspector.includes('enterBranch('));
 assert.ok(!engine.includes('A2_CANDIDATE_SHA'));
});
