import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const original=JSON.parse(readFileSync(new URL('../data/nlst-a1/study-manifest.json',import.meta.url),'utf8'));
const candidate=JSON.parse(readFileSync(new URL('../data/nlst-a1/a2-candidate-manifest.json',import.meta.url),'utf8'));
const qa=JSON.parse(readFileSync(new URL('../data/nlst-a1/a2-candidate-qa.json',import.meta.url),'utf8'));
const voxel=JSON.parse(readFileSync(new URL('../data/nlst-a1/a2-candidate-voxel-qa.json',import.meta.url),'utf8'));
const inspector=readFileSync(new URL('../anatomy/inspector-app.js',import.meta.url),'utf8');
const engine=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const rename=readFileSync(new URL('../anatomy/a2_ras_relabel.py',import.meta.url),'utf8');
const maskAudit=readFileSync(new URL('../anatomy/a2_ct_surface_voxel_qa.py',import.meta.url),'utf8');

test('A2 candidate uses exactly the same NIfTI source and an independent versioned mesh hash',()=>{
 assert.equal(candidate.source_sha256,original.source_sha256);
 assert.notEqual(candidate.mesh_sha256,original.mesh_sha256);
 assert.equal(candidate.mesh_sha256,'8f8bea9a95fc21a44ca10244c91c249d2836539bf119973bcb18c614c9886d2f');
 assert.deepEqual(candidate.source_shape_vox,[512,512,150]);
 assert.deepEqual(candidate.sform_RAS_mm,original.sform_RAS_mm);
 assert.deepEqual(candidate.model_from_ras_mm,original.model_from_ras_mm);
 assert.equal(candidate.quality_gates.laterality_labels,'CANDIDATE_CORRECTED_RAS_INDEPENDENT_REVIEW_PENDING');
 assert.equal(candidate.quality_gates.bronchoscope_to_ct_registration,'BLOCKED');
 assert.equal(candidate.quality_gates.mask_accuracy,'NOT_INDEPENDENTLY_VALIDATED');
 assert.equal(candidate.binary_objects_in_git,false);
});
test('A2 relabel preserves all original triangle topology; masks remain provisional',()=>{
 assert.equal(qa.candidate_mesh_sha256,candidate.mesh_sha256);
 assert.equal(qa.original_mesh_sha256,original.mesh_sha256);
 assert.equal(qa.source_geometry_original_unchanged,true);
 assert.equal(qa.laterality_name_conflict_corrected_in_candidate,true);
 assert.equal(qa.findings.length,2);
 assert.equal(qa.findings.reduce((s,x)=>s+x.faces,0),50188);
 assert.equal(qa.findings.find(x=>x.candidate_patient_side==='left').cross_midline_leakage_review_required,true);
 assert.equal(qa.segmentation_anatomical_validation,'NOT_VALIDATED');
 assert.equal(qa.CT_linked_bronchoscopy,'BLOCKED');
 assert.equal(qa.clinical_use,false);
 assert.ok(rename.includes('np.array_equal(mesh.vertices, original.vertices)'));
 assert.ok(rename.includes('np.array_equal(mesh.faces, original.faces)'));
});
test('A2 voxel QC discloses CT-derived threshold circularity and source midline contamination',()=>{
 assert.equal(voxel.candidate_mesh_sha256,candidate.mesh_sha256);
 assert.equal(voxel.source_sha256,candidate.source_sha256);
 assert.ok(voxel.overlap_voxels>25000);
 assert.ok(voxel.overlap_volume_ml>15&&voxel.overlap_volume_ml<18);
 assert.equal(voxel.measurements.length,2);
 const left=voxel.measurements.find(x=>x.side==='left');
 assert.ok(left.volume_fraction_on_opposite_side>.01);
 assert.ok(left.inside_mesh_HU_threshold_fraction>.95);
 assert.equal(voxel.mask_approval,'FAILED_PENDING_CLINICAL_REVIEW');
 assert.match(voxel.threshold_method,/NOT independent ground truth|not independent ground truth/i);
 assert.match(maskAudit,/vtkPolyDataToImageStencil/);
 assert.match(maskAudit,/sha256/);
});
test('A2 inspector distinguishes source-integrity match from anatomical or clinical acceptance',()=>{
 assert.ok(inspector.includes('A2_CANDIDATE_SHA='));
 assert.ok(inspector.includes('a2CandidateReviewPending()'));
 assert.ok(inspector.includes('16.3 mL'));
 assert.ok(inspector.includes('A2 CANDIDATE · CLINICAL QA PENDING'));
 assert.ok(inspector.includes('CT-linked bronchoscopy'));
 assert.ok(!inspector.includes('enterBranch('),'No CT candidate may control bronchoscope');
 assert.ok(!engine.includes('A2_CANDIDATE_SHA'),'Experimental source surface must not override procedure engine automatically');
});
