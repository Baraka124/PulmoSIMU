import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const root=new URL('../data/nlst-a1/',import.meta.url);
const a31=JSON.parse(readFileSync(new URL('a31-study-manifest.json',root)));
const qa=JSON.parse(readFileSync(new URL('a31-source-QA.json',root)));
const structure=JSON.parse(readFileSync(new URL('a32-a34-source-mask-QA.json',root)));
const script=readFileSync(new URL('../anatomy/a31_disjoint_airspace.py',import.meta.url),'utf8');
const gateSource=readFileSync(new URL('../anatomy/a4-source-gate.js',import.meta.url),'utf8');
const inspector=readFileSync(new URL('../anatomy/inspector-app.js',import.meta.url),'utf8');
const html=readFileSync(new URL('../anatomy-inspector.html',import.meta.url),'utf8');
const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
function gate(args){const window={};runInNewContext(gateSource,{window});
 return window.PulmoSourceAnatomyGate.assess(args)}
test('A3.1 research binary and original CT have separate SHA-256 identities',()=>{
 assert.equal(qa.input_ct_sha256,a31.source_sha256);
 assert.equal(qa.derived_glb_sha256,a31.mesh_sha256);
 assert.equal(qa.derived_labelmap_sha256,a31.source_labelmap_sha256);
 assert.equal(qa.source_overlap_voxels,27702);
 assert.equal(qa.output_overlap_voxels,0);
 assert.equal(qa.voxel_count_preserved_across_bilateral_union,true);
 assert.equal(qa.surfaces.left.mesh_watertight,true);
 assert.equal(qa.surfaces.right.mesh_watertight,true);
 assert.equal(qa.surfaces.left.candidate_faces+qa.surfaces.right.candidate_faces,190846);
 assert.equal(qa.clinical_anatomical_validation,'NOT_VALIDATED');
 assert.equal(a31.quality_gates.ct_linked_bronchoscopy,'BLOCKED');
 assert.match(script,/PreserveTopologyOn/);
 assert.match(script,/sha256\(ct\)/);
 assert.match(script,/voxelize\(/);
 assert.ok(!script.includes('return True # clinical'));
});
test('A3.2-A3.4 require actual sourced five lobes, airway lumen and vessels',()=>{
 assert.equal(structure.CT_SHA256,a31.source_sha256);
 assert.equal(structure.A31_labelmap_SHA256,a31.source_labelmap_sha256);
 for(const group of Object.values(structure.groups)){
  assert.equal(group.complete,false);
  assert.equal(group.status,'MISSING_SOURCE_MASKS');
 }
 assert.equal(structure.ct_registered_bronchoscopy,'BLOCKED_NO_REVIEWED_AIRWAY_CENTRELINE');
 assert.equal(structure.clinical_segmentation_approval,'PENDING_INDEPENDENT_REVIEW');
});
test('A4 never equates file-hash/source-coframing match with CT-specific bronchoscopy',()=>{
 const matched=gate({manifest:a31,ctSha:a31.source_sha256,meshSha:a31.mesh_sha256,structureQA:structure});
 assert.equal(matched.researchAnatomyInspectionAllowed,true);
 assert.equal(matched.clinicalAnatomyValidated,false);
 assert.equal(matched.CTLinkedBronchoscopyAllowed,false);
 assert.equal(matched.researchStageStatus[0].id,'A3.1');
 assert.equal(matched.researchStageStatus[4].status,'CT_PROCEDURE_BLOCKED');
 const forged=gate({manifest:a31,ctSha:'wrong-hash',meshSha:a31.mesh_sha256,structureQA:structure,
    clinicianReview:{status:'APPROVED_FOR_RESEARCH_PROCEDURE',reviewer:'Test',
      source_sha256:a31.source_sha256,mesh_sha256:a31.mesh_sha256,signed_report_sha256:'f'.repeat(64)},
    sourceAirway:{centrelineCoframed:true,topologyReviewed:true,lumenWallValidated:true,
      ct_source_sha256:a31.source_sha256},
    instrument:{sourceDimensionReviewed:true,calibratedDiameterMm:3,collisionModelReviewed:true}});
 assert.equal(forged.CTLinkedBronchoscopyAllowed,false);
});
test('Source inspector exposes source-only opacity, component isolation, MPR clipping and QA review',()=>{
 for(const value of ['id="surface-opacity"','id="section-plane"','id="structure-qa-file"',
   'id="source-gates"','a4-source-gate.js'])assert.ok(html.includes(value),value);
 for(const value of ['applySurfaceOpacity','mesh.userData.pulmoSurfaceIndex=i',
    "if(choice.startsWith('mesh:'))","const axis={sagittal:0,coronal:1,axial:2}",
    "state.structureQA=data","updateResearchGates()"])assert.ok(inspector.includes(value),value);
 assert.ok(!inspector.includes('scene.add(new THREE.CatmullRomCurve3'),'No invented bronchi');
});

test('A4 prevents the original authored bronchoscope from being misrepresented as CT registered',()=>{
 assert.ok(main.includes('a4-source-gate.js?v=A4-CT-PROCEDURE-GATE'));
 assert.ok(xr.includes("if(next==='procedure'&&getMeshMode()!=='reference')"));
 assert.ok(xr.includes('CT-SPECIFIC BRONCHOSCOPY BLOCKED / NO REGISTERED AIRWAY'));
 assert.ok(xr.includes("A4SourceProcedureStatus:getMeshMode()==='patient'"));
 assert.ok(xr.includes("(a.id!=='procedure'||role==='reference')"));
});

test('A4 forbids patient-linked procedure even when untrusted browser review metadata claims approval',()=>{
 const groups={
  'A3.2 - lobes':{complete:true},'A3.3 - bronchial lumen':{complete:true},
  'A3.4 - vessels':{complete:true}
 };
 const fakeQA={CT_SHA256:a31.source_sha256,
  A31_labelmap_SHA256:a31.source_labelmap_sha256,
  groups,lobar_topology_gate:'NONOVERLAPPING_UNREVIEWED'};
 const review={status:'APPROVED_FOR_RESEARCH_PROCEDURE',reviewer:'Example Clinician',
   source_sha256:a31.source_sha256,mesh_sha256:a31.mesh_sha256,
   signed_report_sha256:'f'.repeat(64)};
 const result=gate({
  manifest:a31,ctSha:a31.source_sha256,meshSha:a31.mesh_sha256,structureQA:fakeQA,
  clinicianReview:review,
  sourceAirway:{centrelineCoframed:true,topologyReviewed:true,lumenWallValidated:true,
   ct_source_sha256:a31.source_sha256},
  instrument:{sourceDimensionReviewed:true,calibratedDiameterMm:3,collisionModelReviewed:true}
 });
 assert.equal(result.researchProcedureCandidateEligible,true);
 assert.equal(result.CTLinkedBronchoscopyAllowed,false,
  'User-provided strings cannot prove independent clinical review or hardware acceptance');
 assert.equal(result.clinicalAnatomyValidated,false);
});
