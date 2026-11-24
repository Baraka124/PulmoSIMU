import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const base=new URL('../',import.meta.url);
const source=path=>readFileSync(new URL(path,base),'utf8');
const root={};
for(const path of ['anatomy/nifti-source.js','xr/ct-spatial-mpr.js',
                   'xr/science-integrity.js','xr/acceptance-gates.js'])
 runInNewContext(source(path),{window:root});
const affine=[[1,0,0,-4],[0,-1,0,2],[0,0,2,-20],[0,0,0,1]];
const volume={
 dims:[4,3,2],affine,spacing:[1,1,2],slope:1,intercept:-1024,
 ras(i,j,k){return [i-4,2-j,2*k-20]},
 voxel(i,j,k){return 120*i+20*j+10*k-1024}
};
const manifest={
 source_shape_vox:[4,3,2],sform_RAS_mm:affine,
 HU_stored_slope:1,HU_stored_intercept:-1024,
 source_sha256:'a'.repeat(64),mesh_sha256:'b'.repeat(64),
 model_from_ras_mm:[[.001,0,0,0],[0,0,.001,.39],[0,-.001,0,.12],[0,0,0,1]]
};
const study={volume,manifest,ctSha:'a'.repeat(64),meshSha:'b'.repeat(64),meshVerified:true};
test('E2 real MPR is source-hash and sform gated; cannot attach unrelated CT',()=>{
 assert.equal(root.PulmoXRCTSpatial.gate(study).ok,true);
 assert.equal(root.PulmoXRCTSpatial.gate({...study,ctSha:'c'.repeat(64)}).ok,false);
 assert.equal(root.PulmoXRCTSpatial.gate({...study,meshVerified:false}).ok,false);
 assert.equal(root.PulmoXRCTSpatial.gate({...study,volume:{...volume,affine:[[0,0,0,0]]}}).ok,false);
});
test('E2 axial, coronal and sagittal sections map to actual source voxels',()=>{
 const a=root.PulmoXRCTSpatial.geometrySpec(volume,'axial',1);
 assert.deepEqual(Array.from(a.sample(3,2)),[3,2,1]);
 const c=root.PulmoXRCTSpatial.geometrySpec(volume,'coronal',2);
 assert.deepEqual(Array.from(c.sample(3,0)),[3,2,1]);
 const s=root.PulmoXRCTSpatial.geometrySpec(volume,'sagittal',3);
 assert.deepEqual(Array.from(s.sample(2,1)),[3,2,0]);
 assert.throws(()=>root.PulmoXRCTSpatial.geometrySpec(volume,'fake',0),/Unknown/);
});
test('E3 scientific model domain never equates CT MPR with patient physics or procedure',()=>{
 const ref=root.PulmoXRScience.classify({modelRole:'reference',referenceSolverReady:true});
 assert.equal(ref.respiratoryPhysics,'R3_ASSUMED_REFERENCE_ONLY');
 assert.equal(ref.bronchoscopy,'ORIGINAL_REFERENCE_SIMULATOR_ONLY');
 const ct=root.PulmoXRScience.classify({modelRole:'patient',ctStudy:study,referenceSolverReady:true});
 assert.equal(ct.imaging,'SOURCE_MATCHED_RESEARCH');
 assert.equal(ct.respiratoryPhysics,'NOT_ATTACHED');
 assert.equal(ct.bronchoscopy,'CT_SPECIFIC_PROCEDURE_BLOCKED');
 assert.equal(ct.patientSpecificPressureFlow,false);
 assert.equal(root.PulmoXRScience.motionAcceptance(null).ok,false);
 assert.equal(root.PulmoXRScience.measurementMM([1,2,3],[4,6,3]),5);
 assert.equal(root.PulmoXRScience.canCoupleToCT({
  ctSha256:'a'.repeat(64),geometrySha256:'b'.repeat(64),
  solverSourceSha256:'c'.repeat(64),measuredBoundaryConditions:true,
  independentValidation:true}),false);
});
test('E1+E2 canonical XR preserves source selection and implements orientation lock',()=>{
 const page=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
 const xr=source('xr/procedural-immersive.js');
 for(const k of ['getCTStudy:()=>meshMode', 'loadResearchCTVolume(this)',
     'ct-spatial-mpr.js?v=E2-SOURCE-MPR','science-integrity.js?v=E3-SOURCE-SCIENCE',
     'acceptance-gates.js?v=E4-EVIDENCE', 'window.PulmoNiftiSource.readNifti(bytes)',
     "patientData.ctVolume=null;patientData.ctVolumeSha=null;"])
   assert.ok(page.includes(k),'Missing original-workstation E2 integration: '+k);
 for(const k of ["orientationLocked=axisMode==='locked'",
    'mGrp.quaternion.copy(orientationAnchor)',
    "['yaw','pitch','roll','locked']",
    "ctSpatial=global.PulmoXRCTSpatial.create(THREE,mGrp,study)",
    "r.intersectObject(ctSpatial.group,true)",
    "ctSpatial.dispose();ctSpatial=null;",
    "getMeshMode()==='reference'&&global.PulmoXRReferencePhysics?.step",
    'scienceIntegrity:scienceSnapshot()',
    'E4Acceptance:global.PulmoXRAcceptance?.assess'])
   assert.ok(xr.includes(k),'Missing original-engine XR invariant: '+k);
});
test('E4 evidence is conservative; no automatic clinical or Quest freeze',()=>{
 const a=root.PulmoXRAcceptance.assess({});
 assert.equal(a.researchFreezeEligible,false);
 assert.equal(a.gpuFrameTime,'NOT_MEASURED');
 assert.equal(a.headsetPhysicalAcceptance,'PENDING_EXPLICIT_QUEST_2_RUN');
 assert.equal(Array.from(root.PulmoXRAcceptance.gates).length,4);
});
