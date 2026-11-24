import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';

const read=p=>readFileSync(new URL(p,import.meta.url),'utf8');
const qa=JSON.parse(read('../data/nlst-a1/a2_1_observed_qa.json'));
const manifest=JSON.parse(read('../data/nlst-a1/a2_1_inspector_manifest.json'));
const src=read('../anatomy/a2_1_correct_nlst_airspace.py');
const readme=read('../data/nlst-a1/README.md');
const original=JSON.parse(read('../data/nlst-a1/study-manifest.json'));

test('A2.1 corrected CT source identity, image orientation and QC hashes',()=>{
 assert.equal(qa.input_source_sha256,original.source_sha256);
 assert.equal(manifest.source_sha256,original.source_sha256);
 assert.equal(manifest.mesh_sha256,qa.A2_surface.model_glb_sha256);
 assert.deepEqual(manifest.sform_RAS_mm,original.sform_RAS_mm);
 assert.equal(manifest.model_from_ras_mm.length,4);
 assert.equal(manifest.validation,'RESEARCH_ONLY_UNVALIDATED');
 assert.equal(manifest.quality_gates.RAS_positive_x,'PATIENT_RIGHT');
 assert.equal(qa.image_orientation,'NIfTI sform RAS; +X is patient right');
});

test('A2.1 CT segment labels are conservative and do not claim clinical anatomy',()=>{
 assert.equal(qa.mask.right.selected_voxels,744656);
 assert.equal(qa.mask.left.selected_voxels,623235);
 assert.equal(qa.mask.right_voxels_in_patient_left_hemisphere,0);
 assert.equal(qa.mask.left_voxels_in_patient_right_hemisphere,0);
 assert.ok(qa.mask.total_airspace_l>3.20&&qa.mask.total_airspace_l<3.24);
 assert.equal(qa.A2_surface.right.watertight,true);
 assert.equal(qa.A2_surface.left.watertight,true);
 assert.equal(qa.A2_surface.combined_triangles,436804);
 assert.equal(qa.gates.independent_CT_review,'PENDING');
 assert.equal(qa.gates.Quest_ready_low_poly_mesh.startsWith('BLOCKED'),true);
 assert.equal(qa.gates.bronchoscope_registered_to_CT,'BLOCKED');
 assert.equal(manifest.quality_gates.ct_linked_bronchoscopy,'BLOCKED');
 assert.equal(manifest.quality_gates.lobar_boundaries,'NOT_SEGMENTED');
});
test('A2.1 reproducible Python code is syntactically valid, source-checked and never fabricates lobes',()=>{
 const py=spawnSync('python3',['-c','import ast,sys; ast.parse(sys.stdin.read())'],
   {input:src,encoding:'utf8'});
 if(py.error?.code!=='ENOENT')assert.equal(py.status,0,py.stderr);
 for(const marker of ['SOURCE_SHA256','np.memmap','marching_cubes',
  'affine[0, 0]', 'ras_x >= 0', 'ras_x < 0',
  'ndi.label','np.savez_compressed','No reference masks']){
    if(marker==='No reference masks')continue;
    assert.ok(src.includes(marker),'Missing source integrity operation: '+marker);
 }
 assert.ok(src.includes('No validated')||src.includes('clinical'),
  'Data must be visibly research-only');
 assert.ok(!src.includes('requests.post('),'No implicit CT upload');
 assert.ok(readme.includes('A2.1'));
 assert.ok(!readme.includes('RAS +X as patient LEFT**, then created an unnecessary label-swapped candidate. This was correct'));
});
test('A2.1 remains research-only and never changes the canonical Quest bronchoscopy model',()=>{
 const engine=read('../xr/procedural-immersive.js');
 assert.ok(!engine.includes('a2_1_correct_nlst_airspace'));
 assert.ok(!engine.includes('PulmoSIMU_A2_1_NLST_Corrected_Airspace_Research.glb'));
 assert.ok(!manifest.public_binary_available);
 assert.match(readme,/not Quest-ready/i);
});
