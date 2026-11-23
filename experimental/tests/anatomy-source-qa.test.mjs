import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

const script=readFileSync(new URL('../anatomy/nifti-source.js',import.meta.url),'utf8');
const app=readFileSync(new URL('../anatomy/inspector-app.js',import.meta.url),'utf8');
const html=readFileSync(new URL('../anatomy-inspector.html',import.meta.url),'utf8');
const studioPath=new URL('../../studio.html',import.meta.url);
const studio=existsSync(studioPath)?readFileSync(studioPath,'utf8'):null;
const sandbox={window:{}};
runInNewContext(script,sandbox);
const {readNifti,matchQA,getWorldFromRAS}=sandbox.window.PulmoNiftiSource;
function sample(){
 const ab=new ArrayBuffer(352+2*8),d=new DataView(ab);
 d.setInt32(0,348,true);d.setInt16(40,3,true);
 for(let a=0;a<3;a++)d.setInt16(42+a*2,2,true);
 d.setInt16(70,512,true);d.setInt16(72,16,true);
 d.setFloat32(80,1,true);d.setFloat32(84,1,true);d.setFloat32(88,2,true);
 d.setFloat32(108,352,true);d.setFloat32(112,1,true);d.setFloat32(116,-1024,true);
 d.setInt16(254,1,true);
 const affine=[[1,0,0,-100],[0,-1,0,130],[0,0,2,-400]];
 for(let i=0;i<3;i++)for(let j=0;j<4;j++)d.setFloat32(280+i*16+j*4,affine[i][j],true);
 for(let n=0;n<8;n++)d.setUint16(352+n*2,1000+n,true);
 return ab;
}
test('A1 NIfTI reader interprets real little-endian 3D voxel layout and explicit sform',()=>{
 const v=readNifti(sample());
 assert.deepEqual(Array.from(v.dims),[2,2,2]);
 assert.equal(v.voxel(0,0,0),-24);
 assert.equal(v.voxel(1,0,0),-23);
 assert.equal(v.voxel(0,1,0),-22);
 assert.equal(v.voxel(0,0,1),-20);
 assert.deepEqual(Array.from(v.ras(1,1,1)),[-99,129,-398]);
 const qa={source_shape_vox:[2,2,2],sform_RAS_mm:[[1,0,0,-100],[0,-1,0,130],[0,0,2,-400],[0,0,0,1]],
  HU_stored_slope:1,HU_stored_intercept:-1024,
  model_from_ras_mm:[[.001,0,0,0],[0,0,.001,.39],[0,-.001,0,.122],[0,0,0,1]]};
 assert.equal(matchQA(v,qa).valid,true);
 const world=getWorldFromRAS(v.ras(1,1,1),qa);
 for(const [i,expected]of [-.099,-.008,-.007].entries())assert.ok(Math.abs(world[i]-expected)<1e-9);
 qa.sform_RAS_mm[0][3]+=3;
 assert.equal(matchQA(v,qa).valid,false);
});
test('A1 refuses ambiguous source registration and 4D volumes',()=>{
 const source=sample(),view=new DataView(source);
 view.setInt16(254,0,true);
 assert.throws(()=>readNifti(source),/sform/);
 view.setInt16(254,1,true);view.setInt16(40,4,true);
 assert.throws(()=>readNifti(source),/single 3D/);
});
test('A1 research viewer keeps local CT and GLB data separate from the original airway procedure',()=>{
 for(const x of ['id="ct-file"','id="mesh-file"','id="manifest-file"',
 'id="mpr-axial"','id="mpr-coronal"','id="mpr-sagittal"','id="clip-toggle"',
 'id="cross-3d"','id="qa-details"'])assert.ok(html.includes(x),x);
 assert.ok(html.includes('RESEARCH'));
 assert.ok(app.includes("crypto.subtle.digest('SHA-256'"));
 assert.ok(app.includes('state.ctSha===state.qa.source_sha256'));
 assert.ok(app.includes('state.meshSha===state.qa.mesh_sha256'));
 assert.ok(app.includes('matchQA(state.volume,state.qa)'));
 assert.ok(!app.includes('fetch('),'No uploading or remote loading user imaging');
 assert.ok(!app.includes('localStorage'),'No persistent CT data storage');
 assert.ok(!app.includes('indexedDB'),'No persistent CT data storage');
 if(studio){assert.ok(studio.includes('modeSource'));
 assert.ok(studio.includes("src:'./experimental/anatomy-inspector.html'"));}
 assert.ok(!app.includes('enterBranch('),'CT mask cannot enter original bronchoscopy without validated lumen');
});