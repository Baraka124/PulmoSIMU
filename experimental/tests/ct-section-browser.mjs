/* Regression for source inspector CT-plane blanking bug.
 * Synthetic bytes only; never uploads private NLST data or makes clinical claims.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';

const host=process.env.PULMOSIMU_TEST_ORIGIN||'http://127.0.0.1:8765';
const sha=b=>createHash('sha256').update(b).digest('hex');
function syntheticNifti(){
 const n=6,buf=Buffer.alloc(352+n*n*n);
 buf.writeInt32LE(348,0);buf.writeInt16LE(3,40);
 for(let a=0;a<3;a++){buf.writeInt16LE(n,42+2*a);buf.writeFloatLE(1,80+4*a)}
 buf.writeInt16LE(2,70);buf.writeInt16LE(8,72);buf.writeFloatLE(352,108);
 buf.writeFloatLE(1,112);buf.writeFloatLE(-1024,116);
 buf.writeInt16LE(1,254);
 const rows=[[1,0,0,-3],[0,1,0,-3],[0,0,1,-3]];
 rows.forEach((row,i)=>row.forEach((value,j)=>buf.writeFloatLE(value,280+16*i+4*j)));
 for(let k=0;k<n;k++)for(let j=0;j<n;j++)for(let i=0;i<n;i++)
   buf[352+i+n*(j+n*k)]=(i*19+j*23+k*29)%256;
 return buf;
}
function syntheticGLB(){
 const positions=[-.12,-.12,-.12,.12,-.12,-.12,0,.12,-.12,0,0,.12];
 const vertices=Buffer.alloc(positions.length*4);
 positions.forEach((v,i)=>vertices.writeFloatLE(v,i*4));
 const faces=[0,1,2,0,1,3,0,2,3,1,2,3];
 const indices=Buffer.alloc(faces.length*2);
 faces.forEach((v,i)=>indices.writeUInt16LE(v,i*2));
 const bin=Buffer.concat([vertices,indices]);
 const json={
  asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],
  nodes:[{name:'test-source-surface',mesh:0}],
  meshes:[{primitives:[{attributes:{POSITION:0},indices:1,material:0}]}],
  materials:[{pbrMetallicRoughness:{baseColorFactor:[.65,.73,.79,1],roughnessFactor:1}}],
  buffers:[{byteLength:bin.length}],
  bufferViews:[{buffer:0,byteOffset:0,byteLength:vertices.length,target:34962},
    {buffer:0,byteOffset:vertices.length,byteLength:indices.length,target:34963}],
  accessors:[
    {bufferView:0,componentType:5126,count:4,type:'VEC3',
      min:[-.12,-.12,-.12],max:[.12,.12,.12]},
    {bufferView:1,componentType:5123,count:12,type:'SCALAR',min:[0],max:[3]}]
 };
 const bytes=Buffer.from(JSON.stringify(json),'utf8');
 const padded=Buffer.concat([bytes,Buffer.alloc((4-bytes.length%4)%4,0x20)]);
 const glb=Buffer.alloc(12+8+padded.length+8+bin.length);
 glb.writeUInt32LE(0x46546c67,0);glb.writeUInt32LE(2,4);glb.writeUInt32LE(glb.length,8);
 glb.writeUInt32LE(padded.length,12);glb.writeUInt32LE(0x4e4f534a,16);
 padded.copy(glb,20);
 const p=20+padded.length;
 glb.writeUInt32LE(bin.length,p);glb.writeUInt32LE(0x004e4942,p+4);bin.copy(glb,p+8);
 return glb;
}
const nii=syntheticNifti(),glb=syntheticGLB();
const affine=[[1,0,0,-3],[0,1,0,-3],[0,0,1,-3],[0,0,0,1]];
const manifest={
 source_shape_vox:[6,6,6],sform_RAS_mm:affine,spacing_mm:[1,1,1],
 source_sha256:sha(nii),mesh_sha256:sha(glb),
 HU_stored_slope:1,HU_stored_intercept:-1024,
 model_from_ras_mm:[[.05,0,0,0],[0,.05,0,0],[0,0,.05,0],[0,0,0,1]],
 geometry_class:'PRELIMINARY_CT_DERIVED_AIRSPACE_MASK',
 validation:'RESEARCH_ONLY_UNVALIDATED'
};
const browser=await chromium.launch({headless:true,args:['--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:950}});
const pageErrors=[];page.on('pageerror',e=>pageErrors.push(e.message));
try{
 await page.goto(host+'/experimental/anatomy-inspector.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>window.PulmoAnatomyInspector?.status && window.PulmoXRCTSpatial?.create,
   null,{timeout:45000});
 await page.locator('#manifest-file').setInputFiles({
   name:'synthetic-source.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(manifest))});
 await page.waitForFunction(()=>window.PulmoAnatomyInspector?.status?.hasQA);
 await page.locator('#mesh-file').setInputFiles({
   name:'synthetic-test-lung.glb',mimeType:'model/gltf-binary',buffer:glb});
 await page.waitForFunction(()=>window.PulmoAnatomyInspector?.status?.hasGLB);
 await page.locator('#ct-file').setInputFiles({
   name:'synthetic-test-ct.nii',mimeType:'application/octet-stream',buffer:nii});
 await page.waitForFunction(()=>window.PulmoAnatomyInspector?.status?.sourceMatched,{},{timeout:45000});
 assert.equal(await page.locator('#clip-toggle').isDisabled(),false);
 let before=await page.evaluate(()=>window.PulmoAnatomyInspector.status);
 assert.equal(before.sourceMeshesVisible,1);
 assert.equal(before.sourceMaterialClippingActive,false);
 await page.locator('#clip-toggle').check();
 await page.waitForFunction(()=>window.PulmoAnatomyInspector.status.sourceSectionVisible);
 let active=await page.evaluate(()=>window.PulmoAnatomyInspector.status);
 assert.equal(active.sourceSectionMode,'source-voxel-plane');
 assert.equal(active.sourceMeshesVisible,1,'Enabling the CT section must not blank the original mesh');
 assert.equal(active.sourceMaterialClippingActive,false,'Never use destructive shader material clipping');
 await page.locator('#section-plane').selectOption('coronal');
 active=await page.evaluate(()=>window.PulmoAnatomyInspector.status);
 assert.equal(active.sectionPlane,'coronal');
 assert.equal(active.sourceSectionVisible,true);
 assert.equal(active.sourceMeshesVisible,1);
 await page.locator('#clip-toggle').uncheck();
 active=await page.evaluate(()=>window.PulmoAnatomyInspector.status);
 assert.equal(active.sourceSectionVisible,false);
 assert.equal(active.sourceMeshesVisible,1);
 await page.locator('#clip-toggle').check();
 active=await page.evaluate(()=>window.PulmoAnatomyInspector.status);
 assert.equal(active.sourceSectionVisible,true);
 assert.equal(active.sourceMeshesVisible,1);
 assert.deepEqual(pageErrors,[],'No WebGL/JavaScript uncaught errors');
 console.log('PASS source inspector: matching synthetic CT+GLB+manifest; 3D voxel plane ON/OFF/coronal; original mesh remains visible');
}finally{await page.close();await browser.close();}
