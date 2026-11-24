/* PulmoSIMU E2 · Source-verified CT multiplanar sections in the ORIGINAL XR engine.
 * Uses local NIfTI HU voxels; no inferred anatomy, clinical registration,
 * replicated procedure tree, or upload. Section planes share the source
 * lung's display transform but are NOT permitted as bronchoscopy geometry.
 */
(function(root){
'use strict';
const AXES=['axial','coronal','sagittal'],HASH=/^[0-9a-f]{64}$/i;
function gate(study){
 const {volume:v,manifest:m,ctSha,meshSha,meshVerified}=study||{};
 if(!v||!m||!HASH.test(ctSha||'')||!HASH.test(meshSha||'')||!meshVerified)
   return Object.freeze({ok:false,reason:'Local source volume, verified GLB and manifest required'});
 if(ctSha.toLowerCase()!==String(m.source_sha256||'').toLowerCase()||
    meshSha.toLowerCase()!==String(m.mesh_sha256||'').toLowerCase())
   return Object.freeze({ok:false,reason:'Study SHA-256 mismatch'});
 if(!root.PulmoNiftiSource?.matchQA)
   return Object.freeze({ok:false,reason:'NIfTI source adapter unavailable'});
 const match=root.PulmoNiftiSource.matchQA(v,m);
 if(!match.valid)return Object.freeze({ok:false,reason:match.reason});
 if(!Array.isArray(m.model_from_ras_mm)||m.model_from_ras_mm.length!==4||
    m.model_from_ras_mm.some(row=>!Array.isArray(row)||row.length!==4||row.some(x=>!Number.isFinite(x))))
   return Object.freeze({ok:false,reason:'Explicit CT-to-model transform required'});
 if(m.quality_gates?.laterality_labels==='CANDIDATE_CORRECTED_RAS_INDEPENDENT_REVIEW_PENDING')
   return Object.freeze({ok:false,reason:'Invalid historical RAS-swapped candidate'});
 return Object.freeze({ok:true,reason:'Source bytes and affine matched; tissue segmentation UNVALIDATED'});
}
function geometrySpec(volume,plane,index){
 if(!AXES.includes(plane))throw Error('Unknown CT section orientation');
 const [nx,ny,nz]=volume?.dims||[];
 if(![nx,ny,nz].every(n=>Number.isInteger(n)&&n>=2))throw Error('Invalid CT dimensions');
 const axis=AXES.indexOf(plane),dim=[nz,ny,nx][axis];
 const k=Math.max(0,Math.min(dim-1,Math.round(Number(index)||0)));
 let w,h,sample;
 if(plane==='axial'){w=nx;h=ny;sample=(u,v)=>[u,v,k];}
 else if(plane==='coronal'){w=nx;h=nz;sample=(u,v)=>[u,k,nz-1-v];}
 else{w=ny;h=nz;sample=(u,v)=>[k,u,nz-1-v];}
 return Object.freeze({plane,index:k,dim,w,h,sample,
  corners:[sample(0,0),sample(w-1,0),sample(w-1,h-1),sample(0,h-1)]});
}
function create(THREE,modelGroup,study){
 const accepted=gate(study);if(!accepted.ok)throw Error(accepted.reason);
 if(!THREE?.CanvasTexture||!THREE?.BufferGeometry||!modelGroup?.isObject3D)
   throw Error('Three.js source group is unavailable');
 const volume=study.volume,manifest=study.manifest;
 const group=new THREE.Group();group.name='xr-ct-verified-mpr';modelGroup.add(group);
 const cross=volume.dims.map(n=>Math.floor((n-1)/2));
 const entries=new Map(),colors={axial:0xc5d7db,coronal:0xb7c7cf,sagittal:0xd1c7b2};
 const asModel=ijk=>root.PulmoNiftiSource.getWorldFromRAS(volume.ras(...ijk),manifest);
 function render(plane){
   const axis={axial:2,coronal:1,sagittal:0}[plane];
   const spec=geometrySpec(volume,plane,cross[axis]);
   const old=entries.get(plane);
   if(old){group.remove(old.mesh);old.texture.dispose();old.geometry.dispose();old.material.dispose();}
   // The CT's original samples are displayed without anatomy synthesis.
   // Keep one sub-512 texture per orientation to protect Quest GPU memory.
   const rate=Math.max(1,Math.ceil(Math.max(spec.w,spec.h)/512));
   const w=Math.max(2,Math.ceil(spec.w/rate)),h=Math.max(2,Math.ceil(spec.h/rate));
   const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;
   const ctx=canvas.getContext('2d',{alpha:false});
   const img=ctx.createImageData(w,h);
   for(let y=0;y<h;y++)for(let x=0;x<w;x++){
     const ijk=spec.sample(Math.min(spec.w-1,Math.floor(x*spec.w/w)),
                           Math.min(spec.h-1,Math.floor(y*spec.h/h)));
     const hu=volume.voxel(...ijk);
     const gray=Math.round(Math.max(0,Math.min(255,255*(hu+1350)/1500)));
     const p=4*(x+w*y);img.data[p]=gray;img.data[p+1]=gray;
     img.data[p+2]=gray;img.data[p+3]=255;
   }
   ctx.putImageData(img,0,0);
   const texture=new THREE.CanvasTexture(canvas);
   texture.minFilter=THREE.LinearFilter;texture.magFilter=THREE.LinearFilter;
   texture.generateMipmaps=false;
   const geo=new THREE.BufferGeometry();
   const corners=spec.corners.map(asModel);
   geo.setAttribute('position',new THREE.Float32BufferAttribute(corners.flat(),3));
   geo.setAttribute('uv',new THREE.Float32BufferAttribute([0,1,1,1,1,0,0,0],2));
   geo.setIndex([0,1,2,0,2,3]);geo.computeVertexNormals();
   const material=new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,
      transparent:true,opacity:.82,depthWrite:false,polygonOffset:true,
      polygonOffsetFactor:-1});
   const mesh=new THREE.Mesh(geo,material);mesh.name='xr-source-'+plane+'-section';
   mesh.renderOrder=8;mesh.userData={researchOnly:true,plane,voxelIndex:spec.index};
   // Thin source-coordinate frame, never a physiological or model measurement.
   group.add(mesh);entries.set(plane,{mesh,texture,geometry:geo,material});
 }
 try{for(const p of AXES)render(p)}
 catch(error){dispose();throw error}
 function setIndex(plane,index){
   if(!AXES.includes(plane))return false;
   const axis={axial:2,coronal:1,sagittal:0}[plane];
   if(!Number.isFinite(index))return false;
   const next=Math.max(0,Math.min(volume.dims[axis]-1,Math.round(index)));
   // Avoid rebuilding a GPU texture on every inspector camera or UI update.
   if(next===cross[axis])return snapshot();
   cross[axis]=next;
   render(plane);return snapshot();
 }
 function step(plane,delta=1){
   if(!AXES.includes(plane))return false;
   const axis={axial:2,coronal:1,sagittal:0}[plane];
   const d=volume.dims[axis];
   return setIndex(plane,((cross[axis]+Math.round(delta))%d+d)%d);
 }
 function setVisible(value){group.visible=Boolean(value);return group.visible}
 function snapshot(){
  return Object.freeze({source:'verified-local-NIfTI-1',status:'RESEARCH_ONLY_SEGMENTATION_UNVALIDATED',
   sourceSha256:study.ctSha,meshSha256:study.meshSha,
   sections:Object.freeze({axial:cross[2],coronal:cross[1],sagittal:cross[0]}),
   crossRASmm:volume.ras(...cross),visible:group.visible,shadedCT:true});
 }
 function dispose(){
   for(const item of entries.values()){
     group.remove(item.mesh);item.geometry.dispose();item.material.dispose();item.texture.dispose();
   }entries.clear();modelGroup.remove(group);
 }
 group.visible=false;
 return Object.freeze({setIndex,step,setVisible,snapshot,dispose,group});
}
root.PulmoXRCTSpatial=Object.freeze({gate,geometrySpec,create});
})(typeof window!=='undefined'?window:globalThis);
