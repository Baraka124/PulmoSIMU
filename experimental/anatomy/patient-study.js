(function studyModule(root){
'use strict';
const fail=(msg)=>{throw new Error(msg)};
const hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const matrix=m=>{
 if(!Array.isArray(m)||m.length!==4||m.some(r=>!Array.isArray(r)||r.length!==4||r.some(x=>!Number.isFinite(x)))||m[3].some((x,i)=>Math.abs(x-(i===3?1:0))>1e-10))fail('Expected finite affine 4×4 matrix');
 return m.map(r=>r.slice());
};
function inverse(a){
 const m=matrix(a).map((r,i)=>[...r,...Array.from({length:4},(_,j)=>+(i===j))]);
 for(let i=0;i<4;i++){
  let p=i;for(let j=i+1;j<4;j++)if(Math.abs(m[j][i])>Math.abs(m[p][i]))p=j;
  if(Math.abs(m[p][i])<1e-12)fail('Singular or ill-conditioned affine');
  [m[i],m[p]]=[m[p],m[i]];const v=m[i][i];m[i]=m[i].map(x=>x/v);
  for(let j=0;j<4;j++)if(j!==i){const f=m[j][i];m[j]=m[j].map((x,k)=>x-f*m[i][k]);}
 }
 return matrix(m.map(r=>r.slice(4)));
}
function apply(a,p){
 if(!Array.isArray(p)||p.length!==3||p.some(x=>!Number.isFinite(x)))fail('Invalid point');
 const q=matrix(a).slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);
 if(q.some(x=>!Number.isFinite(x)))fail('Nonfinite transformed point');return q;
}
function product(a,b){a=matrix(a);b=matrix(b);return matrix(a.map(r=>b[0].map((_,j)=>r.reduce((s,x,k)=>s+x*b[k][j],0))))}
const flip=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]];
function frame(a,dims){
 a=matrix(a);
 if(!Array.isArray(dims)||dims.length!==3||dims.some(n=>!Number.isSafeInteger(n)||n<2))fail('Invalid source dimensions');
 const inv=inverse(a),spacing=[0,1,2].map(j=>Math.hypot(...a.slice(0,3).map(r=>r[j])));
 if(spacing.some(x=>!Number.isFinite(x)||x<.001||x>100))fail('Implausible voxel spacing');
 let error=0;for(const i of [0,dims[0]-1])for(const j of [0,dims[1]-1])for(const k of [0,dims[2]-1]){
 const p=[i,j,k],q=apply(inv,apply(a,p));error=Math.max(error,Math.hypot(...p.map((x,n)=>x-q[n])));
 }
 if(error>1e-5)fail('Affine roundtrip exceeded tolerance');
 return {voxel_to_ras_mm:a,ras_mm_to_voxel:inv,voxel_to_lps_mm:product(flip,a),lps_mm_to_voxel:inverse(product(flip,a)),spacing_mm:spacing,roundtrip_max_vox:error};
}
function build({volume,ctSha,meshSha=null,qa=null,qaSha=null,headerMatched=false,meshLoaded=false}={}){
 if(!volume||!hash(ctSha))fail('Load a parsed CT and exact SHA-256 first');
 if(volume.spatialUnit!==2)fail('NIfTI spatial units must explicitly be millimetres');
 const spatial=frame(volume.affine,volume.dims);
 if(!Array.isArray(volume.spacing)||volume.spacing.some((x,i)=>!Number.isFinite(x)||Math.abs(x-spatial.spacing_mm[i])>Math.max(.001,.001*spatial.spacing_mm[i])))fail('NIfTI pixdim and affine disagree');
 if(!Number.isFinite(volume.slope)||!Number.isFinite(volume.intercept))fail('Invalid intensity scaling');
 let surface=null;
 if(meshLoaded&&headerMatched&&hash(meshSha)&&hash(qaSha)&&qa?.source_sha256===ctSha&&qa?.mesh_sha256===meshSha){
  const m=matrix(qa.model_from_ras_mm),mInv=inverse(m);
  surface={sha256:meshSha,reconstruction_manifest_sha256:qaSha,model_from_ras_mm:m,ras_mm_from_model:mInv,status:'FILE_AND_HEADER_MATCHED_UNREVIEWED'};
 }
 return {schema:'pulmosimu.study/1',source:{format:'NIFTI1_3D',sha256:ctSha,dimensions:volume.dims.slice(),sform_code:volume.sformCode,scalar_datatype:volume.datatype,stored_slope:volume.slope,stored_intercept:volume.intercept},spatial,surface,review:{segmentation:'NOT_VALIDATED',patient_specific_procedure:'BLOCKED',clinical_use:'NOT_AUTHORIZED'}};
}
function canonical(x){if(x===null||typeof x!=='object')return JSON.stringify(x);if(Array.isArray(x))return '['+x.map(canonical).join(',')+']';return '{'+Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+canonical(x[k])).join(',')+'}'}
function verify(imported,current){if(!imported||imported.schema!=='pulmosimu.study/1'||canonical(imported)!==canonical(current))fail('Study package does not match the currently loaded source and reconstruction');return true}
root.PulmoStudy=Object.freeze({frame,apply,inverse,build,verify});
})(typeof window!=='undefined'?window:globalThis);
