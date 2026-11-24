/* PulmoSIMU A2.3 - local, source-bound CT review fiducials.
 * These marks are manual observations, NOT segmented lobes, fissure detections,
 * measurements, approved labels or authorization for CT-specific procedures.
 * Nothing is transmitted or stored automatically.
 */
(function(root){
'use strict';
const SCHEMA='pulmosimu-ct-point-review-v1';
const REGIONS=Object.freeze(['unassigned','RUL','RML','RLL','LUL','LLL']);
const FINDINGS=Object.freeze(['fissure-proposal','boundary-concern','source-artifact','other']);
const MAX_MARKS=500;
const digest=v=>typeof v==='string'&&/^[0-9a-f]{64}$/.test(v);
const finite=v=>typeof v==='number'&&Number.isFinite(v);
function matrixOK(m){
 return Array.isArray(m)&&m.length===4&&m.every(row=>Array.isArray(row)&&
  row.length===4&&row.every(finite))&&
  m[3].every((v,i)=>Math.abs(v-[0,0,0,1][i])<1e-7);
}
function identity({ctSha,meshSha,qa,headerValid}={}){
 if(headerValid!==true||!qa||!digest(ctSha)||!digest(meshSha)||
   qa.source_sha256!==ctSha||qa.mesh_sha256!==meshSha||
   !Array.isArray(qa.source_shape_vox)||qa.source_shape_vox.length!==3||
   !qa.source_shape_vox.every(n=>Number.isInteger(n)&&n>1)||
   !matrixOK(qa.sform_RAS_mm)||!matrixOK(qa.model_from_ras_mm))return null;
 return Object.freeze({
   source_sha256:ctSha,mesh_sha256:meshSha,
   source_shape_vox:qa.source_shape_vox.slice(),
   sform_RAS_mm:qa.sform_RAS_mm.map(row=>row.slice())
 });
}
function rasOf(voxel,study){
 const m=study.sform_RAS_mm;
 return m.slice(0,3).map(row=>
  Math.round((row[0]*voxel[0]+row[1]*voxel[1]+row[2]*voxel[2]+row[3])*1000)/1000);
}
function checkMark(raw,study){
 if(!study||!raw||typeof raw!=='object')throw Error('A matched CT study is required');
 if(!Array.isArray(raw.voxel)||raw.voxel.length!==3||
  raw.voxel.some((n,i)=>!Number.isInteger(n)||n<0||n>=study.source_shape_vox[i]))
  throw Error('Review voxel lies outside this source CT');
 if(!REGIONS.includes(raw.region)||!FINDINGS.includes(raw.finding))
  throw Error('Unknown review region or finding');
 const note=raw.note===undefined?'':raw.note;
 if(typeof note!=='string'||note.length>280||/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(note))
  throw Error('Invalid review note');
 if(typeof raw.id!=='string'||!/^[a-zA-Z0-9-]{3,80}$/.test(raw.id))
  throw Error('Invalid review marker identifier');
 const ras=rasOf(raw.voxel,study);
 if(raw.ras_mm!==undefined&&
   (!Array.isArray(raw.ras_mm)||raw.ras_mm.length!==3||
    raw.ras_mm.some((n,i)=>!finite(n)||Math.abs(n-ras[i])>0.002)))
  throw Error('Review RAS coordinates do not match the original CT sform');
 return Object.freeze({id:raw.id,voxel:raw.voxel.slice(),ras_mm:ras,
  region:raw.region,finding:raw.finding,note:note.trim()});
}
function newMark(study,voxel,region,finding,note='',id){
 const markerId=id||('m-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2,12));
 return checkMark({id:markerId,voxel,region,finding,note},study);
}
function sameStudy(a,b){
 if(!a||!b||a.source_sha256!==b.source_sha256||
  a.mesh_sha256!==b.mesh_sha256||
  !Array.isArray(a.source_shape_vox)||!Array.isArray(a.sform_RAS_mm))return false;
 if(a.source_shape_vox.length!==3||
  a.source_shape_vox.some((n,i)=>n!==b.source_shape_vox[i]))return false;
 return a.sform_RAS_mm.length===4&&a.sform_RAS_mm.every((row,i)=>
  Array.isArray(row)&&row.length===4&&row.every((n,j)=>finite(n)&&
   Math.abs(n-b.sform_RAS_mm[i][j])<=0.0001));
}
function exportReview(study,marks){
 if(!study||!Array.isArray(marks)||marks.length>MAX_MARKS)
  throw Error('A matched study and bounded annotation list are required');
 const safe=marks.map(m=>checkMark(m,study));
 if(new Set(safe.map(x=>x.id)).size!==safe.length)throw Error('Duplicate review marker');
 return {schema:SCHEMA,source_type:'manual-point-review',
  interpretation:'NOT CLINICAL VALIDATION OR SEGMENTATION',
  study,annotations:safe};
}
function importReview(bundle,study){
 if(!bundle||bundle.schema!==SCHEMA||bundle.source_type!=='manual-point-review'||
  !sameStudy(bundle.study,study)||!Array.isArray(bundle.annotations)||
  bundle.annotations.length>MAX_MARKS)
  throw Error('Review file is invalid or bound to a different exact CT/GLB study');
 const marks=bundle.annotations.map(m=>checkMark(m,study));
 if(new Set(marks.map(m=>m.id)).size!==marks.length)
  throw Error('Review file contains duplicate markers');
 return marks;
}
root.PulmoCTReview=Object.freeze({SCHEMA,REGIONS,FINDINGS,MAX_MARKS,
 identity,rasOf,newMark,exportReview,importReview,sameStudy});
})(typeof window!=='undefined'?window:globalThis);
