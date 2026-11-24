/* PulmoSIMU A1 — local, browser-only NIfTI-1 inspection.
   Supports uncompressed single-file 3D .nii, common scalar dtypes, NIfTI sform.
   No network use. Deliberately rejects geometry alignment without sform.
*/
(function(root){
'use strict';
function finite(v){return Number.isFinite(v)}
function assert(ok,message){if(!ok)throw new Error(message)}
function readNifti(buffer){
 assert(Object.prototype.toString.call(buffer)==='[object ArrayBuffer]',
  'Choose an uncompressed .nii file. Compressed .nii.gz is not supported.');
 assert(buffer.byteLength>=352,'NIfTI source file is truncated');
 const dv=new DataView(buffer);
 let little=true;
 if(dv.getInt32(0,true)===348)little=true;
 else if(dv.getInt32(0,false)===348)little=false;
 else throw new Error('Not a NIfTI-1 file (header size != 348).');
 const dimCount=dv.getInt16(40,little);
 assert(dimCount===3,'A single 3D NIfTI volume is required (4D/5D is not supported).');
 const dims=[42,44,46].map(off=>dv.getInt16(off,little));
 assert(dims.every(x=>x>=2&&x<=2048),'Invalid NIfTI dimensions.');
 const n=dims[0]*dims[1]*dims[2];
 assert(n<=250_000_000,'NIfTI volume exceeds local research viewer capacity.');
 const datatype=dv.getInt16(70,little),bitpix=dv.getInt16(72,little);
 const supported={2:[1,'getUint8'],4:[2,'getInt16'],8:[4,'getInt32'],16:[4,'getFloat32'],64:[8,'getFloat64'],512:[2,'getUint16'],768:[4,'getUint32']};
 const info=supported[datatype];
 assert(info&&info[0]*8===bitpix,'Unsupported NIfTI scalar datatype / bit depth.');
 const offsetFloat=dv.getFloat32(108,little),offset=Math.ceil(offsetFloat);
 assert(finite(offsetFloat)&&offset>=352&&offset+info[0]*n<=buffer.byteLength,'NIfTI voxel data is truncated or offset is invalid.');
 let slope=dv.getFloat32(112,little),intercept=dv.getFloat32(116,little);
 if(!finite(slope)||slope===0)slope=1;
 if(!finite(intercept))intercept=0;
 const spacing=[80,84,88].map(off=>Math.abs(dv.getFloat32(off,little)));
 assert(spacing.every(v=>finite(v)&&v>0),'Invalid voxel spacing');
 const sformCode=dv.getInt16(254,little);
 assert(sformCode>0,'This research viewer requires an explicit NIfTI sform affine for registration checks.');
 const affine=[280,296,312].map(off=>[0,4,8,12].map(d=>dv.getFloat32(off+d,little)));
 affine.push([0,0,0,1]);
 assert(affine.flat().every(finite),'Invalid NIfTI spatial transform');
 const voxel=(i,j,k)=>{
  const x=Math.max(0,Math.min(dims[0]-1,Math.round(i)));
  const y=Math.max(0,Math.min(dims[1]-1,Math.round(j)));
  const z=Math.max(0,Math.min(dims[2]-1,Math.round(k)));
  const address=offset+info[0]*(x+dims[0]*(y+dims[1]*z));
  const raw=info[0]===1?dv[info[1]](address):dv[info[1]](address,little);
  return raw*slope+intercept;
 };
 const ras=(i,j,k)=>affine.slice(0,3).map(row=>row[0]*i+row[1]*j+row[2]*k+row[3]);
 return Object.freeze({dims,spacing,datatype,slope,intercept,affine,sformCode,
  littleEndian:little,byteLength:buffer.byteLength,voxel,ras,kind:'NIfTI-1/3D'});
}
function matchQA(volume,qa){
 if(!volume||!qa)return {valid:false,reason:'CT volume and reconstruction manifest required'};
 const dims=qa.source_shape_vox;
 if(!Array.isArray(dims)||dims.length!==3||dims.some((x,i)=>x!==volume.dims[i]))
  return {valid:false,reason:'CT dimensions differ from reconstruction manifest'};
 const affine=qa.sform_RAS_mm;
 if(!Array.isArray(affine)||affine.length!==4||affine.some((row,i)=>
   !Array.isArray(row)||row.length!==4||row.some((v,j)=>!finite(v)||Math.abs(v-volume.affine[i][j])>0.001)))
  return {valid:false,reason:'CT sform RAS matrix differs from reconstruction manifest'};
 if(Math.abs((qa.HU_stored_slope??1)-volume.slope)>1e-4||
    Math.abs((qa.HU_stored_intercept??0)-volume.intercept)>1e-3)
  return {valid:false,reason:'CT scalar calibration differs from reconstruction manifest'};
 return {valid:true,reason:'Header dimensions, sform and intensity scaling match manifest; segmentation NOT validated'};
}
function getWorldFromRAS(ras,qa){
 if(!Array.isArray(ras)||ras.length!==3||ras.some(v=>!finite(v)))throw new Error('Invalid spatial coordinate');
 const mat=qa?.model_from_ras_mm;
 if(!Array.isArray(mat)||mat.length!==4||mat.some(row=>!Array.isArray(row)||row.length!==4||row.some(v=>!finite(v))))
  throw new Error('Explicit model_from_ras_mm transform is required');
 return mat.slice(0,3).map(row=>row[0]*ras[0]+row[1]*ras[1]+row[2]*ras[2]+row[3]);
}
root.PulmoNiftiSource=Object.freeze({readNifti,matchQA,getWorldFromRAS});
})(typeof window!=='undefined'?window:globalThis);