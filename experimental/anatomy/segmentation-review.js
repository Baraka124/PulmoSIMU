(function segmentation(root){
'use strict';
const assert=(x,m)=>{if(!x)throw Error(m)};
const hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const kinds=['airway_lumen','RUL','RML','RLL','LUL','LLL','pulmonary_artery','pulmonary_vein','lung'];
function bind(study,manifest,bytes,digest){
 assert(study?.schema==='pulmosimu.study/1'&&hash(study.source?.sha256),'Valid H study required');
 assert(manifest?.schema==='pulmosimu.mask/1'&&manifest.ct_sha256===study.source.sha256,'Mask belongs to another CT');
 assert(hash(digest)&&manifest.mask_sha256===digest,'Mask SHA-256 mismatch');
 assert(kinds.includes(manifest.structure),'Unsupported structure');
 const d=study.source.dimensions;
 assert(Array.isArray(d)&&d.length===3&&d.every(n=>Number.isSafeInteger(n)&&n>=2),'Invalid dimensions');
 assert(Array.isArray(manifest.dimensions)&&manifest.dimensions.length===3&&d.every((n,i)=>n===manifest.dimensions[i]),'Mask dimensions differ');
 const a=manifest.voxel_to_ras_mm,b=study.spatial.voxel_to_ras_mm;
 assert(Array.isArray(a)&&a.length===4&&a.every((r,i)=>Array.isArray(r)&&r.length===4&&r.every((x,j)=>Number.isFinite(x)&&Math.abs(x-b[i][j])<=.001)),'Mask affine differs');
 assert(manifest.encoding==='uint8-x-fast-binary'&&bytes instanceof Uint8Array&&bytes.length===d[0]*d[1]*d[2],'Invalid raw mask length or encoding');
 let count=0;for(const v of bytes){assert(v===0||v===1,'Mask is not binary');count+=v;}
 const edits=new Map();
 function index(p){assert(Array.isArray(p)&&p.length===3&&p.every((x,i)=>Number.isInteger(x)&&x>=0&&x<d[i]),'Voxel out of bounds');return p[0]+d[0]*(p[1]+d[1]*p[2]);}
 function at(p){const i=index(p);return edits.get(i)??bytes[i];}
 function set(p,value){assert(value===0||value===1,'Invalid binary edit');const i=index(p);assert(edits.has(i)||value===bytes[i]||edits.size<10000,'Edit limit reached');if(value===bytes[i])edits.delete(i);else edits.set(i,value);}
 function metrics(){let added=0,removed=0;for(const [i,v] of edits){if(v===1)added++;else removed++;}const m=b;const det=Math.abs(m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]));const revised=count+added-removed;return {original_voxels:count,edited_voxels:revised,added,removed,volume_ml:revised*det/1000,dice_to_original:count+revised?2*(count-removed)/(count+revised):null};}
 function patch(){return {schema:'pulmosimu.mask-patch/1',ct_sha256:study.source.sha256,mask_sha256:digest,structure:manifest.structure,edits:[...edits].sort((a,b)=>a[0]-b[0]).map(([index,value])=>({index,value})),review:'PENDING'};}
 return Object.freeze({at,set,metrics,patch,clear:()=>edits.clear(),structure:manifest.structure,digest});
}
root.PulmoSegmentation=Object.freeze({bind,kinds:Object.freeze(kinds)});
})(typeof window!=='undefined'?window:globalThis);
