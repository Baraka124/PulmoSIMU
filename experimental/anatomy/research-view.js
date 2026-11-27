/* PulmoSIMU CT research-surface bridge.
 * The A1/A2 airspace GLB is a source-derived study object, NOT a
 * patient-specific segmented airway or an original simulator branch graph.
 * Verify bytes before attaching the surface to the original THREE scene.
 */
(function(global){
'use strict';
const SHA256=/^[a-f0-9]{64}$/i;
function isWorkerAirway(m){
 const a=m?.model_from_ras_mm;
 return Boolean(m?.validation==="UNREVIEWED_SOURCE_MASK_SURFACE"&&SHA256.test(m.mask_sha256||"")&&
  Array.isArray(m.source_shape_vox)&&m.source_shape_vox.length===3&&m.source_shape_vox.every(x=>Number.isSafeInteger(x)&&x>=2)&&
  Array.isArray(a)&&a.length===4&&a.every(r=>Array.isArray(r)&&r.length===4&&r.every(Number.isFinite))&&
  a[3].every((x,i)=>x===(i===3?1:0))&&
  [[.001,0,0],[0,0,.001],[0,-.001,0]].every((row,i)=>row.every((x,j)=>Math.abs(x-a[i][j])<1e-10)));
}
function isResearchManifest(m){
 return Boolean(m&&typeof m==='object'&&!Array.isArray(m)
  &&(isWorkerAirway(m)||/^(?:PRELIMINARY_CT_DERIVED_AIRSPACE_MASK|A2_1_CONSERVATIVE_BILATERAL_CT_AIRSPACE_REVIEW)$/.test(m.geometry_class||''))
  &&typeof m.mesh_sha256==='string'&&SHA256.test(m.mesh_sha256)
  &&typeof m.source_sha256==='string'&&SHA256.test(m.source_sha256)
  &&Array.isArray(m.model_from_ras_mm)&&m.model_from_ras_mm.length===4
  &&!Array.isArray(m.segments));
}
async function verifyGLB(buffer,manifest,cryptoProvider=global.crypto){
 if(!isResearchManifest(manifest))throw Error('Not a recognized CT research-surface manifest');
 if(!(buffer instanceof ArrayBuffer)||!buffer.byteLength)throw Error('CT surface bytes missing');
 if(!cryptoProvider?.subtle?.digest)throw Error('SHA-256 unavailable; study surface will not load unverified');
 const data=await cryptoProvider.subtle.digest('SHA-256',buffer);
 const sha=Array.from(new Uint8Array(data),v=>v.toString(16).padStart(2,'0')).join('');
 if(sha.toLowerCase()!==manifest.mesh_sha256.toLowerCase())
  throw Error('CT surface SHA-256 does not match selected study manifest');
 return Object.freeze({verifiedBytes:true,sha256:sha,
  clinicalValidated:false,sourceVolumeVerified:false,
  airwayRegistrationVerified:false});
}
function describe(manifest){
 if(!isResearchManifest(manifest))return null;
 const flags=manifest.quality_gates||{};
 return Object.freeze({kind:'ct-airspace-research',study:isWorkerAirway(manifest)?'CT-DERIVED AIRWAY LUMEN · RESEARCH ONLY':'CT-DERIVED AIRSPACE · RESEARCH ONLY',
  anatomyValidated:false,bronchoscopyAllowed:false,
  sourceVolumeVerified:false,
  warning:flags.right_lung_mask_leakage==='FLAGGED_FOR_REVIEW'
   ?'RIGHT-SURFACE QA FLAGGED · SOURCE SEGMENTATION UNREVIEWED'
   :'SOURCE SEGMENTATION NOT CLINICALLY VALIDATED'});
}
global.PulmoCTResearchBridge=Object.freeze({isResearchManifest,isWorkerAirway,verifyGLB,describe});
})(typeof window!=='undefined'?window:globalThis);
