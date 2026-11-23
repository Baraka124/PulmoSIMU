/* PulmoSIMU CT research-surface bridge.
 * The A1/A2 airspace GLB is a source-derived study object, NOT a
 * patient-specific segmented airway or an original simulator branch graph.
 * Verify bytes before attaching the surface to the original THREE scene.
 */
(function(global){
'use strict';
const SHA256=/^[a-f0-9]{64}$/i;
function isResearchManifest(m){
 return Boolean(m&&typeof m==='object'&&!Array.isArray(m)
  &&typeof m.geometry_class==='string'
  &&/CT_DERIVED_AIRSPACE_MASK/.test(m.geometry_class)
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
 return Object.freeze({kind:'ct-airspace-research',study:'CT-DERIVED AIRSPACE · RESEARCH ONLY',
  anatomyValidated:false,bronchoscopyAllowed:false,
  sourceVolumeVerified:false,
  warning:flags.right_lung_mask_leakage==='FLAGGED_FOR_REVIEW'
   ?'RIGHT-SURFACE QA FLAGGED · SOURCE SEGMENTATION UNREVIEWED'
   :'SOURCE SEGMENTATION NOT CLINICALLY VALIDATED'});
}
global.PulmoCTResearchBridge=Object.freeze({isResearchManifest,verifyGLB,describe});
})(typeof window!=='undefined'?window:globalThis);
