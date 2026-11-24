/* A2.3 — research-only CT lobe review records. No geometry edits or clinical unlocks. */
(function(root){
'use strict';
const TYPES=Object.freeze(['right-oblique','right-horizontal','left-oblique','uncertain','artifact']);
const HEX=/^[0-9a-f]{64}$/i;
function source(manifest){
  if(!manifest||!HEX.test(manifest.source_sha256||'')||!HEX.test(manifest.mesh_sha256||''))
    throw Error('Exact SHA-256 CT and mesh manifest required');
  if(!Array.isArray(manifest.model_from_ras_mm)||manifest.model_from_ras_mm.length!==4)
    throw Error('Model-to-RAS transform missing');
  return {ct_sha256:manifest.source_sha256.toLowerCase(),mesh_sha256:manifest.mesh_sha256.toLowerCase()};
}
function point(raw){
  if(!raw||!TYPES.includes(raw.type))throw Error('Unknown annotation type');
  if(!Array.isArray(raw.ras_mm)||raw.ras_mm.length!==3||
     !raw.ras_mm.every(n=>typeof n==='number'&&Number.isFinite(n)&&Math.abs(n)<2000))
    throw Error('Invalid RAS millimetre coordinates');
  const note=String(raw.note||'').trim();
  if(note.length>240)throw Error('Annotation note too long');
  return Object.freeze({type:raw.type,ras_mm:raw.ras_mm.map(n=>Math.round(n*100)/100),note});
}
function session(manifest,settings={},points=[]){
  const identity=source(manifest);
  if(!root.PulmoCTLobeTerritories?.parameters)throw Error('A2.2 lobe atlas unavailable');
  if(!Array.isArray(points)||points.length>500)throw Error('Maximum 500 review points');
  return Object.freeze({
    schema:'pulmosimu.ct-lobe-review/1',
    status:'RESEARCH_HYPOTHESES_ONLY',
    source:identity,
    atlas:{...root.PulmoCTLobeTerritories.parameters(settings)},
    annotations:points.map(point)
  });
}
function parse(value,manifest){
  const payload=typeof value==='string'?JSON.parse(value):value;
  if(!payload||payload.schema!=='pulmosimu.ct-lobe-review/1'||
     payload.status!=='RESEARCH_HYPOTHESES_ONLY')throw Error('Unsupported review schema');
  const expected=source(manifest);
  if(payload.source?.ct_sha256!==expected.ct_sha256||
     payload.source?.mesh_sha256!==expected.mesh_sha256)throw Error('CT/GLB hash mismatch: review not imported');
  if(!payload.atlas||typeof payload.atlas!=='object'||Array.isArray(payload.atlas))
    throw Error('Atlas settings missing');
  for(const k of ['rightMajor','rightMinor','leftMajor'])
    if(!Number.isFinite(payload.atlas[k]))throw Error('Invalid atlas offset');
  const s=session(manifest,payload.atlas,payload.annotations);
  for(const k of ['rightMajor','rightMinor','leftMajor','selected'])
    if(s.atlas[k]!==payload.atlas[k])throw Error('Atlas settings outside supported range');
  return s;
}
root.PulmoCTReview=Object.freeze({TYPES,source,point,session,parse});
})(typeof window!=='undefined'?window:globalThis);
