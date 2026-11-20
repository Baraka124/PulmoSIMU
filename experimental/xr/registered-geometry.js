/* PulmoSIMU XR-08: fail-closed geometry and scope contract.
 * Classic script, compatible with PNEUMO·VR Three.js r128.
 * Display-frame co-location is NOT a DICOM-to-patient registration
 * validation and radius checks are NOT wall-contact physics.
 */
(function(global){
'use strict';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const triplet=x=>Array.isArray(x)&&x.length===3&&x.every(finite);
const fail=msg=>{throw Error('XR-08 geometry: '+msg)};
function assessRegistration(meta,bounds){
 if(!meta||meta.version!=='pneumovr-1.1')fail('unsupported metadata version');
 const tr=meta.transform;
 if(!tr||!triplet(tr.centre_mm)||!finite(tr.scale)||tr.scale<=0||tr.scale>10000)fail('missing or invalid physical transform');
 if(!bounds||!triplet(bounds.min)||!triplet(bounds.max))fail('GLB surface bounds are missing');
 if(bounds.min.some((x,i)=>x>=bounds.max[i]))fail('degenerate GLB bounds');
 if(!Array.isArray(meta.segments)||!meta.segments.length)fail('airway centerline graph missing');
 if(meta.segments.length>100000)fail('unreasonable graph size');
 const idSet=new Set(),segments=new Map(),centerline=[];
 for(const seg of meta.segments){
  if(!Number.isInteger(seg.id)||idSet.has(seg.id))fail('duplicate or invalid airway ID');
  idSet.add(seg.id);segments.set(seg.id,seg);
  if(!Array.isArray(seg.pts)||seg.pts.length<2||seg.pts.length>50000||seg.pts.some(x=>!triplet(x)))fail('invalid airway coordinates');
  if(!finite(seg.r0)||!finite(seg.r1)||seg.r0<=0||seg.r1<=0)fail('invalid airway radius');
  for(const p of seg.pts)centerline.push(p);
 }
 const roots=meta.segments.filter(s=>!idSet.has(s.parent));
 if(roots.length!==1)fail('airway graph must have exactly one root');
 const visiting=new Set(),visited=new Set(),children=new Map([...idSet].map(id=>[id,[]]));
 for(const seg of meta.segments)if(seg.id!==roots[0].id){
  if(!idSet.has(seg.parent))fail('missing airway parent');
  children.get(seg.parent).push(seg.id);
 }
 function walk(id){
  if(visiting.has(id))fail('cyclic airway graph');
  if(visited.has(id))fail('multiple parent airway graph');
  visiting.add(id);visited.add(id);
  for(const child of children.get(id))walk(child);
  visiting.delete(id);
 }
 walk(roots[0].id);
 if(visited.size!==meta.segments.length)fail('disconnected airway graph');
 const diagonal=Math.hypot(...bounds.min.map((v,i)=>bounds.max[i]-v));
 if(!finite(diagonal)||diagonal<1e-6)fail('degenerate GLB diagonal');
 const margin=Math.max(diagonal*.12,.006);
 let outside=0;
 for(const p of centerline)if(p.some((v,i)=>v<bounds.min[i]-margin||v>bounds.max[i]+margin))outside++;
 if(outside!==0)fail('GLB and airway centerline coordinates are not co-framed');
 const extent=bounds.min.map((v,i)=>bounds.max[i]-v);
 const sceneUnitMetres=1/tr.scale;
 if(!finite(sceneUnitMetres)||sceneUnitMetres<=0)fail('invalid metres-per-source-unit');
 const synthetic=Boolean(tr.is_synthetic);
 const result=Object.freeze({
  status:'coframed',source:synthetic?'synthetic':'ct-derived',
  meshAlignment:'plausible-only',clinicalValidated:false,anatomicalValidated:false,
  fullDicomOrientationValidated:false,modelToPatientRegistrationValidated:false,
  sceneUnitMetres,sourceScaleEligible:!synthetic,
  branchCount:segments.size,centerlinePoints:centerline.length,
  boundsDiagonalSceneUnits:diagonal,boundsDiagonalMetres:diagonal*sceneUnitMetres,
  maxBoundsMarginSceneUnits:margin,
  // No patient names, IDs or identifying metadata are retained.
  notice:synthetic?'Synthetic reference; source scale not eligible':
    'Source-scale inferred from pipeline transform; DICOM registration not independently verified'
 });
 return result;
}
function assessLumenClearance({segment,t,scopeDiameterMm,mmPerSceneUnit,minClearanceMm=.25}={}){
 if(!segment||!finite(segment.r0)||!finite(segment.r1)||segment.r0<=0||segment.r1<=0||
 !finite(t)||!finite(scopeDiameterMm)||scopeDiameterMm<=0||
 !finite(mmPerSceneUnit)||mmPerSceneUnit<=0||
 !finite(minClearanceMm)||minClearanceMm<0)fail('invalid scope clearance inputs');
 const fraction=Math.max(0,Math.min(1,t));
 const diameterMm=2*(segment.r0+(segment.r1-segment.r0)*fraction)*mmPerSceneUnit;
 const clearanceMm=diameterMm-scopeDiameterMm;
 return Object.freeze({
  canAdvance:clearanceMm>=minClearanceMm,diameterMm,scopeDiameterMm,clearanceMm,
  basis:'source-radius-estimate',wallContactSimulated:false,
  clinicalValidated:false
 });
}
global.PulmoXR08=Object.freeze({assessRegistration,assessLumenClearance});
})(typeof window!=='undefined'?window:globalThis);
