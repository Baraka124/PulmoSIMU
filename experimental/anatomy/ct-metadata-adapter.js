/**
 * PulmoSIMU anatomical provenance / CT-derived airway metadata adapter.
 *
 * The original pipeline exports physically measured segment centreline and
 * radii in its *display frame*: pts = (pt_mm-centre_mm)/1000*scale.
 * Consequently metres per scene unit = 1/scale. No guesswork is permitted.
 * This parser processes metadata only; GLB rendering and CT segmentation
 * quality must be validated independently. Never persist patient identifiers
 * from the metadata in UI/readouts.
 */
import {loadAirwayGraph} from './airway-graph.js';

const finite=(v)=>typeof v==='number'&&Number.isFinite(v);
const triplet=(p)=>Array.isArray(p)&&p.length===3&&p.every(finite);
const fail=(reason)=>{throw Error('Invalid PNEUMO·VR metadata: '+reason)};
export function adaptCtMetadata(metadata){
 if(!metadata||!Array.isArray(metadata.segments)||!metadata.segments.length)fail('missing segments');
 if(!metadata.transform||!finite(metadata.transform.scale)||metadata.transform.scale<=0)fail('missing positive transform.scale');
 if(!triplet(metadata.transform.centre_mm))fail('centre_mm required');
 const sceneUnitMetres=1/metadata.transform.scale;
 // Pipeline's fixed target_span=2.4 is a graphics normalization, not lung size.
 if(!finite(sceneUnitMetres)||sceneUnitMetres<=0)fail('invalid units');
 const raw=metadata.segments;
 const idSet=new Set(raw.map(s=>s.id));
 if(idSet.size!==raw.length||raw.some(s=>!Number.isInteger(s.id)))fail('invalid or duplicate IDs');
 const rootCandidates=raw.filter(s=>!idSet.has(s.parent));
 if(rootCandidates.length!==1)fail('expected a single proximal root');
 const rootId=rootCandidates[0].id;
 const children=new Map(raw.map(s=>[s.id,[]]));
 for(const s of raw)if(s.id!==rootId){
  if(!children.has(s.parent))fail('missing parent for '+s.id);
  children.get(s.parent).push(s.id);
 }
 // Normalise arbitrary CT pipeline IDs to a tree rooted at 0 in *graph IDs*,
 // but retain source IDs for cross-reference with patient GLB metadata.
 const visited=new Set(),active=new Set();
 function visit(id){if(active.has(id))fail('cyclic graph');if(visited.has(id))fail('multiple parents');active.add(id);visited.add(id);for(const c of children.get(id))visit(c);active.delete(id)}
 visit(rootId);if(visited.size!==raw.length)fail('disconnected components');
 const sorted=[rootId,...raw.map(x=>x.id).filter(id=>id!==rootId)];
 const rename=new Map(sorted.map((id,i)=>[id,i]));
 const nodes=sorted.map(sourceId=>{
  const s=raw.find(x=>x.id===sourceId);
  if(!Array.isArray(s.pts)||s.pts.length<2||!s.pts.every(triplet))fail('invalid centreline: '+sourceId);
  if(!finite(s.r0)||!finite(s.r1)||s.r0<=0||s.r1<=0)fail('invalid radii: '+sourceId);
  return {id:rename.get(sourceId),sourceId,name:String(s.name||'Unnamed airway'),
   generation:Number.isInteger(s.gen)?s.gen:0,radiusStart:s.r0,radiusEnd:s.r1,
   points:s.pts.map(x=>[...x]),children:children.get(sourceId).map(id=>rename.get(id))};
 });
 const graph=loadAirwayGraph({
  schema:'pulmosimu-airway-reference-r2',
  units:'CT pipeline display scene units; SI conversion requires provenance.sceneUnitMetres',
  source:'PNEUMO·VR pneumovr-1.1 metadata',
  nodes
 });
 // Region labels in CT segmentation may not be sufficiently specific.
 // The graph adapter's branch-ID-based lobe classifier must NOT be used
 // because CT segment identifiers do not match procedural SEGS IDs.
 const provenance=Object.freeze({
  kind:'ct-derived',conversion:'metresPerSceneUnit = 1 / transform.scale',
  sceneUnitMetres,worldFrame:'PNEUMO·VR centred and scaled GLB',
  originalSourceIds:sorted,validated:false,patientSpecific:true,
  disclaimer:'CT-derived segment locations; neither segment labels nor radii independently validated'
 });
 return {graph,provenance};
}
/** Precise inverse of the transform applied in pipeline.py. */
export function sceneToPatientMm(scenePoint,transform){
 if(!triplet(scenePoint))fail('scene point must be [x,y,z]');
 if(!transform||!finite(transform.scale)||transform.scale<=0||!triplet(transform.centre_mm))fail('invalid transform');
 return scenePoint.map((v,i)=>v*1000/transform.scale+transform.centre_mm[i]);
}
export function patientMmToScene(mmPoint,transform){
 if(!triplet(mmPoint))fail('point must be [x,y,z]');
 if(!transform||!finite(transform.scale)||transform.scale<=0||!triplet(transform.centre_mm))fail('invalid transform');
 return mmPoint.map((v,i)=>(v-transform.centre_mm[i])/1000*transform.scale);
}
