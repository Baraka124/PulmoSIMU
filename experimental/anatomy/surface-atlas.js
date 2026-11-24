/**
 * PulmoSIMU A2 — explicit surface geometry registration contract.
 *
 * A surface atlas is supplied by an independently licensed source. This file
 * contains NO anatomical coordinates, no clinical segmentation, and no
 * inference from bounding boxes. Labels are mapped ONLY by manifest entries.
 * All geometry remains unvalidated until separately reviewed.
 */
import {LOBES,VASCULAR_LANDMARKS} from './whole-lung-atlas.js';

const allowed=new Map([
 ...LOBES.map(l=>[l.id,'lobe']),
 ...VASCULAR_LANDMARKS.map(v=>[v.id,'vessel'])
]);
const isText=v=>typeof v==='string'&&v.trim().length>0&&v.length<=240;
const vector=v=>Array.isArray(v)&&v.length===3&&v.every(n=>typeof n==='number'&&Number.isFinite(n));
const clean=s=>s.trim();
export function validateSurfaceManifest(data){
 if(!data||data.schema!=='pulmosimu-surface-atlas-v1')throw Error('Unsupported atlas manifest schema');
 if(!isText(data.datasetId))throw Error('Dataset ID required');
 if(!isText(data.source)||!isText(data.license)||!isText(data.rightsStatement))throw Error('Source, license and rights statement required');
 if(!isText(data.coordinateFrame))throw Error('Coordinate frame required');
 if(!['mm','m'].includes(data.unit))throw Error('Physical unit must be mm or m');
 if(data.patientSpecific!==false)throw Error('Research atlas importer requires non-patient-specific data');
 if(!Array.isArray(data.structures)||data.structures.length===0||data.structures.length>100)throw Error('Expected 1..100 structures');
 const objectNames=new Set(),ids=new Set();
 const structures=data.structures.map((entry,i)=>{
   if(!entry||!isText(entry.id)||!allowed.has(entry.id))throw Error('Unknown anatomical structure at index '+i);
   if(ids.has(entry.id))throw Error('Duplicate structure '+entry.id);
   if(entry.kind!==allowed.get(entry.id))throw Error('Structure kind inconsistent with atlas');
   if(!isText(entry.objectName)||objectNames.has(entry.objectName))throw Error('Duplicate or missing GLB object name');
   if(!['derived','segmented','authored'].includes(entry.geometryOrigin))throw Error('Geometry origin required');
   if(!isText(entry.evidence))throw Error('Structure evidence note required');
   if(entry.sourceValidated!==false&&entry.sourceValidated!==true)throw Error('Validation flag must be explicit');
   objectNames.add(entry.objectName);ids.add(entry.id);
   return Object.freeze({
     id:entry.id,kind:entry.kind,objectName:clean(entry.objectName),
     geometryOrigin:entry.geometryOrigin,evidence:clean(entry.evidence),
     sourceValidated:entry.sourceValidated
   });
 });
 // Physical scale conversion is uniform. Registration into the PNEUMO·VR
 // procedural frame is forbidden without an independently verified transform.
 const metresPerSourceUnit=data.unit==='mm'?0.001:1;
 return Object.freeze({
   schema:data.schema,datasetId:clean(data.datasetId),source:clean(data.source),
   license:clean(data.license),rightsStatement:clean(data.rightsStatement),
   coordinateFrame:clean(data.coordinateFrame),unit:data.unit,
   patientSpecific:false,metresPerSourceUnit,
   structures:Object.freeze(structures),
   validation:{anatomyIndependentlyValidated:false,
     caveat:'Manifest confirms claimed structure identity only; it cannot verify geometry, registration, segmentation accuracy or rights.'}
 });
}
export function matchAtlasObjects(manifest,objectNames){
 if(!Array.isArray(objectNames)||objectNames.some(x=>typeof x!=='string'))throw Error('GLB object list invalid');
 const available=new Set(objectNames);
 const missing=manifest.structures.filter(s=>!available.has(s.objectName)).map(s=>s.objectName);
 if(missing.length)throw Error('GLB missing objects: '+missing.join(', '));
 return Object.freeze(Object.fromEntries(manifest.structures.map(s=>[s.objectName,s.id])));
}
export function atlasCoverage(manifest){
 const lobeIds=manifest.structures.filter(s=>s.kind==='lobe').map(s=>s.id);
 const vesselIds=manifest.structures.filter(s=>s.kind==='vessel').map(s=>s.id);
 const missingLobes=LOBES.filter(x=>!lobeIds.includes(x.id)).map(x=>x.id);
 return {lobes:lobeIds.length,vessels:vesselIds.length,
  missingLobes,completeLobarSet:missingLobes.length===0,
  sourceValidatedCount:manifest.structures.filter(x=>x.sourceValidated).length,
  anatomicalAccuracyEstablished:false};
}
