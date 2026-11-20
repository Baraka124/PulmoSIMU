/**
 * PulmoSIMU anatomical object contract.
 * Provenance travels with every selectable mesh across Studio and XR.
 * Display transformations NEVER alter source units or create clinical precision.
 */
export const REFERENCE_SOURCE='pulmosimu-airway-reference-r2';
export const SCHEMATIC_SOURCE='pulmosimu-schematic-lobes-r1';
export const LOBES=Object.freeze(['RUL','RML','RLL','LUL','LLL']);
export const SOURCE_DESCRIPTORS=Object.freeze({
 [REFERENCE_SOURCE]:Object.freeze({
  type:'procedural-reference',geometry:'source-procedural',units:'uncalibrated-scene-units',
  clinicalValidated:false,physicalScaleValidated:false,measurementsAllowed:false,
  provenance:'Original PNEUMO VR SEGS reference snapshot; no patient registration'
 }),
 [SCHEMATIC_SOURCE]:Object.freeze({
  type:'schematic-fallback',geometry:'schematic-envelope',units:'display-only',
  clinicalValidated:false,physicalScaleValidated:false,measurementsAllowed:false,
  provenance:'Illustrative ellipsoidal lobar envelopes; not segmented pleural anatomy'
 })
});
export function anatomyObjectRecord({sourceId,role,regionId=null,branchId=null}){
 const d=SOURCE_DESCRIPTORS[sourceId];
 if(!d)throw Error('Unknown anatomical source');
 if(!['lobe','airway'].includes(role))throw Error('Unknown anatomical role');
 if(regionId!==null&&!LOBES.includes(regionId))throw Error('Invalid lobar reference ID');
 if(role==='lobe'&&(sourceId!==SCHEMATIC_SOURCE||regionId===null||branchId!==null))
  throw Error('Invalid schematic lobe record');
 if(role==='airway'&&(sourceId!==REFERENCE_SOURCE||!Number.isInteger(branchId)||branchId<0))
  throw Error('Invalid airway branch record');
 const localId=role==='airway'?String(branchId):regionId;
 return Object.freeze({
  id:`${sourceId}:${role}:${localId}`,sourceId,role,regionId,branchId,
  sourceUnits:d.units,sourceGeometry:d.geometry,
  clinicalValidated:false,physicalScaleValidated:false,measurementsAllowed:false
 });
}
export function tagAnatomyObject(object,record){
 if(!object||typeof object!=='object'||!record||!SOURCE_DESCRIPTORS[record.sourceId])
  throw Error('Invalid scene anatomical object');
 if(object.userData==null||typeof object.userData!=='object')object.userData={};
 object.userData.anatomyRecord=record;
 return object;
}
export function canReportMetric(record,metric){
 // Formal gate: no length, resistance, compliance, or clinical metric can be
 // asserted from arbitrary procedural units or schematic lobar envelopes.
 if(!record||!SOURCE_DESCRIPTORS[record.sourceId])return false;
 if(!record.measurementsAllowed||!record.physicalScaleValidated)return false;
 return ['length','area','volume','diameter','angle'].includes(metric);
}
