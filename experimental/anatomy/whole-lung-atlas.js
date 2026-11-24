/**
 * PulmoSIMU — anatomical structure registry (A1).
 * Standard lobar and commonly used bronchopulmonary segment nomenclature.
 * Segment variants, especially left S7/S8 and S1/S2, exist. This is a
 * topology/terminology atlas, not a segmentation mask or coordinate model.
 * No vessel location, fissure curvature, or lobar volume is inferred.
 */
export const LOBES=Object.freeze([
 {id:'RUL',side:'right',name:'Right upper lobe',segmentIds:['R-S1','R-S2','R-S3']},
 {id:'RML',side:'right',name:'Right middle lobe',segmentIds:['R-S4','R-S5']},
 {id:'RLL',side:'right',name:'Right lower lobe',segmentIds:['R-S6','R-S7','R-S8','R-S9','R-S10']},
 {id:'LUL',side:'left',name:'Left upper lobe (including lingula)',segmentIds:['L-S1+2','L-S3','L-S4','L-S5']},
 {id:'LLL',side:'left',name:'Left lower lobe',segmentIds:['L-S6','L-S7+8','L-S9','L-S10']}
]);
const segment=(id,lobe,name,variantNote='')=>({id,lobe,name,variantNote,geometry:'not-provided'});
export const SEGMENTS=Object.freeze([
 segment('R-S1','RUL','Apical'),segment('R-S2','RUL','Posterior'),segment('R-S3','RUL','Anterior'),
 segment('R-S4','RML','Lateral'),segment('R-S5','RML','Medial'),
 segment('R-S6','RLL','Superior'),segment('R-S7','RLL','Medial basal'),segment('R-S8','RLL','Anterior basal'),
 segment('R-S9','RLL','Lateral basal'),segment('R-S10','RLL','Posterior basal'),
 segment('L-S1+2','LUL','Apicoposterior','S1 and S2 may be described separately'),
 segment('L-S3','LUL','Anterior'),segment('L-S4','LUL','Superior lingular'),segment('L-S5','LUL','Inferior lingular'),
 segment('L-S6','LLL','Superior'),segment('L-S7+8','LLL','Anteromedial basal','S7 and S8 may be described separately; variable medial basal bronchus'),
 segment('L-S9','LLL','Lateral basal'),segment('L-S10','LLL','Posterior basal')
]);
export const FISSURES=Object.freeze([
 {id:'right-oblique',side:'right',separates:[['RUL','RLL'],['RML','RLL']],name:'Right oblique fissure'},
 {id:'right-horizontal',side:'right',separates:[['RUL','RML']],name:'Right horizontal fissure'},
 {id:'left-oblique',side:'left',separates:[['LUL','LLL']],name:'Left oblique fissure'}
]);
export const VASCULAR_LANDMARKS=Object.freeze([
 {id:'RPA',side:'right',name:'Right pulmonary artery',role:'pulmonary arterial inflow',relatedLobes:['RUL','RML','RLL'],geometry:'not-provided'},
 {id:'LPA',side:'left',name:'Left pulmonary artery',role:'pulmonary arterial inflow',relatedLobes:['LUL','LLL'],geometry:'not-provided'},
 {id:'RSPV',side:'right',name:'Right superior pulmonary vein',role:'venous return',relatedLobes:['RUL','RML'],geometry:'not-provided'},
 {id:'RIPV',side:'right',name:'Right inferior pulmonary vein',role:'venous return',relatedLobes:['RLL'],geometry:'not-provided'},
 {id:'LSPV',side:'left',name:'Left superior pulmonary vein',role:'venous return',relatedLobes:['LUL'],geometry:'not-provided'},
 {id:'LIPV',side:'left',name:'Left inferior pulmonary vein',role:'venous return',relatedLobes:['LLL'],geometry:'not-provided'}
]);
export const ATLAS_PROVENANCE=Object.freeze({
 schema:'pulmosimu-anatomical-atlas-a1',kind:'standard-nomenclature',
 geometry:'none',measured:false,patientSpecific:false,
 disclaimer:'Terminology and typical drainage only. Anatomical variants exist. Does not represent patient geometry, fissure positions or vessel trajectories.'
});
export function validateAtlas(){
 const lobes=new Set(LOBES.map(x=>x.id)),segments=new Set(SEGMENTS.map(x=>x.id));
 if(lobes.size!==5||segments.size!==SEGMENTS.length)throw Error('Duplicate or missing anatomical IDs');
 if(LOBES.some(x=>x.segmentIds.some(y=>!segments.has(y))))throw Error('Missing segment');
 if(SEGMENTS.some(x=>!lobes.has(x.lobe)))throw Error('Invalid segment lobe');
 const allocation=LOBES.flatMap(x=>x.segmentIds);
 if(allocation.length!==segments.size||new Set(allocation).size!==segments.size)throw Error('Segment belongs to multiple lobes');
 if(FISSURES.some(f=>f.separates.some(pair=>pair.some(x=>!lobes.has(x)))))throw Error('Invalid fissure adjacency');
 if(VASCULAR_LANDMARKS.some(x=>x.relatedLobes.some(l=>!lobes.has(l))))throw Error('Invalid vascular relation');
 return {lobes:lobes.size,segments:segments.size,fissures:FISSURES.length,vascularLandmarks:VASCULAR_LANDMARKS.length};
}
export function describeLobe(id){
 const lobe=LOBES.find(x=>x.id===id);
 if(!lobe)throw Error('Unknown lung lobe '+id);
 return { ...lobe,segments:SEGMENTS.filter(s=>s.lobe===id),
  fissures:FISSURES.filter(f=>f.separates.some(pair=>pair.includes(id))),
  vessels:VASCULAR_LANDMARKS.filter(v=>v.relatedLobes.includes(id)),
  shapeProvenance:'no anatomical surface data'};
}
