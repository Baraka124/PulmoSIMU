/**
 * PulmoSIMU shared anatomical focus protocol v1.
 * Deliberately small: only explicit canonical reference anatomy identifiers
 * and navigation focus cross the workspaces. No clinical images, patient data,
 * drug parameters, physiological state, or procedural camera/scope commands.
 */
export const CHANNEL='pulmosimu.reference-focus.v1';
export const SOURCE='original-procedural-SEGS-snapshot-r2';
export const REGIONS=Object.freeze(['RUL','RML','RLL','LUL','LLL']);
const regionSet=new Set(REGIONS);
export const BRANCH_REGION_ROOTS=Object.freeze({2:'LUL',3:'LLL',5:'RUL',6:'RML',7:'RLL'});
const allowedSources=new Set(['procedural','mechanistic','xr','studio']);
export function regionForBranch(graph,branchId){
 if(!graph||!Number.isInteger(branchId))return null;
 // Work only on the versioned reference graph, not any imported patient CT.
 if(graph.schema!=='pulmosimu-airway-reference-r2')return null;
 if(typeof graph.regionOf==='function')return graph.regionOf(branchId);
 return null;
}
export function createFocus({regionId,branchId=null,origin='studio',revision=0}){
 if(!regionSet.has(regionId))throw Error('Unknown reference lobe');
 if(!allowedSources.has(origin))throw Error('Unknown workspace origin');
 if(branchId!==null&&(!Number.isInteger(branchId)||branchId<0||branchId>10000))throw Error('Invalid branch ID');
 if(!Number.isSafeInteger(revision)||revision<0)throw Error('Invalid revision');
 return Object.freeze({channel:CHANNEL,source:SOURCE,type:'anatomical-focus',
  regionId,branchId,origin,revision});
}
export function readFocus(value){
 if(!value||typeof value!=='object'||Array.isArray(value))return null;
 if(value.channel!==CHANNEL||value.source!==SOURCE||value.type!=='anatomical-focus')return null;
 try{return createFocus(value)}catch{return null}
}
/** Receiver must separately check event.origin and event.source. */
export function acceptFocus(current,candidate){
 const next=readFocus(candidate);
 if(!next||next.origin==='studio'&&next.revision===0)return current;
 if(current&&next.revision<=current.revision)return current;
 return next;
}
