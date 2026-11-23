/**
 * PulmoSIMU Workspace Integration — URL handoff for research focus only.
 * This is deliberately not a patient-data exchange or clinical session store.
 * Only five canonical procedural-reference regions can cross workspaces.
 */
import {REGIONS} from './reference-focus.js';
export const WORKSPACES=Object.freeze({
 procedure:Object.freeze({path:'./procedural-workspace.html',origin:'procedural'}),
 mechanisms:Object.freeze({path:'./experimental/lab.html',origin:'mechanistic'}),
 source:Object.freeze({path:'./experimental/anatomy-inspector.html',origin:'source-local'}),
 xr:Object.freeze({path:'./index.html?xr=procedural&release=XR09-UNIFIED',origin:'xr'})
});
const regionSet=new Set(REGIONS);
export function safeReferenceRegion(v){return regionSet.has(v)?v:null}
export function safeWorkspace(v){return Object.hasOwn(WORKSPACES,v)?v:null}
/** Parse launch state: invalid values are ignored, never interpreted as data. */
export function readWorkspaceLaunch(query=''){
 const params=new URLSearchParams(String(query).replace(/^\?/,'')); 
 const mode=safeWorkspace(params.get('mode'))||'procedure';
 return Object.freeze({mode,
  // A source NIfTI/GLB study must never receive synthetic reference-lobe context.
  regionId:mode==='source'?null:safeReferenceRegion(params.get('focus'))});
}
/** Make a same-origin relative URL with only well-formed canonical focus. */
export function workspaceLaunch(path,{mode=null,regionId=null}={}){
 const url=new URL(path,'https://reference.local/');
 if(mode!==null){if(!safeWorkspace(mode))throw Error('Invalid workspace');url.searchParams.set('mode',mode)}
 if(regionId!==null){if(!safeReferenceRegion(regionId))throw Error('Invalid reference region');url.searchParams.set('focus',regionId)}
 return url.pathname.replace(/^\//,'')+url.search+url.hash;
}
