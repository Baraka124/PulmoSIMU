/* PulmoSIMU XR-14: source-derived procedure guidance.
 * Guidance is built from the ORIGINAL SEG_MAP and scope state.
 * No invented patient anatomy, case-specific route, or safety prediction.
 */
(function(global){
'use strict';
const TOOLS=Object.freeze({nav:'NAVIGATION',biopsy:'BIOPSY SIMULATION',
  measure:'MEASUREMENT',annotate:'ANNOTATION'});
const clamp=(n,low,high)=>Math.max(low,Math.min(high,n));
function route({segments={},segmentId,history=[],position=0,lengthMm=()=>0}={}){
 const current=segments[segmentId];if(!current)return {labels:[],distanceMm:0,children:[]};
 const visited=[];let id=segmentId;
 for(let i=0;i<64&&id!==undefined&&!visited.includes(id);i++){
  visited.unshift(id);const parent=Object.values(segments).find(s=>s.children?.includes(id));id=parent?.id;
 }
 const distanceMm=visited.reduce((sum,key,i)=>sum+Math.max(0,Number(lengthMm(key))||0)*(i===visited.length-1?clamp(Number(position)||0,0,1):1),0);
 return Object.freeze({ids:visited,labels:visited.map(k=>segments[k]?.name||String(k)),distanceMm,children:(current.children||[]).map((k,index)=>({id:k,index,label:segments[k]?.name||String(k)}))});
}
function describe({segmentId,segment,segments={},position=0,atBranch=false,
  clinicalMode='nav',clearance=null,monitorStatus='not-started',guideStep=0}={}){
 if(!segment||!Number.isFinite(Number(segmentId))){
  return Object.freeze({status:'UNAVAILABLE',heading:'NO SOURCE AIRWAY',
   location:'Unknown reference segment',generation:null,progressPercent:0,
   instruction:'Original airway graph is unavailable. Do not infer a route.',
   branches:[],tool:'UNAVAILABLE',monitor:'NO LIVE SOURCE',step:0,canChooseBranch:false,
   provenance:'REFERENCE DATA ONLY · NOT CLINICALLY VALIDATED'});
 }
 const children=(Array.isArray(segment.children)?segment.children:[])
   .map((id,i)=>({id,index:i,segment:segments[id]}))
   .filter(x=>x.segment&&typeof x.segment.name==='string')
   .slice(0,3)
   .map(x=>Object.freeze({index:x.index,id:x.id,label:x.segment.name,
     enabled:Boolean(atBranch)}));
 const progressPercent=Math.round(100*clamp(Number(position)||0,0,1));
 const tool=TOOLS[clinicalMode]||'REFERENCE INSTRUMENT';
 let heading='NAVIGATE';
 let instruction='Use the left stick to advance or retract the original scope.';
 if(atBranch&&children.length){
   heading='AIRWAY JUNCTION';
   instruction='Choose a labelled source airway below. Branch buttons are active.';
 }else if(children.length&&progressPercent>=80){
   heading='APPROACHING BRANCH';
   instruction='Advance gently to the junction; branches activate when reached.';
 }else if(!children.length&&progressPercent>70){
   heading='TERMINAL REFERENCE SEGMENT';
   instruction='Inspect this segment or retract; no additional source branch is recorded.';
 }else if(clinicalMode!=='nav'){
   heading='INSTRUMENT ACTIVE';
   instruction='The selected tool is simulated; inspect its effect in the original procedure.';
 }
 const monitor=monitorStatus==='live'?'LIVE SOURCE VIEW':
  monitorStatus==='unavailable'?'SOURCE MONITOR UNAVAILABLE':
  'CHECK ENDOSCOPIC SOURCE';
 const clearanceText=clearance?.canAdvance===false
   ?'REFERENCE CLEARANCE GATE BLOCKING ADVANCE':
    clearance?.canAdvance===true
    ?'REFERENCE CLEARANCE ESTIMATE ONLY'
    :'NO VERIFIED INSTRUMENT-WALL CLEARANCE';
 return Object.freeze({status:atBranch?'JUNCTION':'TRACKING',
   heading,location:String(segment.name||'Unnamed source segment').slice(0,100),
   generation:Number.isFinite(segment.gen)?segment.gen:null,
   segmentId,progressPercent,instruction,
   branches:Object.freeze(children),canChooseBranch:Boolean(atBranch&&children.length),
   tool,monitor,clearance:clearanceText,
   step:clamp(Math.floor(guideStep)||0,0,4),
   provenance:'REFERENCE SIMULATION · NOT PATIENT-SPECIFIC · UNVALIDATED'});
}
global.PulmoXRProcedureGuide=Object.freeze({describe,route,TOOLS});
})(typeof window!=='undefined'?window:globalThis);
