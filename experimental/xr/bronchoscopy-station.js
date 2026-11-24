/* PulmoSIMU XR-16 — procedure-station ergonomics and bounded instrument input.
 * Simulation-only, original SEGS and camScope remain the data source.
 * Pure numerical contracts are shared by Quest runtime and Node tests.
 */
(function(root){
'use strict';
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
const LAYOUTS=Object.freeze({
 stacked:Object.freeze({
  monitor:Object.freeze({depth:1.24,right:.72,y:.13,scale:1.06}),
  guide:Object.freeze({depth:1.24,right:.72,y:-.56,scale:.98}),
  label:'STACKED'}),
 side:Object.freeze({
  monitor:Object.freeze({depth:1.24,right:.23,y:.05,scale:1.03}),
  guide:Object.freeze({depth:1.25,right:1.18,y:.03,scale:.95}),
  label:'SIDE BY SIDE'}),
 focus:Object.freeze({
  monitor:Object.freeze({depth:1.19,right:.32,y:.04,scale:1.28}),
  guide:Object.freeze({depth:1.27,right:1.25,y:-.25,scale:.85}),
  label:'ENDOSCOPY FOCUS'})
});
const VIDEO=Object.freeze({
 balanced:Object.freeze({width:800,height:450,intervalMs:55,label:'BALANCED'}),
 detail:Object.freeze({width:1024,height:576,intervalMs:42,label:'DETAIL'}),
 economy:Object.freeze({width:640,height:360,intervalMs:90,label:'ECONOMY'})
});
const OPTICS=Object.freeze({standard:100,wide:110,detail:88});
const keys=Object.freeze({layout:['stacked','side','focus'],
 video:['balanced','detail','economy'],optics:['standard','wide','detail'],
 control:['standard','fine']});
function make(){
 const state={layout:'stacked',video:'balanced',optics:'standard',control:'standard'};
 function next(which){
  if(!keys[which])throw Error('Unknown station control');
  const choices=keys[which];
  state[which]=choices[(choices.indexOf(state[which])+1)%choices.length];
  return snapshot();
 }
 function snapshot(){return Object.freeze({
  ...state,pose:LAYOUTS[state.layout],capture:VIDEO[state.video],
  fovDegrees:OPTICS[state.optics],
  insertionMultiplier:state.control==='fine'?.42:1
 })}
 return Object.freeze({next,snapshot,reset(){
   Object.assign(state,{layout:'stacked',video:'balanced',optics:'standard',control:'standard'});
   return snapshot();
 }});
}
function tipStep(current,axis,dt){
 const step=clamp(Number(dt)||0,0,.06);
 const demand=Math.abs(axis)<.13?0:clamp(axis,-1,1);
 // Relative, bounded motor command. Releasing the stick holds the last tip
 // angle, unlike spring-to-centre camera controls; not a calibrated scope.
 return clamp(current+demand*step*1.16,-1,1);
}
function rollStep(current,axis,dt){
 const step=clamp(Number(dt)||0,0,.06);
 const demand=Math.abs(axis)<.13?0:clamp(axis,-1,1);
 // Winding in radians; separate from insertion and distal tip steering.
 const next=current+demand*step*1.45;
 return Math.atan2(Math.sin(next),Math.cos(next));
}
function velocity(axis,scopeSpeed,multiplier=1){
 if(Math.abs(axis)<.13)return 0;
 return -clamp(axis,-1,1)*clamp(Number(scopeSpeed)||0,0,60)/1500*
   clamp(Number(multiplier)||1,.2,1);
}
function rateLimit(current,target,dt,rate=2.0){
 const s=clamp(Number(dt)||0,0,.06)*rate;
 return clamp(target,current-s,current+s);
}
root.PulmoXRBronchoscopyStation=Object.freeze({
 create:make,LAYOUTS,VIDEO,OPTICS,tipStep,rollStep,velocity,rateLimit
});
})(typeof window!=='undefined'?window:globalThis);
