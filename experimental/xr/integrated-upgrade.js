/* A–G enhancement pass. Reference-model software only; no clinical or hardware actuation. */
(function(root){
'use strict';
const finite=Number.isFinite,clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const valid=p=>Array.isArray(p)&&p.length===3&&p.every(finite);
const norm=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
function inspectionVisibility(ids,selected,mode){
 return ids.map(id=>mode==='isolate'&&selected? id===selected:true);
}
function workspaceTransform({head,forward,right,room,command,step=.35,radius=2.7}){
 if(![head,forward,right,room].every(valid))return null;
 const move={forward:forward,back:forward.map(x=>-x),left:right.map(x=>-x),right}[command];
 if(move){const n=Math.hypot(move[0],move[2]);if(n<.5)return null;
  const d=[move[0]*step/n,0,move[2]*step/n],target=head.map((v,i)=>v+d[i]);
  if(Math.hypot(target[0]-room[0],target[2]-room[2])>radius)return null;
  return {translation:d.map(x=>-x),yaw:0};
 }
 const yaw=command==='turn-left'?-Math.PI/6:command==='turn-right'?Math.PI/6:0;
 if(!yaw)return null;
 // Reference-space transform maps new space into old; rotate about the viewer.
 const cs=Math.cos(yaw),sn=Math.sin(yaw),rot=[cs*head[0]+sn*head[2],head[1],-sn*head[0]+cs*head[2]];
 return {translation:head.map((v,i)=>v-rot[i]),yaw};
}
function grasp({wrist,palm,index,middle,ring,pinky,thumb,held=false}){
 if(![wrist,palm,index,middle,ring,pinky,thumb].every(valid))return false;
 const span=norm(wrist,palm);if(span<.025||span>.12)return false;
 const threshold=held?1.12:.85;
 const curled=[index,middle,ring,pinky].filter(p=>norm(p,palm)/span<threshold).length;
 return curled>=3&&norm(thumb,palm)/span<(held?1.55:1.35);
}
function createTrackingQuality(){
 let previous=null,stable=0,samples=0,rejected=0,lost=0,largest=0;
 function observe(point){
  if(!valid(point)){if(previous)lost++;previous=null;stable=0;return snapshot();}
  if(previous){const d=norm(point,previous);largest=Math.max(largest,d);stable=d<.025?stable+1:0;if(d>.15)rejected++;}
  previous=point.slice();samples++;return snapshot();
 }
 function snapshot(){return {samples,stableIntervals:stable,trackingInterruptions:lost,largeDiscontinuities:rejected,maxSampleChangeM:samples>1?largest:null,stableCandidate:stable>=3,absoluteAccuracyMeasured:false};}
 return {observe,snapshot,reset(){previous=null;stable=samples=rejected=lost=largest=0;}};
}
function accessoryClearance({lengthMm,diameterMm,worldPerMm,toSource,field,sourceMmPerUnit,open=false}){
 if(!finite(lengthMm)||lengthMm<0||lengthMm>12||!(diameterMm>0)||!(worldPerMm>0)||!(sourceMmPerUnit>0)||!toSource||!field)return {state:'UNAVAILABLE',allowed:false,marginMm:null};
 const radius=diameterMm/2+(open?.65:0),end=lengthMm+(open?2:0);
 let margin=Infinity;
 // Check the geometric channel axis at 0.5 mm spacing, matching accessory offset.
 for(let s=0;s<=end+.001;s+=.5){
  const p=toSource([1.05*worldPerMm,-.9*worldPerMm,-Math.min(s,end)*worldPerMm]);
  const f=valid(p)?field(p):NaN;if(!finite(f))return {state:'UNAVAILABLE',allowed:false,marginMm:null};
  margin=Math.min(margin,-f*sourceMmPerUnit-radius);
 }
 return {state:margin>=0?'GEOMETRIC_CLEAR':'REFERENCE_WALL_INTERFERENCE',allowed:margin>=0,marginMm:margin,forceMeasured:false};
}
function createCouplingEvidence(){
 let samples=0,skipped=0,work=0,maxResidual=0,minVolume=Infinity,maxVolume=-Infinity,minScale=Infinity,maxScale=-Infinity;
 function step(p,m,dt){
  if(!p||!m||![p.pressurePa,p.flowM3PerS,p.volumeM3,p.conservationResidualM3PerS,m.scale,dt].every(finite)||dt<=0||dt>.1){skipped++;return snapshot();}
  samples++;work+=p.pressurePa*p.flowM3PerS*dt;maxResidual=Math.max(maxResidual,Math.abs(p.conservationResidualM3PerS));
  minVolume=Math.min(minVolume,p.volumeM3);maxVolume=Math.max(maxVolume,p.volumeM3);minScale=Math.min(minScale,m.scale);maxScale=Math.max(maxScale,m.scale);
  return snapshot();
 }
 function snapshot(){return {samples,skipped,referencePressureFlowIntegralJ:samples?work:null,maxConservationResidualM3PerS:samples?maxResidual:null,volumeRangeM3:samples?[minVolume,maxVolume]:null,displayScaleRange:samples?[minScale,maxScale]:null,coupling:'ONE_WAY_R3_TO_REFERENCE_GEOMETRY',measuredMechanicalWork:false,feedbackSolved:false};}
 return {step,snapshot,reset(){samples=skipped=work=maxResidual=0;minVolume=minScale=Infinity;maxVolume=maxScale=-Infinity;}};
}
root.PulmoXRIntegratedUpgrade=Object.freeze({inspectionVisibility,workspaceTransform,grasp,createTrackingQuality,accessoryClearance,createCouplingEvidence});
})(typeof window!=='undefined'?window:globalThis);
