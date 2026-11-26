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
// Virtual-only obstacle path test. No claim of physical Guardian/room geometry.
function clearWorkspacePath(start,end,obstacles=[],buffer=.18){
 if(!valid(start)||!valid(end)||!finite(buffer)||buffer<0)return false;
 for(const o of obstacles){if(!valid(o?.center)||!finite(o.radius)||o.radius<0)return false;
  const a=[start[0],start[2]],b=[end[0],end[2]],p=[o.center[0],o.center[2]],d=b.map((v,i)=>v-a[i]);
  const len=d[0]**2+d[1]**2,t=len?clamp(((p[0]-a[0])*d[0]+(p[1]-a[1])*d[1])/len,0,1):0;
  const nearest=Math.hypot(a[0]+t*d[0]-p[0],a[1]+t*d[1]-p[1]),initial=Math.hypot(a[0]-p[0],a[1]-p[1]),final=Math.hypot(b[0]-p[0],b[1]-p[1]);
  if(nearest<o.radius+buffer&&!(initial<o.radius+buffer&&final>initial&&nearest>=initial-1e-6))return false;
 }return true;
}
function targetOffset({head,target,room,radius=2.7,obstacles=[]}){
 if(![head,target,room].every(valid)||Math.hypot(target[0]-room[0],target[2]-room[2])>radius||!clearWorkspacePath(head,target,obstacles))return null;
 return {translation:[head[0]-target[0],0,head[2]-target[2]],yaw:0};
}
function createRegionalEvidence(){
 let previous=null,regions={},accepted=0,gaps=0;
 function step(p){
  const t=p?.tSeconds,flows=p?.regionalFlowM3PerS;
  if(!finite(t)||!flows||!Object.keys(flows).length||!Object.values(flows).every(finite)){previous=null;return snapshot();}
  if(previous){const dt=t-previous.t;
   if(dt>0&&dt<=.15&&Object.keys(flows).every(k=>finite(previous.flows[k]))){
    for(const [k,q] of Object.entries(flows)){const mean=(q+previous.flows[k])*.5;const a=regions[k]||(regions[k]={inspiredM3:0,expiredM3:0,netM3:0,peakAbsFlowM3PerS:0});a.netM3+=mean*dt;a.inspiredM3+=Math.max(0,mean)*dt;a.expiredM3+=Math.max(0,-mean)*dt;a.peakAbsFlowM3PerS=Math.max(a.peakAbsFlowM3PerS,Math.abs(q));}accepted++;
   }else if(dt!==0)gaps++;
  }previous={t,flows:{...flows}};return snapshot();
 }
 function snapshot(){return {acceptedIntervals:accepted,gaps,regions:JSON.parse(JSON.stringify(regions)),source:'R3_REGIONAL_FLOW_OBSERVATION',regionalTissueDeformationSolved:false};}
 return {step,snapshot,suspend(){previous=null},reset(){previous=null;regions={};accepted=gaps=0;}};
}
// Reference-space display constraints, not an inferred physical room.
function boundedPanelPosition(head,target,{minDistance=.45,maxDistance=2.1,maxVertical=.85}={}){
 if(!valid(head)||!valid(target))return null;
 const d=target.map((v,i)=>v-head[i]);d[1]=clamp(d[1],-maxVertical,maxVertical);const n=Math.hypot(...d);
 if(n<.001)return null;const s=clamp(n,minDistance,maxDistance)/n;
 return head.map((v,i)=>v+d[i]*s);
}
function limitedStep(previous,target,dt,maxSpeed=1.4){
 if(!valid(previous)||!valid(target)||!finite(dt)||dt<=0||dt>.12||!finite(maxSpeed)||maxSpeed<=0)return null;
 const d=target.map((v,i)=>v-previous[i]),n=Math.hypot(...d);return n>0?previous.map((v,i)=>v+d[i]*Math.min(1,maxSpeed*dt/n)):previous.slice();
}
function observeEnvironment(frame){
 const features={planes:'UNAVAILABLE',meshes:'UNAVAILABLE',light:'NOT_QUERIED'};
 try{if(frame?.detectedPlanes&&typeof frame.detectedPlanes.size==='number')features.planes=frame.detectedPlanes.size>0?'OBSERVED':'EMPTY';}catch{features.planes='QUERY_FAILED';}
 try{if(frame?.detectedMeshes&&typeof frame.detectedMeshes.size==='number')features.meshes=frame.detectedMeshes.size>0?'OBSERVED':'EMPTY';}catch{features.meshes='QUERY_FAILED';}
 return {...features,roomGeometryStored:false,physicalOcclusion:false};
}
function contactEstimate(marginMm,diameterMm){
 if(!finite(marginMm)||!finite(diameterMm)||diameterMm<=0)return {state:'UNAVAILABLE',normalizedInterference:null};
 return {state:marginMm<0?'GEOMETRIC_OVERLAP':'GEOMETRIC_CLEAR',normalizedInterference:clamp(-marginMm/diameterMm,0,1),forceN:null,tissueDeformed:false};
}
function regionalDisplayScale(netM3,baselineM3){if(!finite(netM3)||!finite(baselineM3)||baselineM3<=0)return null;return clamp(Math.cbrt(Math.max(.001,1+netM3/baselineM3)),.96,1.08);}
root.PulmoXRIntegratedUpgrade=Object.freeze({inspectionVisibility,workspaceTransform,grasp,createTrackingQuality,accessoryClearance,createCouplingEvidence,clearWorkspacePath,targetOffset,createRegionalEvidence,boundedPanelPosition,limitedStep,observeEnvironment,contactEstimate,regionalDisplayScale});
})(typeof window!=='undefined'?window:globalThis);
