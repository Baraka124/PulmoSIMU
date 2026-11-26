(function hapticLab(root){
'use strict';
const finite=Number.isFinite,clip=(x,a,b)=>Math.min(b,Math.max(a,x));
function estimate({shaft,reference,tracking,procedure,marginAgeMs,advancing=false,blocked=false}={}){
 if(!reference||!tracking||!procedure||!finite(marginAgeMs)||marginAgeMs<0||marginAgeMs>250||!finite(shaft?.minMarginMm))return {state:'UNAVAILABLE',risk:null,marginMm:null,physicalForceN:null,actuation:false};
 const margin=shaft.minMarginMm,near=clip((.8-margin)/.8,0,1),interference=clip(-margin/2,0,1),bend=finite(shaft.maxCurvature)?clip(shaft.maxCurvature/.055-1,0,1):0;
 const risk=clip(Math.max(near*.45,interference,bend*.4,blocked&&advancing?.65:0),0,1);
 return {state:margin<0?'SAMPLED_INTERFERENCE':blocked&&advancing?'ADVANCE_BLOCKED':risk>0?'NEAR_LIMIT':'CLEAR_SAMPLE',risk,marginMm:margin,curvaturePerMm:finite(shaft.maxCurvature)?shaft.maxCurvature:null,advancing:Boolean(advancing),physicalForceN:null,actuation:false,calibrated:false};
}
function createCue(){
 let enabled=false,last=-Infinity,count=0;
 return Object.freeze({set(v){enabled=!!v;last=-Infinity;},reset(){enabled=false;last=-Infinity;count=0;},step({now,evidence,activeGrip,allowed}){if(!enabled||!allowed||!activeGrip||!finite(now)||now-last<350||!finite(evidence?.risk)||!evidence.advancing||evidence.risk<.25)return null;last=now;count++;return {intensity:clip(.12+.36*evidence.risk,.12,.48),durationMs:35,kind:'DISCRETE_GEOMETRIC_CUE'};},snapshot:()=>({enabled,requestedCues:count,forceFeedback:false,deliveryVerified:false})});
}
function createObserver(parseFrame){
 let last=null,force=null,seq=-1,fseq=-1,lastAt=-Infinity,forceAt=-Infinity,accepted=0,rejected=0,forceAccepted=0,forceRejected=0;
 function reset(){last=null;force=null;seq=fseq=-1;lastAt=forceAt=-Infinity;accepted=rejected=forceAccepted=forceRejected=0;}
 function ingest(line,now){
  if(!finite(now)||now<0||typeof line!=='string'||line.length>512){rejected++;return {accepted:false,kind:'motion',reason:'INVALID_FRAME'};}
  let obj;try{obj=JSON.parse(line);}catch{rejected++;return {accepted:false,kind:'motion',reason:'INVALID_JSON'};}
  if(obj?.protocol==='PULMOSIMU-LOAD/1'){
   if(!Number.isSafeInteger(obj.seq)||obj.seq<=fseq||now<forceAt||!finite(obj.axialForceN)||Math.abs(obj.axialForceN)>100||typeof obj.calibrationId!=='string'||!/^[A-Za-z0-9_-]{6,64}$/.test(obj.calibrationId)){forceRejected++;force=null;return {accepted:false,kind:'load',reason:'INVALID_LOAD_FRAME'};}
   force={axialForceN:obj.axialForceN,calibrationId:obj.calibrationId};fseq=obj.seq;forceAt=now;forceAccepted++;return {accepted:true,kind:'load'};
  }
  try{const p=parseFrame(line);if(p.seq<=seq||now<lastAt)throw Error('NON_MONOTONIC_SAMPLE');last=p;seq=p.seq;lastAt=now;accepted++;return {accepted:true,kind:'motion'};}catch(e){last=null;rejected++;return {accepted:false,kind:'motion',reason:String(e.message)};}
 }
 function snapshot(now){const age=now-lastAt,fage=now-forceAt,fresh=!!last&&finite(age)&&age>=0&&age<=250,forceFresh=!!force&&finite(fage)&&fage>=0&&fage<=250;return {state:fresh?'FRESH_MOTION':last?'STALE_MOTION':'NO_MOTION',motion:fresh?{insertionMm:last.insertionMm,shaftRollDeg:last.shaftRollDeg,tipPitchDeg:last.tipPitchDeg,tipYawDeg:last.tipYawDeg}:null,reportedAxialForceN:forceFresh?force.axialForceN:null,loadCalibrationId:forceFresh?force.calibrationId:null,loadState:forceFresh?'DEVICE_REPORTED_UNVERIFIED':force?'STALE':'NO_LOAD',motionAgeMs:fresh?age:null,accepted,rejected,forceAccepted,forceRejected,physicalMetrologyCertified:false,physicalActuation:false};}
 return Object.freeze({ingest,snapshot,reset});
}
function createDryRun(){
 let armed=false,releaseSeen=false,last=-Infinity,frames=0,reason='DISARMED',index=null;
 function stop(why='OPERATOR_STOP'){armed=false;releaseSeen=false;index=null;reason=why;}
 function arm(){armed=true;releaseSeen=false;last=-Infinity;frames=0;index=null;reason='RELEASE_DEADMAN';}
 function step({now,reference,tracking,procedure,connected,calibrated,fresh,deadman,eligible,evidence}){
  if(!armed)return null;
  if(![reference,tracking,procedure,connected,calibrated,fresh,eligible].every(Boolean)||!finite(now)||!finite(evidence?.risk)||now<last){stop('CONTEXT_OR_SAMPLE_LOST');return null;}
  if(!deadman){releaseSeen=true;last=now;index=null;reason='WAITING_FOR_DEADMAN';return null;}
  if(!releaseSeen){index=null;reason='RELEASE_DEADMAN';return null;}
  if(now-last>250){stop('WATCHDOG_TIMEOUT');return null;}
  if(now-last<50)return null;
  last=now;frames++;index=clip(evidence.risk,0,1);reason='LOCAL_DRY_RUN_ONLY';return {normalizedCue:index,physicalCommand:false,transport:'NONE'};
 }
 return Object.freeze({arm,stop,step,snapshot:()=>({armed,reason,frames,normalizedCue:index,physicalOutput:false,transport:'NONE',activeForceFeedback:false})});
}
root.PulmoXRHapticLab=Object.freeze({estimate,createCue,createObserver,createDryRun});
})(typeof window!=='undefined'?window:globalThis);
