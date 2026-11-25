/* Local, simulator-only calibrated input. No physical actuator or clinical use.
 * Physical measurements remain local; a held dead-man and fresh calibrated
 * observations are mandatory. Motion is incremental, never replayed on rearm. */
(function(root){
'use strict';
const finite=Number.isFinite,clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
const arc=(x)=>((x+540)%360)-180;
function create({calibration}={}){
 if(!calibration?.snapshot)throw Error('CALIBRATION_REQUIRED');
 let armed=false,prior=null,reason='DISARMED',commands=0;
 function snapshot(){return Object.freeze({armed,reason,commands,
  physicalHardwareValidated:false,clinicalValidated:false,physicalActuation:false,
  inputMode:'CALIBRATED_LOCAL_SIMULATOR_ONLY'})}
 function stop(why='OPERATOR_STOP'){armed=false;prior=null;reason=why;calibration.stop(why);return snapshot()}
 function arm(){calibration.arm();armed=true;prior=null;reason='WAITING_FOR_DEADMAN';return snapshot()}
 function step({now,deadman,reference,procedure,visible,connected,segmentLengthMm,deflectionMax}={}){
  if(!armed)return null;
  if(!connected||!visible||!reference||!procedure){prior=null;
   if(!connected||!visible||!reference)stop('CONTEXT_OR_CONNECTION_LOST');
   return null}
  const state=calibration.snapshot(now);
  if(!state.armed){stop(state.lastError||'CALIBRATION_STOP');return null}
  if(!deadman){prior=null;reason='WAITING_FOR_DEADMAN';return null}
  if(!state.corrected){prior=null;reason='WAITING_FOR_FRESH_SAMPLE';return null}
  const v=state.corrected;
  if(!finite(now)||!finite(segmentLengthMm)||segmentLengthMm<=0||
     !finite(deflectionMax)||deflectionMax<=0){stop('INVALID_SOURCE_SCALE');return null}
  if(!prior){prior={v,now};reason='DEADMAN_BASELINE_SET';return null}
  if(v===prior.v)return null;
  const dt=(now-prior.now)/1000;
  const delta={insertion:v.insertionMm-prior.v.insertionMm,
   roll:arc(v.shaftRollDeg-prior.v.shaftRollDeg),
   pitch:v.tipPitchDeg-prior.v.tipPitchDeg,yaw:v.tipYawDeg-prior.v.tipYawDeg};
  prior={v,now};
  if(dt<=0||dt>.25||Object.values(delta).some(x=>!finite(x))||
     Math.abs(delta.insertion)>Math.min(8,1+80*dt)||
     [delta.roll,delta.pitch,delta.yaw].some(x=>Math.abs(x)>Math.min(15,1+180*dt))){
   stop('MOTION_DISCONTINUITY');return null}
  commands++;reason='SIMULATOR_INPUT_ACTIVE';
  return Object.freeze({insertionT:clamp(delta.insertion/segmentLengthMm,-.08,.08),
   rollRad:delta.roll*Math.PI/180,tipH:delta.yaw/deflectionMax,tipV:delta.pitch/deflectionMax});
 }
 return Object.freeze({arm,stop,step,snapshot});
}
root.PulmoXRCalibratedControl=Object.freeze({create});
})(typeof window!=='undefined'?window:globalThis);
