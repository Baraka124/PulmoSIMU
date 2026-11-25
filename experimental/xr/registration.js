/* XR31: rigid landmark registration, local pose quality and frame observation.
 * No tracking sensor fusion, patient registration or hardware actuation. */
(function(root){
'use strict';
const valid=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),mul=(a,k)=>a.map(v=>v*k),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),norm=a=>Math.hypot(...a),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function basis(p){const a=sub(p[1],p[0]),b=sub(p[2],p[0]),la=norm(a),lb=norm(b);if(Math.min(la,lb)<1)throw Error('LANDMARK_BASELINE_TOO_SHORT');const u=mul(a,1/la),n=cross(u,b),ln=norm(n);if(ln/lb<.15)throw Error('LANDMARKS_NEAR_COLLINEAR');const w=mul(n,1/ln);return [u,cross(w,u),w];}
function register(fit,check,{maxResidualMm=3}={}){
 if(!Array.isArray(fit)||fit.length!==3||!Array.isArray(check)||check.length<2||check.length>100||![...fit,...check].every(p=>valid(p?.sourceMm)&&valid(p?.targetMm))||!Number.isFinite(maxResidualMm)||maxResidualMm<=0)throw Error('INVALID_LANDMARKS');
 const a=fit.map(p=>p.sourceMm),b=fit.map(p=>p.targetMm),A=basis(a),B=basis(b);
 const rotate=p=>B.reduce((s,v,i)=>add(s,mul(v,dot(A[i],p))),[0,0,0]);
 const t=sub(b[0],rotate(a[0])),transform=p=>add(rotate(p),t);
 const errors=rows=>rows.map(p=>norm(sub(transform(p.sourceMm),p.targetMm)));
 const f=errors(fit),e=errors(check),rmse=Math.sqrt(e.reduce((s,v)=>s+v*v,0)/e.length),worst=Math.max(...e);
 return {rotation:[0,1,2].map(i=>rotate([0,1,2].map(j=>i===j?1:0))),translationMm:t,fitMaxMm:Math.max(...f),checkRmseMm:rmse,checkMaxMm:worst,checkCount:e.length,status:Math.max(...f,worst)<=maxResidualMm?'PROVISIONAL_TARGET_MET':'TARGET_NOT_MET',physicalRegistrationValidated:false,hardwareCommand:false};
}
function apply(r,p){if(!valid(p)||r?.status!=='PROVISIONAL_TARGET_MET')throw Error('REGISTRATION_UNAVAILABLE');return r.rotation.reduce((s,v,i)=>add(s,mul(v,p[i])),r.translationMm.slice());}
function createObserver(){
 let prev=null,n=0,losses=0,maxJump=0,stale=0;
 function reset(){prev=null;n=losses=maxJump=stale=0;}
 function step({tracked,timeMs,positionMm}){
  if(!tracked||!valid(positionMm)||!Number.isFinite(timeMs)){if(prev)losses++;prev=null;return snapshot('TRACKING_UNAVAILABLE');}
  if(prev){const dt=timeMs-prev.timeMs;if(dt<=0||dt>250){stale++;prev=null;}else maxJump=Math.max(maxJump,norm(sub(positionMm,prev.p)));}
  prev={timeMs,p:positionMm.slice()};n++;return snapshot('VIRTUAL_POSE_OBSERVED');
 }
 function snapshot(state='NOT_OBSERVED'){return {state,samples:n,trackingLosses:losses,timeDiscontinuities:stale,maxSampleDisplacementMm:maxJump,uncertaintyMm:null,uncertaintyStatus:'NO_INDEPENDENT_TRACKING_CALIBRATION',physicalRegistrationValidated:false,rawPoseExported:false};}
 return {step,reset,snapshot};
}
root.PulmoXRRegistration=Object.freeze({register,apply,createObserver});
})(typeof window!=='undefined'?window:globalThis);
