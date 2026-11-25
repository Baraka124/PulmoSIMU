/* XR29 reference-only R3 volume-driven isotropic deformation.
 * The 2.5 L baseline is an assumed numerical volume, not measured FRC.
 * One affine transform for the reference mesh and camera coordinates.
 * No regional tissue motion, CFD, changed resistance or measured 4D CT.
 */
(function(root){
'use strict';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function scaleFor(volumeM3,baselineM3=.0025){
 if(!Number.isFinite(volumeM3)||!Number.isFinite(baselineM3)||baselineM3<=0)throw Error('INVALID_VOLUME');
 return clamp(Math.cbrt(Math.max(.001,1+volumeM3/baselineM3)),.96,1.08);
}
function create({baselineM3=.0025}={}){
 let scale=1,min=1,max=1,samples=0,lastVolume=null;
 function step(physics,enabled,dt){
  const valid=enabled&&Number.isFinite(physics?.volumeM3)&&Number.isFinite(dt)&&dt>0&&dt<=.25;
  if(!valid){scale=1;return snapshot(false);}
  lastVolume=physics.volumeM3;const target=scaleFor(lastVolume,baselineM3);
  scale+=(target-scale)*(1-Math.exp(-dt/.18));min=Math.min(min,scale);max=Math.max(max,scale);samples++;
  return snapshot(true);
 }
 function snapshot(active=false){return {active,scale,minScale:min,maxScale:max,samples,volumeM3:lastVolume,baselineM3,model:'R3_DRIVEN_AFFINE_REFERENCE',patientSpecific:false,tissueSolved:false,feedbackToR3:false};}
 function reset(){scale=min=max=1;samples=0;lastVolume=null;return snapshot();}
 return {step,snapshot,reset};
}
root.PulmoXRRespiratoryMotion=Object.freeze({scaleFor,create});
})(typeof window!=='undefined'?window:globalThis);
