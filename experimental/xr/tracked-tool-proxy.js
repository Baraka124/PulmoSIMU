/* PulmoSIMU XR17C — strictly visual tracked controller instrument proxy.
 * Uses WebXR right-controller grip pose ONLY, never claims a bronchoscope
 * is physically calibrated. No feedback or effect on original scope.
 */
(function(root){
'use strict';
function create(){
 let enabled=false,tracking=false,observedFrames=0,lostFrames=0,transitions=0;
 function set(value){enabled=Boolean(value);if(!enabled)tracking=false;return snapshot()}
 function observe({rightGripTracked=false,procedure=false,immersiveVR=false}={}){
  const valid=enabled&&Boolean(rightGripTracked)&&Boolean(procedure)&&Boolean(immersiveVR);
  if(tracking!==valid){transitions++;if(tracking&&!valid)lostFrames++}
  tracking=valid;if(tracking)observedFrames++;
  return tracking;
 }
 function snapshot(){return Object.freeze({
  version:'XR17C_CONTROLLER_PROXY_V1',enabled,tracking,observedFrames,
  trackingLossTransitions:lostFrames,stateTransitions:transitions,
  inputSource:'RIGHT_WEBXR_CONTROLLER_GRIP',sensorCalibration:false,
  physicalBronchoscopeConnected:false,physicalToolIdentified:false,
  shaftShape:'ILLUSTRATIVE_150MM_VISUAL_REFERENCE_NOT_A_MEDICAL_DEVICE',
  drivesScope:false,biomechanicalContact:false,realDeviceValidation:false
 })}
 function reset(){enabled=false;tracking=false;observedFrames=0;lostFrames=0;transitions=0;return snapshot()}
 return Object.freeze({set,observe,snapshot,reset});
}
root.PulmoXRTrackedToolProxy=Object.freeze({create});
})(typeof window!=='undefined'?window:globalThis);
