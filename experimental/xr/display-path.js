/* PulmoSIMU XR-17C — evidence-based immersive display-path negotiation.
 * Three.js r128 currently owns XRWebGLLayer/baseLayer. Replacing its
 * framebuffer with a compositor quad layer without a compatible projection
 * renderer would black-screen the headset. Therefore this module never
 * mutates XRSession.updateRenderState or WebGL framebuffer bindings.
 * A genuine compositor path is reported as a separate future integration gate.
 */
(function(root){
'use strict';
function create({bindingConstructor=null}={}){
 let result=null,checks=0;
 function inspect({session=null,renderer=null,gl=null}={}){
  checks++;
  const features=session?.enabledFeatures;
  let enabled=null;
  if(features&&typeof features[Symbol.iterator]==='function'){
   try{enabled=new Set(Array.from(features,x=>String(x)))}catch{}
  }
  const requested=Boolean(enabled?.has('layers'));
  const bindingAvailable=typeof bindingConstructor==='function';
  const legacyBaseLayer=Boolean(session?.renderState?.baseLayer);
  const currentProjectionLayer=Boolean(session?.renderState?.layers?.some?.(layer=>
    layer?.constructor?.name==='XRProjectionLayer'));
  // Never assume user-agent support from XRWebGLBinding presence alone.
  // The next renderer generation needs explicit projection-layer integration,
  // a compatible GL context and stereo frame-target rendering.
  const available=Boolean(session&&renderer&&gl&&requested&&bindingAvailable);
  result=Object.freeze({
   version:'XR17C_DISPLAY_PATH_V1',
   requestedLayers:requested,featureReporting:enabled?'runtime-grants':'not-reported',
   bindingAvailable,legacyBaseLayer,currentProjectionLayer,
   quadEligibility:available?'COMPOSITOR_CAPABILITY_PRESENT':'NOT_PROVEN',
   actualDisplayPath:'SAME_CONTEXT_OFFSCREEN_WEBGL_RENDER_TARGET',
   compositorQuadActive:false,
   reason:!session?'NO_SESSION':
     !enabled?'SESSION_FEATURES_NOT_REPORTED':
     !requested?'LAYERS_NOT_GRANTED':
     !bindingAvailable?'XR_WEBGL_BINDING_UNAVAILABLE':
     legacyBaseLayer?'THREE_R128_BASE_LAYER_OWNS_RENDERER':
     !currentProjectionLayer?'PROJECTION_LAYER_NOT_INTEGRATED':
     'RENDERER_COMPOSITOR_ADAPTER_NOT_VALIDATED',
   safeFallback:true,
   safety:'DO_NOT_REPLACE_XR_RENDER_STATE_WITHOUT_STEREO_PROJECTION_ADAPTER'
  });
  return result;
 }
 function snapshot(){return Object.freeze(result||{
  version:'XR17C_DISPLAY_PATH_V1',actualDisplayPath:'SAME_CONTEXT_OFFSCREEN_WEBGL_RENDER_TARGET',
  compositorQuadActive:false,quadEligibility:'NOT_CHECKED',reason:'NO_SESSION',
  safeFallback:true,checks
 });}
 function reset(){result=null}
 return Object.freeze({inspect,snapshot,reset});
}
root.PulmoXRDisplayPath=Object.freeze({create});
})(typeof window!=='undefined'?window:globalThis);
