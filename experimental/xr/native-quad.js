/* XR19 native endoscopy layer. Projection renderer owns stereo; this adapter
 * adds/removes only its quad and copies the original offscreen pixels.
 * No second camera, CPU readback, or claim of physical headset validation. */
(function(root){
'use strict';
function create({transformConstructor=root.XRRigidTransform}={}){
 let session=null,binding=null,projection=null,quad=null,fb=null,gl=null;
 let active=false,attempted=false,reason='NO_SESSION',frames=0;
 function snapshot(){return Object.freeze({compositorQuadActive:active,
  actualDisplayPath:active?'NATIVE_XR_QUAD_LAYER':'SAME_CONTEXT_OFFSCREEN_WEBGL_RENDER_TARGET',
  reason,frames,physicalHeadsetValidated:false})}
 function detach(){
  active=false;
  if(quad){
   const old=quad;
   try{const layers=Array.from(session.renderState.layers||[]).filter(x=>x!==old);
    if(projection&&!layers.includes(projection))layers.unshift(projection);
    session.updateRenderState({layers});}
   catch(e){reason='LAYER_DETACH_FAILED';}
   try{old.destroy?.()}catch{}
   quad=null;
  }
  if(fb){try{gl.deleteFramebuffer(fb)}catch{}fb=null}
 }
 function reset(){detach();session=null;binding=null;projection=null;gl=null;attempted=false;frames=0;reason='NO_SESSION'}
 function bind(renderer){
  reset();session=renderer.xr.getSession();gl=renderer.getContext();
  binding=renderer.xr.getBinding?.();projection=renderer.xr.getBaseLayer?.();
  const grants=session?.enabledFeatures;
  if(!session||!grants||!Array.from(grants).includes('layers')||
     !renderer.capabilities.isWebGL2||!binding?.createQuadLayer||
     !binding?.getSubImage||!projection||session.renderState.baseLayer||
     typeof transformConstructor!=='function'){
   reason='NATIVE_LAYER_REQUIREMENTS_NOT_MET';binding=null;return snapshot();
  }
  reason='READY_FOR_PROCEDURE';return snapshot();
 }
 function update({renderer,frame,target,space,position,orientation,visible}){
  if(!visible){detach();attempted=false;reason=binding?'PROCEDURE_HIDDEN':reason;return false}
  if(!binding||!frame||!space||!target)return false;
  try{
   const pose=new transformConstructor(position,orientation);
   if(!quad){
    if(attempted)return false;attempted=true;
    quad=binding.createQuadLayer({space,transform:pose,width:.94,height:.529,
     viewPixelWidth:1280,viewPixelHeight:720,layout:'mono',textureType:'texture',
     colorFormat:gl.RGBA8,isStatic:false});
    const layers=Array.from(session.renderState.layers||[]);
    if(!layers.includes(projection))layers.unshift(projection);
    session.updateRenderState({layers:[...layers,quad]});
    reason='WAITING_FOR_NATIVE_SUBIMAGE';return false;
   }
   if(!Array.from(session.renderState.layers||[]).includes(quad))return false;
   quad.transform=pose;
   const sub=binding.getSubImage(quad,frame);
   if(!sub?.colorTexture||!sub.viewport)throw Error('NATIVE_SUBIMAGE_UNAVAILABLE');
   const previous=renderer.getRenderTarget(),oldRead=gl.getParameter(gl.READ_FRAMEBUFFER_BINDING),
     oldDraw=gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING),scissor=gl.isEnabled(gl.SCISSOR_TEST);
   try{
    renderer.setRenderTarget(target);
    const source=gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
    if(!source)throw Error('SOURCE_FRAMEBUFFER_UNAVAILABLE');
    fb=fb||gl.createFramebuffer();if(!fb)throw Error('QUAD_FRAMEBUFFER_UNAVAILABLE');
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,fb);
    gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,sub.colorTexture,0);
    if(gl.checkFramebufferStatus(gl.DRAW_FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('QUAD_FRAMEBUFFER_INCOMPLETE');
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER,source);
    if(gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('SOURCE_FRAMEBUFFER_INCOMPLETE');
    const v=sub.viewport;
    gl.disable(gl.SCISSOR_TEST);
    gl.blitFramebuffer(0,0,target.width,target.height,v.x,v.y,v.x+v.width,v.y+v.height,gl.COLOR_BUFFER_BIT,gl.LINEAR);
    if(gl.getError()!==gl.NO_ERROR)throw Error('NATIVE_BLIT_FAILED');
   }finally{
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER,oldRead);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,oldDraw);
    if(scissor)gl.enable(gl.SCISSOR_TEST);else gl.disable(gl.SCISSOR_TEST);
    renderer.setRenderTarget(previous);
   }
   active=true;frames++;reason='NATIVE_SUBIMAGE_UPDATED';return true;
  }catch(e){detach();attempted=true;reason='FALLBACK_'+String(e?.message||e).slice(0,100);return false}
 }
 return Object.freeze({bind,update,snapshot,reset});
}
root.PulmoXRNativeQuad=Object.freeze({create});
})(typeof window!=='undefined'?window:globalThis);
