/* PulmoSIMU XR-17B — physical MR placement lifecycle.
 * One original scene; actual WebXR hit-test results only.
 * No fabricated plane, mesh reconstruction, depth map, persistence or clinical
 * registration. Anchoring is optional and never assumed from passthrough.
 */
(function(root){
'use strict';
const finite3=p=>p&&[p.x,p.y,p.z].every(Number.isFinite)&&
  Math.max(Math.abs(p.x),Math.abs(p.y),Math.abs(p.z))<100;
const coordinate=p=>Object.freeze({x:p.x,y:p.y,z:p.z});
function create(){
 let session=null,source=null,anchor=null,lastHit=null,lastResult=null,origin=null;
 let candidate=null,placement=null,mode='inactive',state='idle';
 let error=null,anchorStatus='none',pendingConfirm=false,epoch=0;
 let confirmations=0,repositions=0,hitFrames=0,trackingLost=0;
 function removeAnchor(){
  const existing=anchor;anchor=null;
  try{existing?.delete?.()}catch{}
 }
 function reset(){
  ++epoch;
  try{source?.cancel?.()}catch{}
  source=null;removeAnchor();lastHit=null;lastResult=null;candidate=null;
  placement=null;origin=null;pendingConfirm=false;mode='inactive';
  state='idle';anchorStatus='none';error=null;
  confirmations=0;repositions=0;hitFrames=0;trackingLost=0;session=null;
 }
 function snapshot(){
  // Only capability/quality status, not private room positions or device poses.
  return Object.freeze({
   version:'XR17B_MR_PLACEMENT_V1',mode,state,error,anchorStatus,
   hitTestActive:Boolean(source),hitAvailable:Boolean(candidate),
   userConfirmed:Boolean(placement),unregisteredPreview:state==='preview',
   placementCount:confirmations,repositions,hitFrames,trackingLost,
   physicalDepthOcclusion:false,roomMeshReconstruction:false,
   crossSessionAnchorPersistence:false,patientCTRegistration:false
  });
 }
 async function start(xrSession){
  reset();session=xrSession;mode='mr';state='initializing';
  const token=epoch;
  if(!session||typeof session.requestReferenceSpace!=='function'||
     typeof session.requestHitTestSource!=='function'){
    state='unsupported';error='HIT_TEST_API_UNAVAILABLE';return snapshot();
  }
  try{
   // Viewer-space rays target the real environment around the headset.
   const viewer=await session.requestReferenceSpace('viewer');
   if(token!==epoch)return snapshot();
   const acquired=await session.requestHitTestSource({space:viewer});
   if(token!==epoch){try{acquired?.cancel?.()}catch{};return snapshot()}
   if(!acquired)throw Error('NO_HIT_TEST_SOURCE');
   source=acquired;state='scanning';
  }catch(e){
   if(token===epoch){state='unsupported';error=String(e?.message||e).slice(0,140)}
  }
  return snapshot();
 }
 function scan(frame,referenceSpace){
  if(mode!=='mr'||!frame||!referenceSpace)return null;
  if(anchor&&placement&&state==='placed'){
   try{
    const p=frame.getPose(anchor.anchorSpace,referenceSpace)?.transform?.position;
    if(finite3(p)){
     const next=coordinate(p);
     const prev=origin||next;
     const delta={x:next.x-prev.x,y:next.y-prev.y,z:next.z-prev.z};
     origin=next;anchorStatus='tracking';
     // Return only incremental translation; consumers preserve manual grabs.
     return {kind:'anchor-delta',delta};
    }
    if(anchorStatus==='tracking')trackingLost++;
    anchorStatus='tracking-lost';
   }catch(e){
    anchorStatus='tracking-lost';error=String(e?.message||e).slice(0,140);
   }
   return null;
  }
  if(state!=='scanning'&&state!=='confirm-pending')return null;
  if(!source||typeof frame.getHitTestResults!=='function'){
   candidate=null;lastResult=null;return null;
  }
  try{
   const hit=frame.getHitTestResults(source)?.[0]||null;
   const p=hit?.getPose?.(referenceSpace)?.transform?.position;
   if(!finite3(p)){
    candidate=null;lastResult=null;lastHit=null;
    if(pendingConfirm){pendingConfirm=false;state='scanning'}
    return null;
   }
   candidate=coordinate(p);lastResult=hit;hitFrames++;
   if(pendingConfirm){
    pendingConfirm=false;placement=candidate;origin=candidate;
    confirmations++;state='placed';anchorStatus='not-anchored';
    const token=epoch;
    if(typeof hit.createAnchor==='function'){
     anchorStatus='anchor-requested';
     try{
      // Called inside the XR animation frame. No automatic physical
      // registration is claimed until a real anchor is returned.
      Promise.resolve(hit.createAnchor()).then(a=>{
       if(token!==epoch||state!=='placed'){try{a?.delete?.()}catch{};return}
       if(a?.anchorSpace){anchor=a;anchorStatus='anchor-acquired'}
       else anchorStatus='not-supported';
      }).catch(e=>{
       if(token===epoch){anchorStatus='not-supported';error=String(e?.message||e).slice(0,140)}
      });
     }catch(e){anchorStatus='not-supported';error=String(e?.message||e).slice(0,140)}
    }
    return {kind:'placed',point:placement};
   }
   return {kind:'candidate',point:candidate};
  }catch(e){
   candidate=null;lastResult=null;error=String(e?.message||e).slice(0,140);
   return null;
  }
 }
 function confirm(){
  if(state!=='scanning'||!candidate||!lastResult)
   return Object.freeze({accepted:false,reason:'NO_CURRENT_SURFACE_HIT'});
  pendingConfirm=true;state='confirm-pending';
  return Object.freeze({accepted:true,reason:'CONFIRM_ON_NEXT_XR_FRAME'});
 }
 function reposition(){
  if(mode!=='mr')return false;
  ++epoch;repositions++;removeAnchor();
  placement=null;candidate=null;lastResult=null;lastHit=null;origin=null;
  pendingConfirm=false;anchorStatus='none';
  state=source?'scanning':'unsupported';
  return Boolean(source);
 }
 function preview(){
  if(mode!=='mr')return false;
  ++epoch;removeAnchor();placement=null;candidate=null;origin=null;
  pendingConfirm=false;state='preview';anchorStatus='none';
  return true;
 }
 return Object.freeze({start,scan,confirm,reposition,preview,reset,snapshot});
}
root.PulmoXRMRPlacement=Object.freeze({create});
})(typeof window!=='undefined'?window:globalThis);
