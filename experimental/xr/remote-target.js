/* PulmoSIMU XR-10 — triangle-accurate distance selection of the
 * ORIGINAL anatomical meshes. No boxes, proxy lung or generated anatomy.
 * Classic script compatible with Three.js r128.
 */
(function(global){
 'use strict';
 function pick(THREE,pointer,roots,{maxDistance=3.5}={}){
  if(!THREE?.Raycaster||!THREE?.Vector3||!THREE?.Matrix4||
    !pointer?.isObject3D||!Array.isArray(roots)||
    !Number.isFinite(maxDistance)||maxDistance<=0||maxDistance>5){
   return Object.freeze({hit:false,reason:'invalid-ray-input'});
  }
  const allowed=roots.filter(root=>root?.isObject3D&&root.visible);
  if(!allowed.length)return Object.freeze({hit:false,reason:'no-anatomy-source'});
  pointer.updateWorldMatrix(true,false);
  allowed.forEach(root=>root.updateWorldMatrix(true,true));
  const m=pointer.matrixWorld;
  const origin=new THREE.Vector3().setFromMatrixPosition(m);
  const orientation=new THREE.Matrix4().extractRotation(m);
  const direction=new THREE.Vector3(0,0,-1).applyMatrix4(orientation).normalize();
  if(![origin.x,origin.y,origin.z,direction.x,direction.y,direction.z].every(Number.isFinite))
   return Object.freeze({hit:false,reason:'invalid-pointer-transform'});
  const caster=new THREE.Raycaster(origin,direction,0.015,maxDistance);
  const hits=caster.intersectObjects(allowed,true);
  // Only actual mesh triangles; do not select bounding boxes, lines or panels.
  const first=hits.find(h=>h.object?.isMesh&&h.object.visible&&
    Number.isFinite(h.distance)&&h.distance<=maxDistance);
  if(!first)return Object.freeze({hit:false,reason:'ray-missed-anatomy'});
  return Object.freeze({hit:true,reason:'triangle-ray',distanceM:first.distance,
    point:first.point.clone(),object:first.object});
 }
 global.PulmoXRRemoteTarget=Object.freeze({pick});
})(typeof window!=='undefined'?window:globalThis);
