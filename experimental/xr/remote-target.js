/* PulmoSIMU XR-11 — triangle-accurate distance selection of the
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
  // Original procedural/CT materials may have inconsistent triangle winding.
  // Double-sided *ray testing* finds the nearest real surface; the original
  // source materials are restored synchronously and NEVER repainted.
  const overridden=[],seen=new Set();
  for(const root of allowed)root.traverse(obj=>{
    if(!obj.isMesh||!obj.visible)return;
    for(const mat of (Array.isArray(obj.material)?obj.material:[obj.material])){
      if(!mat||seen.has(mat))continue;
      seen.add(mat);
      if(mat.side!==THREE.DoubleSide){
        overridden.push([mat,mat.side]);
        mat.side=THREE.DoubleSide;
      }
    }
  });
  let hits;
  try{hits=caster.intersectObjects(allowed,true)}
  finally{for(const [mat,side]of overridden)mat.side=side}
  // Only actual mesh triangles, from a visible chain, never simulated proxy
  // boxes, menu surfaces or invisible anatomy.
  const first=hits.find(h=>{
    if(!h.object?.isMesh||!Number.isFinite(h.distance)||h.distance>maxDistance)return false;
    for(let obj=h.object;obj;obj=obj.parent)if(!obj.visible)return false;
    return true;
  });
  if(!first)return Object.freeze({hit:false,reason:'ray-missed-anatomy'});
  return Object.freeze({hit:true,reason:'triangle-ray',distanceM:first.distance,
    point:first.point.clone(),object:first.object});
 }
 global.PulmoXRRemoteTarget=Object.freeze({pick});
})(typeof window!=='undefined'?window:globalThis);
