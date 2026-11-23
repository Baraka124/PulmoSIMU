/* PulmoSIMU XR-09 — geometry-backed near contact (Three.js r128).
 * The prior AABB proximity test let controllers grasp EMPTY SPACE inside
 * a lung's bounding box. This module requires an actual triangle hit
 * within the physical contact radius. It is a visual surface interaction,
 * NOT soft-tissue contact mechanics, pressure or haptic force feedback.
 *
 * Sampling rays are emitted only on contact attempts, not every XRFrame.
 * To protect Quest 2, reject over-budget surfaces rather than silently
 * falling back to approximate boxes.
 */
(function(global){
'use strict';
const DIRECTIONS=[];
for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++){
 if(x===0&&y===0&&z===0)continue;
 const d=Math.hypot(x,y,z);
 DIRECTIONS.push([x/d,y/d,z/d]);
}
function contact(THREE,roots,point,{radius=.155,maxMeshes=256}={}){
 if(!THREE||!THREE.Raycaster||!THREE.Box3||!THREE.Vector3)throw Error('XR contact: Three.js missing');
 if(!point?.isVector3||![point.x,point.y,point.z].every(Number.isFinite))
  throw Error('XR contact: invalid tracked pose');
 if(!Number.isFinite(radius)||radius<=0||radius>.3||
    !Number.isInteger(maxMeshes)||maxMeshes<1||maxMeshes>2048)
  throw Error('XR contact: invalid surface budget');
 const meshes=[],bounds=new THREE.Box3();
 const visited=new Set();
 for(const root of roots||[]){
  if(!root?.isObject3D||!root.visible)continue;
  root.traverse(object=>{
   if(meshes.length>maxMeshes||visited.has(object)||!object.isMesh||!object.visible||!object.geometry)return;
   visited.add(object);
   if(object.geometry.boundingBox===null)object.geometry.computeBoundingBox();
   bounds.setFromObject(object);
   // Fast reject before any full mesh intersection.
   if(!bounds.isEmpty()&&bounds.distanceToPoint(point)<=radius)meshes.push(object);
  });
  if(meshes.length>maxMeshes)return Object.freeze({hit:false,reason:'mesh-budget',distanceM:null});
 }
 if(!meshes.length)return Object.freeze({hit:false,reason:'no-near-surface',distanceM:null});
 const ray=new THREE.Raycaster();ray.near=0;ray.far=radius;
 let closest=null;
 for(const d of DIRECTIONS){
  ray.set(point,new THREE.Vector3(d[0],d[1],d[2]));
  const hits=ray.intersectObjects(meshes,false);
  for(const h of hits){
   if(h.distance>radius||h.distance<0||!Number.isFinite(h.distance))continue;
   if(!closest||h.distance<closest.distance)
    closest=h;
  }
 }
 if(!closest)return Object.freeze({hit:false,reason:'no-triangle-contact',distanceM:null});
 return Object.freeze({hit:true,reason:'mesh-surface',distanceM:closest.distance,
  // Stable object identity for selection only; no personal identifiers.
  meshUuid:closest.object.uuid});
}
global.PulmoXRSurface=Object.freeze({contact,directions:DIRECTIONS.length});
})(typeof window!=='undefined'?window:globalThis);
