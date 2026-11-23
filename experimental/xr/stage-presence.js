/* PulmoSIMU XR-13 · Editorial spatial observatory architecture.
 * Independent of patient/source anatomy, bronchoscopy, CT, and physics.
 * Uses low-draw-call world references instead of mock anatomy or overlays.
 * Compatible with the existing Three.js r128 classic-script renderer.
 */
(function(global){
'use strict';
function create(THREE,room,{floor=null,grid=null,plinth=null,rings=[]}={}){
  if(!THREE?.LineSegments||!THREE?.DirectionalLight||!room?.isObject3D)
    throw new Error('XR-13 requires the original Three.js room');
  const architectural=new THREE.Group();
  architectural.name='xr-spatial-architecture';room.add(architectural);
  // Unified single-draw-call vertical architectural datum lines.
  // Their function is room-scale parallax and orientation, never patient axes.
  const vertices=[];
  const columns=12, radius=3.45;
  for(let i=0;i<columns;i++){
    const a=i*Math.PI*2/columns;
    const x=Math.sin(a)*radius,z=Math.cos(a)*radius;
    vertices.push(x,.04,z,x,2.56,z);
    const n=(i+1)*Math.PI*2/columns;
    vertices.push(x,2.56,z,Math.sin(n)*radius,2.56,Math.cos(n)*radius);
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
  const boundaries=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({
    color:0x66808b,transparent:true,opacity:.115,depthWrite:false
  }));
  boundaries.name='xr-architectural-datums';architectural.add(boundaries);
  // Restrained real scene lighting, not post-processing, bloom or synthetic
  // physiology. Local room support can be toggled without changing anatomy.
  const sky=new THREE.HemisphereLight(0xa1b7c4,0x111820,.23);
  sky.name='xr-observatory-ambient';architectural.add(sky);
  const key=new THREE.DirectionalLight(0xd7e4eb,.34);
  key.position.set(-2.4,3.1,1.4);key.name='xr-observatory-key';
  architectural.add(key);
  const fill=new THREE.DirectionalLight(0x91bacb,.16);
  fill.position.set(1.8,1.5,-2.2);fill.name='xr-observatory-fill';
  architectural.add(fill);
  let mode='studio',changes=0;
  function setMode(next){
    if(next!=='studio'&&next!=='focus')return false;
    if(next!==mode)changes++;
    mode=next;
    boundaries.visible=next==='studio';
    if(grid)grid.visible=next==='studio';
    if(plinth)plinth.visible=next==='studio';
    for(const ring of rings)if(ring)ring.visible=next==='studio';
    if(floor)floor.material.opacity=next==='focus'?.52:1;
    key.intensity=next==='focus'?.48:.34;
    fill.intensity=next==='focus'?.22:.16;
    return true;
  }
  function snapshot(){return Object.freeze({mode,changes,
    datumSegments:vertices.length/6,lighting:'fixed editorial; no anatomy-derived assumptions'})}
  setMode(mode);
  return Object.freeze({setMode,snapshot,root:architectural});
}
global.PulmoXRStagePresence=Object.freeze({create});
})(typeof window!=='undefined'?window:globalThis);
