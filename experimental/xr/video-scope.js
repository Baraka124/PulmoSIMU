/* XR23 — generic reference flexible video bronchoscope, metres in XR.
 * Optical reference: Olympus BF-P190 published 110 deg forward view,
 * distal OD 4.2 mm, tube OD 4.1 mm, channel ID 2 mm.
 * Control housing geometry is illustrative, not OEM CAD or a calibrated device.
 */
(function(root){
'use strict';
const REF=Object.freeze({horizontalFovDeg:110,distalOdMm:4.2,shaftOdMm:4.1,channelIdMm:2,workingLengthMm:600,depthOfFieldMm:[2,50]});
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
function verticalFov(horizontal,aspect){
 if(!Number.isFinite(horizontal)||horizontal<=0||horizontal>=179||!(aspect>0))throw Error('INVALID_OPTICS');
 return 2*Math.atan(Math.tan(horizontal*Math.PI/360)/aspect)*180/Math.PI;
}
function poseBasis(THREE,tangent,roll,h,v,limit,sourceNormal=null){
 const t=tangent.clone().normalize();
 const seed=Math.abs(t.y)>.96?new THREE.Vector3(0,0,1):new THREE.Vector3(0,1,0);
 const normal=sourceNormal?.clone().addScaledVector(t,-sourceNormal.dot(t));
 const right=(normal&&normal.lengthSq()>1e-8?normal.normalize():t.clone().cross(seed).normalize()).applyAxisAngle(t,roll);
 const up=right.clone().cross(t).normalize();
 const ax=clamp(h,-1,1)*limit*Math.PI/180,ay=clamp(v,-1,1)*limit*Math.PI/180;
 const direction=t.clone().addScaledVector(right,Math.tan(ax)).addScaledVector(up,Math.tan(ay)).normalize();
 const correctedUp=up.clone().addScaledVector(direction,-up.dot(direction)).normalize();
 return {direction,up:correctedUp,right};
}
function createClutch(){
 let last=null;
 return Object.freeze({reset(){last=null},step({held,tracking,position,axis,dt}){
  if(!held||!tracking||!Array.isArray(position)||!Array.isArray(axis)||position.length!==3||axis.length!==3||![...position,...axis,dt].every(Number.isFinite)){last=null;return 0}
  const norm=Math.hypot(...axis);
  if(norm<.5||dt<=0||dt>.12){last=null;return 0}
  if(!last){last=position.slice();return 0}
  const delta=position.map((p,i)=>p-last[i]);last=position.slice();
  if(Math.hypot(...delta)>.055){last=null;return 0}
  const mm=1000*delta.reduce((s,d,i)=>s+d*axis[i]/norm,0);
  return clamp(mm,-45*dt,45*dt);
 }});
}
function createVisual(THREE){
 const group=new THREE.Group();group.name='XR23 reference flexible video bronchoscope';
 const polymer=new THREE.MeshStandardMaterial({color:0x242d33,roughness:.68,metalness:.03});
 const grip=new THREE.MeshStandardMaterial({color:0x11191e,roughness:.86});
 const metal=new THREE.MeshStandardMaterial({color:0x8c9ca4,roughness:.32,metalness:.68});
 const accent=new THREE.MeshStandardMaterial({color:0x365766,roughness:.52});
 const glass=new THREE.MeshStandardMaterial({color:0x102836,roughness:.17,metalness:.35});
 const light=new THREE.MeshBasicMaterial({color:0xf2e8cf});
 const dark=new THREE.MeshStandardMaterial({color:0x05090d,roughness:.95});
 const mesh=(geo,mat,parent=group)=>{const m=new THREE.Mesh(geo,mat);parent.add(m);return m};
 const body=mesh(new THREE.LatheGeometry([[0,-.084],[.019,-.084],[.025,-.064],[.027,-.02],[.031,.045],[.027,.065],[.015,.081],[0,.081]].map(p=>new THREE.Vector2(...p)),32),polymer);
 body.name='Control body';
 for(let i=0;i<6;i++){const ring=mesh(new THREE.TorusGeometry(.027,.0015,6,28),grip);ring.rotation.x=Math.PI/2;ring.position.y=-.04+i*.009;}
 const collar=mesh(new THREE.CylinderGeometry(.014,.01,.032,24),grip);collar.position.y=-.095;
 const lever=new THREE.Group();lever.position.set(0,.047,-.029);group.add(lever);
 const leverStem=mesh(new THREE.BoxGeometry(.014,.025,.012),accent,lever);leverStem.position.y=.012;
 const pad=mesh(new THREE.BoxGeometry(.028,.009,.018),grip,lever);pad.position.y=.026;pad.name='Angulation thumb lever';
 const valve=mesh(new THREE.CylinderGeometry(.010,.011,.012,24),metal);valve.position.set(.019,.035,.022);valve.rotation.x=Math.PI/2;
 const button=mesh(new THREE.CylinderGeometry(.009,.009,.005,24),accent);button.position.set(.019,.035,.030);button.rotation.x=Math.PI/2;button.name='Suction valve';
 const port=mesh(new THREE.CylinderGeometry(.008,.009,.023,20),polymer);port.rotation.z=-Math.PI/3;port.position.set(.032,-.018,0);port.name='Working channel port';
 const rim=mesh(new THREE.TorusGeometry(.006,.0014,8,20),metal);rim.rotation.y=Math.PI/2;rim.position.set(.044,-.012,0);
 const hole=mesh(new THREE.CircleGeometry(.005,20),dark);hole.rotation.y=Math.PI/2;hole.position.set(.045,-.012,0);
 const cable=mesh(new THREE.CylinderGeometry(.006,.006,.055,16),grip);cable.position.y=.104;
 const boot=mesh(new THREE.CylinderGeometry(.009,.003,.035,24),grip);boot.position.y=-.119;
 const shaftRoot=new THREE.Group();group.add(shaftRoot);
 const shaftSegments=[];const cylinder=new THREE.CylinderGeometry(1,1,1,12);
 for(let i=0;i<32;i++){const m=mesh(cylinder,grip,shaftRoot);shaftSegments.push(m)}
 const tip=new THREE.Group();tip.name='Distal optical head';shaftRoot.add(tip);
 const cap=mesh(new THREE.CylinderGeometry(1,1,1,24),polymer,tip);cap.rotation.x=Math.PI/2;
 const face=mesh(new THREE.CircleGeometry(1,32),metal,tip);face.rotation.y=Math.PI;
 const lens=mesh(new THREE.CircleGeometry(1,24),glass,tip);lens.rotation.y=Math.PI;
 const leds=[mesh(new THREE.CircleGeometry(1,16),light,tip),mesh(new THREE.CircleGeometry(1,16),light,tip)];
 leds.forEach(m=>m.rotation.y=Math.PI);
 const channel=mesh(new THREE.CircleGeometry(1,24),dark,tip);channel.rotation.y=Math.PI;
 const Y=new THREE.Vector3(0,1,0),Z=new THREE.Vector3(0,0,-1);
 function update({h=0,v=0,roll=0,diameterMm=REF.distalOdMm,connected=false}={}){
  const d=clamp(diameterMm,2.8,6.5)/1000,rad=d*.5;
  const shaftRad=rad*REF.shaftOdMm/REF.distalOdMm;
  lever.rotation.x=-clamp(v,-1,1)*.48;
  shaftRoot.visible=!connected;shaftRoot.rotation.y=roll;
  const points=[new THREE.Vector3(0,-.135,0)];
  // Piecewise constant-curvature distal bending; unloaded illustrative shape.
  // The visible 600 mm shaft is a separate handheld inspection model,
  // not an inferred collision-coupled shape inside the patient's airway.
  let pos=points[0].clone(),dir=new THREE.Vector3(0,-1,0);
  const ds=REF.workingLengthMm/1000/shaftSegments.length;
  for(let i=0;i<shaftSegments.length;i++){
    if(i<10)dir.applyAxisAngle(new THREE.Vector3(1,0,0),-.11);
    if(i>=24){dir.applyAxisAngle(new THREE.Vector3(1,0,0),v*.12);dir.applyAxisAngle(new THREE.Vector3(0,0,1),h*.12)}
    dir.normalize();pos=pos.clone().addScaledVector(dir,ds);points.push(pos);
    const m=shaftSegments[i];m.position.copy(points[i]).lerp(pos,.5);m.quaternion.setFromUnitVectors(Y,dir);
    m.scale.set(shaftRad,ds+.0006,shaftRad);
  }
  tip.position.copy(pos);tip.quaternion.setFromUnitVectors(Z,dir);
  cap.scale.set(rad,.003,rad);face.scale.setScalar(rad);face.position.z=-.0016;
  lens.position.set(-rad*.32,rad*.27,-.00172);lens.scale.setScalar(rad*.33);
  channel.position.set(rad*.28,-rad*.24,-.00173);channel.scale.setScalar(rad*.43);
  leds[0].position.set(rad*.45,rad*.39,-.00174);leds[1].position.set(-rad*.51,-rad*.28,-.00174);
  leds.forEach(m=>m.scale.setScalar(rad*.14));
 }
 update();group.visible=false;
 return Object.freeze({group,update,reference:REF});
}
root.PulmoXRVideoScope=Object.freeze({REF,verticalFov,poseBasis,createClutch,createVisual});
})(typeof window!=='undefined'?window:globalThis);
