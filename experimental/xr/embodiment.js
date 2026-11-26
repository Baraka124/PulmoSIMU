(function embodied(root){
'use strict';
const finite3=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite),dist=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i])),clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const chains=[['wrist','thumb-metacarpal','thumb-phalanx-proximal','thumb-phalanx-distal','thumb-tip'],...['index','middle','ring','pinky'].map(f=>['wrist',f+'-finger-metacarpal',f+'-finger-phalanx-proximal',f+'-finger-phalanx-intermediate',f+'-finger-phalanx-distal',f+'-finger-tip'])];
const names=[...new Set(chains.flat())];
function createGrip(){
 let ready=false,held=false,last=null;
 return Object.freeze({reset(){ready=false;held=false;last=null},step({valid,closed,near,point,dt,axis=[0,1,0]}){
  if(!valid||!finite3(point)||!finite3(axis)||!Number.isFinite(dt)||dt<=0||dt>.08){ready=false;held=false;last=null;return {held:false,deltaMm:0,state:'TRACKING_OR_GATE_LOST'};}
  if(!closed){ready=true;held=false;last=null;return {held:false,deltaMm:0,state:'OPEN'};}
  if(!held){if(!ready||!near){ready=false;return {held:false,deltaMm:0,state:'RELEASE_AND_REGRASP'};}held=true;ready=false;last=point.slice();return {held:true,deltaMm:0,state:'ACQUIRED'};}
  const d=dist(point,last),n=Math.hypot(...axis);if(d>.06||n<.5){held=false;last=null;ready=false;return {held:false,deltaMm:0,state:'DISCONTINUITY'};}
  const mm=1000*point.reduce((s,p,i)=>s+(p-last[i])*axis[i]/n,0);last=point.slice();return {held:true,deltaMm:clamp(mm,-45*dt,45*dt),state:'HELD'};
 }});
}
function elbow(shoulder,wrist,side,upper=.30,lower=.27){
 if(!finite3(shoulder)||!finite3(wrist))return null;
 const d=dist(shoulder,wrist);if(d<.025||d>upper+lower+.12)return null;
 const dir=wrist.map((v,i)=>(v-shoulder[i])/d),l=clamp(d,Math.abs(upper-lower)+.001,upper+lower-.001);
 const a=(upper*upper-lower*lower+l*l)/(2*l),b=Math.sqrt(Math.max(0,upper*upper-a*a));
 let pole=side.map((v,i)=>v*.65+(i===1?-.85:0)),dot=pole.reduce((s,v,i)=>s+v*dir[i],0);pole=pole.map((v,i)=>v-dot*dir[i]);let n=Math.hypot(...pole);if(n<1e-4){pole=[0,0,1].map((v,i)=>v-dir[2]*dir[i]);n=Math.hypot(...pole);}if(n<1e-4)return null;
 return shoulder.map((v,i)=>v+dir[i]*a+pole[i]*b/n);
}
function createVisual(THREE,scene){
 const group=new THREE.Group();group.name='P01 estimated operator and tracked glove surfaces';group.visible=false;scene.add(group);
 const sphere=new THREE.SphereGeometry(1,10,8),cylinder=new THREE.CylinderGeometry(1,1,1,10),Y=new THREE.Vector3(0,1,0);
 const cloth=new THREE.MeshStandardMaterial({color:0x303c44,roughness:.92}),glove=new THREE.MeshStandardMaterial({color:0x8baeb3,roughness:.83}),shoe=new THREE.MeshStandardMaterial({color:0x1c242a,roughness:.92});
 const mesh=(geo,mat)=>{const m=new THREE.Mesh(geo,mat);m.visible=false;group.add(m);return m};
 const ball=(m,p,scale,q)=>{if(!p){m.visible=false;return;}m.position.copy(p);m.scale.set(...scale);if(q)m.quaternion.copy(q);else m.quaternion.identity();m.visible=true;};
 const link=(m,a,b,r)=>{if(!a||!b){m.visible=false;return;}const d=b.clone().sub(a),l=d.length();if(l<.002||l>1.1){m.visible=false;return;}m.position.copy(a).lerp(b,.5);m.quaternion.setFromUnitVectors(Y,d.normalize());m.scale.set(r,l+.004,r);m.visible=true;};
 const torso=mesh(sphere,cloth),hips=mesh(sphere,cloth),legs=[0,1].map(()=>({thigh:mesh(cylinder,cloth),calf:mesh(cylinder,cloth),knee:mesh(sphere,cloth),foot:mesh(sphere,shoe)}));
 const hands=['left','right'].map(()=>({palm:mesh(sphere,glove),joints:Object.fromEntries(names.map(n=>[n,mesh(sphere,glove)])),links:chains.flatMap(c=>c.slice(1).map((n,i)=>({a:c[i],b:n,m:mesh(cylinder,glove)}))),upper:mesh(cylinder,cloth),lower:mesh(cylinder,cloth),elbow:mesh(sphere,cloth)}));
 let evidence={body:'HIDDEN',left:'UNAVAILABLE',right:'UNAVAILABLE'},yaw=new THREE.Vector3(1,0,0);
 function reset(){group.visible=false;for(const m of group.children)m.visible=false;evidence={body:'HIDDEN',left:'UNAVAILABLE',right:'UNAVAILABLE'};}
 function update({head,right,forward,floor,showBody=true,showHands=true,samples={},controllers={}}){
  if(!finite3(head)||!finite3(right)||!finite3(forward)){reset();return;}
  group.visible=true;
  const h=new THREE.Vector3(...head),r=new THREE.Vector3(...right).setY(0);
  if(r.lengthSq()>.2)yaw.copy(r.normalize());
  const side=yaw.clone(),front=new THREE.Vector3(side.z,0,-side.x),q=new THREE.Quaternion().setFromAxisAngle(Y,Math.atan2(-side.z,side.x));
  const height=clamp(h.y-floor,1.05,1.95),seated=height<1.40,ratio=clamp(height/1.68,.78,1.12);
  const chest=h.clone().addScaledVector(front,-.12).add(new THREE.Vector3(0,-.33*ratio,0)),pelvis=chest.clone().add(new THREE.Vector3(0,-.36*ratio,0));
  ball(torso,showBody?chest:null,[.19*ratio,.25*ratio,.125*ratio],q);ball(hips,showBody?pelvis:null,[.17*ratio,.12*ratio,.13*ratio],q);
  for(let i=0;i<2;i++){
   const sign=i?1:-1,leg=legs[i],hip=pelvis.clone().addScaledVector(side,sign*.105*ratio),foot=new THREE.Vector3(hip.x,floor+.055,hip.z).addScaledVector(front,seated?.32:.035),knee=hip.clone().lerp(foot,.5).addScaledVector(front,seated?.22:.035);
   link(leg.thigh,showBody?hip:null,knee,.069*ratio);link(leg.calf,showBody?knee:null,foot,.047*ratio);ball(leg.knee,showBody?knee:null,[.064*ratio,.069*ratio,.064*ratio]);ball(leg.foot,showBody?foot.clone().addScaledVector(front,.065):null,[.057,.052,.125],q);
   const name=i?'right':'left',v=hands[i],sample=samples[name],points=sample?.points||{},w=points.wrist,ctrl=controllers[name],wrist=finite3(w)?new THREE.Vector3(...w):finite3(ctrl)?new THREE.Vector3(...ctrl):null;
   const shoulder=chest.clone().addScaledVector(side,sign*.19*ratio).add(new THREE.Vector3(0,.14*ratio,0)),e=wrist?elbow(shoulder.toArray(),wrist.toArray(),side.clone().multiplyScalar(sign).toArray(),.31*ratio,.27*ratio):null,el=e?new THREE.Vector3(...e):null;
   link(v.upper,showBody?shoulder:null,el,.047*ratio);link(v.lower,showBody?el:null,wrist,.037*ratio);ball(v.elbow,showBody?el:null,[.047,.047,.047]);
   const get=n=>showHands&&finite3(points[n])?new THREE.Vector3(...points[n]):null;
   for(const n of names){const p=get(n),radius=clamp(sample?.radii?.[n]||(/tip$/.test(n)?.007:.009),.004,.014);ball(v.joints[n],p,[radius,radius,radius]);}
   for(const s of v.links){const a=get(s.a),b=get(s.b),rad=clamp(sample?.radii?.[s.b]||.007,.004,.011);link(s.m,a,b,rad);}
   const a=get('index-finger-phalanx-proximal'),b=get('pinky-finger-phalanx-proximal'),wr=get('wrist');
   if(a&&b&&wr){const center=a.clone().lerp(b,.5).lerp(wr,.40),x=a.clone().sub(b),y=a.clone().lerp(b,.5).sub(wr);if(x.length()>.015&&y.length()>.025){const width=x.length(),len=y.length(),qy=y.normalize(),qz=x.clone().cross(qy).normalize();if(qz.lengthSq()>.5){const qx=qy.clone().cross(qz).normalize(),pq=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(qx,qy,qz));ball(v.palm,center,[width*.65,len*.64,.017],pq);}else v.palm.visible=false;}else v.palm.visible=false;}else v.palm.visible=false;
   evidence[name]=sample&&Object.keys(points).length>=20?'DIRECT_HAND_JOINTS':wrist?(sample?'PARTIAL_JOINTS':'CONTROLLER_ENDPOINT'):'UNAVAILABLE';
  }
  evidence.body=showBody?'ESTIMATED_HEAD_RELATIVE_POSTURE':'HIDDEN';evidence.posture=seated?'SEATED_ESTIMATE':'STANDING_ESTIMATE';
 }
 return Object.freeze({group,update,reset,snapshot:()=>({...evidence})});
}
root.PulmoXREmbodiment=Object.freeze({names,createGrip,elbow,createVisual});
})(typeof window!=='undefined'?window:globalThis);
