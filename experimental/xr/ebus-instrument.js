(function ebus(root){
'use strict';
const finite=Number.isFinite,clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),v3=p=>Array.isArray(p)&&p.length===3&&p.every(finite),norm=p=>Math.hypot(...p),unit=p=>{if(!v3(p)||norm(p)<1e-8)throw Error('INVALID_AXIS');return p.map(x=>x/norm(p));},dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function createProbe(){
 let mode='off',extension=0,roll=0,status='STOWED';
 return Object.freeze({select(next){if(!['off','radial','convex'].includes(next))throw Error('INVALID_MODE');if(extension>0&&next!==mode){status='RETRACT_BEFORE_MODE_CHANGE';return false;}mode=next;roll=0;status=mode==='radial'?'PROBE_STOWED':mode==='convex'?'INTEGRATED_SECTOR_REFERENCE':'STOWED';return true;},reset(){mode='off';extension=0;roll=0;status='STOWED';},feed(delta,check){if(mode!=='radial'||!finite(delta))return false;const target=clamp(extension+delta,0,12);if(target>extension&&(typeof check!=='function'||!check(target))){status='CLEARANCE_BLOCKED';return false;}extension=target;status=extension>0?'PROBE_EXTENDED':'PROBE_STOWED';return true;},rotate(delta){if(mode==='radial'&&finite(delta))roll=((roll+delta+Math.PI)%(2*Math.PI)+2*Math.PI)%(2*Math.PI)-Math.PI;},snapshot(){return {mode,extensionMm:extension,rollRad:roll,status,shaftLocked:mode==='radial'&&extension>0,channelOccupied:mode==='radial',acousticValidated:false,patientSpecific:false};}});
}
function probePose({origin,forward,right,up,extensionMm,rollRad=0,mmPerUnit}){
 if(!v3(origin)||!finite(extensionMm)||extensionMm<0||extensionMm>12||!finite(rollRad)||!finite(mmPerUnit)||mmPerUnit<=0)throw Error('INVALID_PROBE_POSE');
 const z=unit(forward),x0=unit(right),x=unit(x0.map((v,i)=>v-dot(x0,z)*z[i])),y0=unit(cross(x,z)),y=dot(y0,unit(up))<0?y0.map(v=>-v):y0;
 const p=origin.map((v,i)=>v+z[i]*extensionMm/mmPerUnit),c=Math.cos(rollRad),s=Math.sin(rollRad);
 return {origin:p,forward:z,right:x.map((v,i)=>c*v+s*y[i]),up:y.map((v,i)=>c*v-s*x[i])};
}
function clearance({pose,extensionMm,field,mmPerUnit,radiusMm=.7}){
 if(typeof field!=='function'||!finite(mmPerUnit)||mmPerUnit<=0||!finite(extensionMm)||extensionMm<0||extensionMm>12)return false;
 const steps=Math.max(1,Math.ceil(extensionMm/.5));
 for(let i=0;i<=steps;i++){const p=pose.origin.map((x,j)=>x-pose.forward[j]*extensionMm*(1-i/steps)/mmPerUnit),d=field(p);if(!finite(d)||-d*mmPerUnit<radiusMm+.2)return false;}
 return true;
}
function sector({origin,forward,up,field,mmPerUnit,rangeMm=20,sectorDeg=60,coupling='air',rays=64}){
 if(!v3(origin)||typeof field!=='function'||!finite(mmPerUnit)||mmPerUnit<=0||!finite(rangeMm)||rangeMm<5||rangeMm>40||!finite(sectorDeg)||sectorDeg<30||sectorDeg>90||!Number.isInteger(rays)||rays<16||rays>96||!['air','ideal-fluid'].includes(coupling))throw Error('INVALID_SECTOR');
 const z=unit(forward),y0=unit(up),y=unit(y0.map((v,i)=>v-dot(y0,z)*z[i]));
 if(!(field(origin)<0))return {state:'ORIGIN_OUTSIDE_LUMEN',rays:[],rangeMm,sectorDeg,coupling};
 const values=[];
 for(let i=0;i<rays;i++){const angle=(i/(rays-1)-.5)*sectorDeg*Math.PI/180,dir=y.map((v,j)=>v*Math.cos(angle)+z[j]*Math.sin(angle));let last=0,hit=null;
  const value=mm=>field(origin.map((v,j)=>v+dir[j]*mm/mmPerUnit));
  for(let k=1;k<=32;k++){const mm=k*rangeMm/32,d=value(mm);if(!finite(d))throw Error('INVALID_FIELD');if(d>=0){let lo=last,hi=mm;for(let n=0;n<6;n++){const mid=(lo+hi)/2;if(value(mid)>=0)hi=mid;else lo=mid;}hit=(lo+hi)/2;break;}last=mm;}values.push(hit);
 }
 return {state:'REFERENCE_SECTOR_INTERFACE_ONLY',rays:values,rangeMm,sectorDeg,coupling,rayCount:rays,hitCount:values.filter(finite).length,acousticValidated:false,tissueBeyondWallModeled:false,patientSpecific:false};
}
function paint(canvas,result){
 const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height,cx=w/2,cy=h*.15,R=Math.min(w*.76,h*.73),a=(result?.sectorDeg||60)*Math.PI/180;
 c.fillStyle='#080d12';c.fillRect(0,0,w,h);c.strokeStyle='#31424c';c.lineWidth=1;
 for(let k=1;k<=4;k++){c.beginPath();c.arc(cx,cy,R*k/4,Math.PI/2-a/2,Math.PI/2+a/2);c.stroke();}
 for(const sign of [-1,1]){c.beginPath();c.moveTo(cx,cy);c.lineTo(cx+sign*R*Math.sin(a/2),cy+R*Math.cos(a/2));c.stroke();}
 for(let i=0;i<(result?.rays?.length||0);i++){const mm=result.rays[i];if(!finite(mm))continue;const theta=Math.PI/2+(i/(result.rays.length-1)-.5)*a,r=mm/result.rangeMm*R;c.strokeStyle=result.coupling==='ideal-fluid'?'#d7e0e3':'#52636d';c.lineWidth=result.coupling==='ideal-fluid'?2:1;c.beginPath();c.arc(cx,cy,r,theta-a/result.rays.length*.55,theta+a/result.rays.length*.55);c.stroke();}
 c.textAlign='center';c.font='600 16px system-ui';c.fillStyle='#b1c4ce';c.fillText('CONVEX SECTOR · REFERENCE',cx,24);c.font='12px system-ui';c.fillText(result?.coupling==='ideal-fluid'?'IDEAL INTERFACE · NO TISSUE DATA':'AIR · GEOMETRY ONLY',cx,h-32);c.fillText(result?.state==='REFERENCE_SECTOR_INTERFACE_ONLY'?result.hitCount+'/'+result.rayCount+' rays · '+result.rangeMm+' mm':'SCAN UNAVAILABLE',cx,h-14);
}
function createVisual(THREE,scene){
 const g=new THREE.Group();g.name='Q source-linked instrument assembly';scene.add(g);g.visible=false;
 const mat=new THREE.MeshStandardMaterial({color:0x34434a,roughness:.7}),probeMat=new THREE.MeshStandardMaterial({color:0xc1a477,roughness:.55}),sectorMat=new THREE.MeshBasicMaterial({color:0x7dabb9,transparent:true,opacity:.20,side:THREE.DoubleSide,depthWrite:false});
 const geo=new THREE.CylinderGeometry(1,1,1,8),Y=new THREE.Vector3(0,1,0),links=Array.from({length:24},()=>{const m=new THREE.Mesh(geo,mat);g.add(m);return m});
 const tip=new THREE.Mesh(geo,probeMat);g.add(tip);tip.visible=false;
 const proximalProbe=new THREE.Mesh(geo,probeMat);g.add(proximalProbe);proximalProbe.visible=false;
 const convex=new THREE.Mesh(new THREE.SphereGeometry(1,12,8),probeMat);g.add(convex);convex.visible=false;
 const fan=new THREE.Mesh(new THREE.CircleGeometry(1,32,Math.PI/2-Math.PI/6,Math.PI/3),sectorMat);g.add(fan);fan.visible=false;
 function line(m,a,b,r){const d=b.clone().sub(a),l=d.length();m.visible=l>1e-5;if(!m.visible)return;m.position.copy(a).lerp(b,.5);m.quaternion.setFromUnitVectors(Y,d.normalize());m.scale.set(r,l+.0004,r);}
 function update({handle,entry,entryDirection,tipWorld,probeWorld,probeForward,probeUp,worldPerMm,diameterMm,mode,extensionMm,showConnection}){
  g.visible=true;
  if(showConnection&&handle&&entry&&entryDirection){
   handle.updateWorldMatrix(true,false);const boot=handle.localToWorld(new THREE.Vector3(0,-.136,0)),dir=new THREE.Vector3(0,-1,0).transformDirection(handle.matrixWorld),distance=boot.distanceTo(entry),reach=Math.min(.24,distance*.35);
   const curve=new THREE.CubicBezierCurve3(boot,boot.clone().addScaledVector(dir,reach),entry.clone().addScaledVector(entryDirection,-reach),entry);
   for(let i=0;i<links.length;i++)line(links[i],curve.getPoint(i/links.length),curve.getPoint((i+1)/links.length),Math.max(.0015,diameterMm*.5*worldPerMm));
   if(mode==='radial'){const port=handle.localToWorld(new THREE.Vector3(.045,-.012,0)),end=handle.localToWorld(new THREE.Vector3(.125+extensionMm/1000,-.012,0));line(proximalProbe,port,end,.0011);}else proximalProbe.visible=false;
  }else{links.forEach(m=>m.visible=false);proximalProbe.visible=false;}
  tip.visible=false;convex.visible=false;fan.visible=false;
  if(mode==='radial'&&extensionMm>0&&tipWorld&&probeWorld)line(tip,tipWorld,probeWorld,.7*worldPerMm);
  if(mode==='convex'&&tipWorld&&probeForward&&probeUp){convex.position.copy(tipWorld).addScaledVector(probeUp,diameterMm*.5*worldPerMm);convex.scale.set(1.5*worldPerMm,1.0*worldPerMm,3*worldPerMm);convex.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),probeForward);convex.visible=true;const normal=probeForward.clone().cross(probeUp).normalize(),qx=probeForward.clone(),qy=probeUp.clone(),qz=qx.clone().cross(qy).normalize();fan.position.copy(convex.position);fan.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(qx,qy,qz));fan.scale.setScalar(20*worldPerMm);fan.visible=normal.lengthSq()>.5;}
 }
 return {group:g,update,hide(){g.visible=false;}};
}
root.PulmoXREBUS=Object.freeze({createProbe,probePose,clearance,sector,paint,createVisual});
})(typeof window!=='undefined'?window:globalThis);
