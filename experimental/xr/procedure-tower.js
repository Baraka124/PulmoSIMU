(function towerModule(root){
'use strict';
const finite=Number.isFinite,v3=x=>Array.isArray(x)&&x.length===3&&x.every(finite);
function supportPose({screen,normal,scale,floor,room,head,patient=null,halfHeight=.344,lowerExtension=.10}){
 if(![screen,normal,room,head].every(v3)||![scale,floor,halfHeight,lowerExtension].every(finite)||scale<.6||scale>1.5||halfHeight<=0||lowerExtension<0)return null;
 const n=Math.hypot(normal[0],normal[2]);if(n<.2)return null;
 const dx=normal[0]/n,dz=normal[2]/n;
 const base=[screen[0]-dx*.15,floor,screen[2]-dz*.15],top=screen[1]-(halfHeight+lowerExtension)*scale-.025;
 const height=top-floor;
 if(height<.38||height>1.95||Math.hypot(base[0]-room[0],base[2]-room[2])>2.6||Math.hypot(base[0]-head[0],base[2]-head[2])<.48)return null;
 if(patient&&v3(patient.neck)&&Array.isArray(patient.basis)&&patient.basis.length===3&&patient.basis.every(v3)){
  const d=base.map((v,i)=>v-patient.neck[i]),x=d.reduce((s,v,i)=>s+v*patient.basis[0][i],0),y=d.reduce((s,v,i)=>s+v*patient.basis[1][i],0);
  if(Math.abs(x)<.64&&y> -1.91&&y<.71)return null;
 }
 return {base,top,height,yaw:Math.atan2(dx,dz),physicalCollisionValidated:false};
}
function create(THREE,scene,{secondary=false}={}){
 const g=new THREE.Group();g.name=secondary?'ui09-ultrasound-equipment-stand':'ui09-endoscopy-equipment-tower';scene.add(g);g.visible=false;
 const graphite=new THREE.MeshStandardMaterial({color:0x30383d,roughness:.72,metalness:.18}),metal=new THREE.MeshStandardMaterial({color:0x7d8a90,roughness:.48,metalness:.35}),porcelain=new THREE.MeshStandardMaterial({color:0xa0aaae,roughness:.8}),dark=new THREE.MeshStandardMaterial({color:0x141a1e,roughness:.92});
 const box=(w,h,d,x,y,z,m)=>{const o=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);o.position.set(x,y,z);g.add(o);return o;};
 const baseWidth=secondary?.40:.52,baseDepth=secondary?.38:.46;
 box(baseWidth,.055,baseDepth,0,.071,0,graphite);
 for(const x of [-baseWidth*.40,baseWidth*.40])for(const z of [-baseDepth*.36,baseDepth*.36]){
  const wheel=new THREE.Mesh(new THREE.CylinderGeometry(.034,.034,.022,12),dark);wheel.rotation.z=Math.PI/2;wheel.position.set(x,.035,z);g.add(wheel);
 }
 const column=box(.085,1,.105,0,.5,-.055,metal),sleeve=box(.125,.29,.14,0,.25,-.055,graphite);
 const neck=box(.15,.04,.23,0,1,.018,metal);
 const moduleGroup=new THREE.Group();g.add(moduleGroup);
 if(!secondary){
  const block=(w,h,d,x,y,z,m)=>{const o=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);o.position.set(x,y,z);moduleGroup.add(o);return o;};
  block(.39,.055,.35,0,.02,.01,graphite);block(.36,.112,.29,0,.104,.015,porcelain);block(.345,.082,.008,0,.104,.164,dark);
  block(.36,.104,.29,0,.218,.015,porcelain);block(.345,.074,.008,0,.218,.164,graphite);
  for(let i=0;i<5;i++)block(.024,.003,.004,-.12+i*.035,.22,.169,metal);
 }
 let state='HIDDEN';
 function hide(){g.visible=false;state='HIDDEN';}
 function update({screen,floor,room,head,patient,enabled,halfHeight,lowerExtension}){
  if(!enabled||!screen?.visible){hide();return false;}
  screen.updateWorldMatrix(true,false);const p=screen.getWorldPosition(new THREE.Vector3()),q=screen.getWorldQuaternion(new THREE.Quaternion()),n=new THREE.Vector3(0,0,1).applyQuaternion(q),s=screen.getWorldScale(new THREE.Vector3()).x;
  const pose=supportPose({screen:p.toArray(),normal:n.toArray(),scale:s,floor,room,head,patient,halfHeight,lowerExtension});
  if(!pose){hide();state='SUPPORT_ENVELOPE_UNAVAILABLE';return false;}
  g.position.set(...pose.base);g.rotation.y=pose.yaw;column.position.y=(pose.height+.095)/2;column.scale.y=pose.height-.095;sleeve.visible=pose.height>.43;neck.position.y=pose.height;
  moduleGroup.position.y=Math.max(.14,Math.min(.42,pose.height-.35));moduleGroup.visible=!secondary&&pose.height>.68;
  g.visible=true;state='AUTHORED_FLOOR_SUPPORT';return true;
 }
 return {group:g,update,hide,snapshot:()=>({state,secondary,physicalCollisionValidated:false,operableHardware:false})};
}
root.PulmoXRProcedureTower=Object.freeze({supportPose,create});
})(typeof window!=='undefined'?window:globalThis);
