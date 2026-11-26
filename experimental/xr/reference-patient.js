(function patient(root){
'use strict';
const v3=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite),sub=(a,b)=>a.map((x,i)=>x-b[i]),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),norm=p=>Math.hypot(...p),unit=p=>{const n=norm(p);if(n<.001)throw Error('Invalid orientation');return p.map(v=>v/n);};
const dimensions=Object.freeze({bedLengthM:2.10,bedWidthM:.72,neckHeightM:.95,mouthLocal:[0,.19,.155],bodyLengthM:1.87});
function placement({head,forward,floor}){
 if(!v3(head)||!v3(forward)||!Number.isFinite(floor))throw Error('Invalid placement');
 const F=unit([forward[0],0,forward[2]]),R=[-F[2],0,F[0]],X=F,Y=R.map(v=>-v),Z=[0,1,0];
 const neck=[head[0]+F[0]*.83-R[0]*.36,floor+dimensions.neckHeightM,head[2]+F[2]*.83-R[2]*.36];
 return {neck,basis:[X,Y,Z],floor,orientation:'SUPINE_REFERENCE'};
}
function toWorld(spec,p){return spec.neck.map((x,i)=>x+spec.basis.reduce((s,b,j)=>s+b[i]*p[j],0));}
function toLocal(spec,p){const d=sub(p,spec.neck);return spec.basis.map(b=>dot(b,d));}
function operatorZone(spec,head){
 if(!spec||!v3(head))return {allowed:false,state:'NO_TRACKED_OPERATOR'};
 const p=toLocal(spec,head),mouth=dimensions.mouthLocal,d=Math.hypot(p[0]-mouth[0],p[1]-mouth[1]),above=p[2];
 // Head projection must be outside the mattress or at its head end.
 const outside=Math.abs(p[0])>=.34||p[1]>=.37;
 const allowed=outside&&d>=.28&&d<=1.15&&above>=.18&&above<=1.40;
 return {allowed,state:allowed?'OPERATOR_ZONE':'APPROACH_HEAD_OF_BED',horizontalDistanceM:d,heightAboveNeckM:above};
}
function createVisual(THREE,scene){
 const g=new THREE.Group();g.name='R01 reference supine manikin and bed';g.visible=false;scene.add(g);
 const skin=new THREE.MeshStandardMaterial({color:0xb8a69c,roughness:.91});
 const transparent=skin.clone();transparent.transparent=true;transparent.opacity=.18;transparent.depthWrite=false;
 const sheet=new THREE.MeshStandardMaterial({color:0x718d98,roughness:.96}),sheetTransparent=sheet.clone();sheetTransparent.transparent=true;sheetTransparent.opacity=.13;sheetTransparent.depthWrite=false;
 const metal=new THREE.MeshStandardMaterial({color:0x87949a,roughness:.42,metalness:.45}),base=new THREE.MeshStandardMaterial({color:0x313d43,roughness:.77}),mattress=new THREE.MeshStandardMaterial({color:0xd0d5d2,roughness:.94}),dark=new THREE.MeshStandardMaterial({color:0x3d2e30,roughness:.9});
 const body=new THREE.Group();g.add(body);const parts=[];
 const sphere=new THREE.SphereGeometry(1,24,16),box=new THREE.BoxGeometry(1,1,1),cyl=new THREE.CylinderGeometry(1,1,1,12);
 function mesh(geo,mat,p,s,parent=g){const m=new THREE.Mesh(geo,mat);m.position.set(...p);m.scale.set(...s);parent.add(m);return m;}
 function ell(p,s,mat=skin){const m=mesh(sphere,mat,p,s,body);parts.push({m,mat});return m;}
 function block(p,s,mat=base){return mesh(box,mat,p,s);}
 block([0,-.60,-.17],[.72,2.10,.10],base);
 block([0,-.60,-.10],[.68,2.03,.07],mattress);
 block([0,-.60,-.73],[.53,1.42,.12],base);
 for(const y of [-1.12,-.10]){block([0,y,-.46],[.18,.22,.49],metal);block([0,y,-.89],[.63,.12,.08],base);for(const x of [-.27,.27]){const wheel=mesh(cyl,base,[x,y,-.905],[.043,.065,.043]);wheel.rotation.z=Math.PI/2;}}
 block([0,.18,-.045],[.36,.36,.06],mattress);
 // Anatomical context is a manikin shell, not a patient-derived external surface.
 ell([0,.205,.055],[.092,.135,.097]);ell([0,.065,.006],[.065,.11,.075]);
 ell([0,-.235,.015],[.205,.285,.119],sheet);ell([0,-.48,.012],[.18,.18,.112],sheet);ell([0,-.64,.01],[.177,.18,.11],sheet);
 for(const s of [-1,1]){
  ell([s*.22,-.19,.01],[.066,.13,.065],sheet);ell([s*.245,-.38,-.005],[.052,.15,.053],sheet);ell([s*.24,-.57,-.002],[.044,.105,.036]);
  ell([s*.09,-.85,-.013],[.087,.24,.081],sheet);ell([s*.092,-1.17,-.024],[.063,.21,.064],sheet);ell([s*.092,-1.42,-.005],[.066,.11,.082],sheet);
  ell([s*.089,.205,.036],[.014,.033,.03]);
 }
 ell([0,.225,.144],[.025,.035,.020]);const mouth=mesh(sphere,dark,dimensions.mouthLocal,[.028,.011,.008],g);mouth.name='Reference mouth landmark (not reconstructed upper airway)';
 let spec=null,view='external';
 function setView(next){if(!['external','transparent','internal'].includes(next))throw Error('Invalid patient view');view=next;body.visible=next!=='internal';mouth.visible=next!=='internal';for(const {m,mat} of parts)m.material=next==='transparent'?(mat===sheet?sheetTransparent:transparent):mat;}
 function place(p){spec=p;g.position.set(...p.neck);g.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(...p.basis.map(v=>new THREE.Vector3(...v))));g.updateWorldMatrix(true,false);g.visible=true;setView('external');}
 return {group:g,place,setView,hide(){g.visible=false;spec=null;},mouthWorld(){return spec?new THREE.Vector3(...toWorld(spec,dimensions.mouthLocal)):null;},zone(head){return operatorZone(spec,head);},snapshot(){return {active:!!spec,view,referenceManikin:true,sourceRegistered:false,upperAirwayReconstructed:false,physicalCollisionSolved:false,placement:spec?{...spec}:null};}};
}
root.PulmoXRPatient=Object.freeze({dimensions,placement,toWorld,toLocal,operatorZone,createVisual});
})(typeof window!=='undefined'?window:globalThis);
