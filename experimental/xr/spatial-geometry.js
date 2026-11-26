/* UI02 spatial surfaces: curved world-space substrates, individually extruded controls,
 * front-only atlas UVs and real geometry hit testing. No clinical or source state. */
(function(root){
'use strict';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function point(u,v,w,h,r=1.8,z=0){const x=(u-.5)*w;return [x,(.5-v)*h,z+(r>0?r-Math.sqrt(Math.max(.00001,r*r-x*x)):0)];}
function patch(T,{u=0,v=0,w=1,h=1,width,height,radius=1.8,z=0,segments=12}){
 const p=[],uv=[],idx=[];
 for(let i=0;i<=segments;i++){const x=u+w*i/segments;p.push(...point(x,v,width,height,radius,z),...point(x,v+h,width,height,radius,z));uv.push(x,1-v,x,1-v-h);if(i<segments){const j=2*i;idx.push(j,j+1,j+2,j+2,j+1,j+3);}}
 const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(p,3));g.setAttribute('uv',new T.Float32BufferAttribute(uv,2));g.setIndex(idx);g.computeVertexNormals();return g;
}
function sides(T,{u=0,v=0,w=1,h=1,width,height,radius=1.8,front=0,back=-.025,segments=12}){
 const p=[],idx=[];const quad=(a,b)=>{const i=p.length/3;p.push(...point(...a,width,height,radius,front),...point(...b,width,height,radius,front),...point(...b,width,height,radius,back),...point(...a,width,height,radius,back));idx.push(i,i+1,i+2,i,i+2,i+3);};
 for(let j=0;j<segments;j++){const x=u+w*j/segments,y=u+w*(j+1)/segments;quad([x,v],[y,v]);quad([y,v+h],[x,v+h]);}quad([u,v+h],[u,v]);quad([u+w,v],[u+w,v+h]);
 const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(p,3));g.setIndex(idx);g.computeVertexNormals();return g;
}
function create(T,mesh,{width,height,radius=1.8,deckStart=.54,footerStart=1,depth=.027,texture}){
 const opts={width,height,radius};
 const face=new T.MeshBasicMaterial({map:texture,side:T.DoubleSide,transparent:false,depthTest:true,depthWrite:true,toneMapped:false});
 const deck=new T.MeshStandardMaterial({color:0x20262b,roughness:.8,metalness:.15,side:T.DoubleSide});
 const casing=new T.MeshStandardMaterial({color:0x3e494f,roughness:.57,metalness:.42,side:T.DoubleSide});
 const back=new T.MeshStandardMaterial({color:0x20262a,roughness:.78,metalness:.18,side:T.DoubleSide});
 const oldGeometry=mesh.geometry,oldMaterial=mesh.material;
 mesh.geometry=patch(T,{...opts});mesh.material=deck;mesh.renderOrder=0;
 oldGeometry?.dispose();oldMaterial?.dispose();
 const add=(g,m,name)=>{const o=new T.Mesh(g,m);o.name=name;mesh.add(o);return o;};
 // Substrate is the actual root and blocks rays even between buttons. The
 // display face covers only informational regions; action faces are separate.
 if(deckStart>0)add(patch(T,{...opts,v:0,h:deckStart,z:.001}),face,'spatial-information-display');
 if(footerStart<1)add(patch(T,{...opts,v:footerStart,h:1-footerStart,z:.001}),face,'spatial-footer-display');
 add(patch(T,{...opts,z:-depth}),back,'spatial-back');
 add(sides(T,{...opts,front:0,back:-depth}),casing,'spatial-machined-edge');
 const keys=new T.Group();keys.name='spatial-physical-controls';mesh.add(keys);
 let signature='',records=[],pressedId=null,pressedAt=-Infinity;
 function setActions(actions){
  const sig=actions.map(a=>[a.id,a.u,a.v,a.w,a.h].join(':')).join('|');
  if(sig!==signature){
   for(const o of [...keys.children]){keys.remove(o);o.traverse(k=>{if(k.geometry)k.geometry.dispose();});}
   records=[];signature=sig;
   for(const a of actions){
    const key=new T.Group();key.name='key-'+a.id;const r={...opts,u:a.u,v:a.v,w:a.w,h:a.h,segments:5};
    const cap=new T.Mesh(patch(T,{...r,z:.012}),face);cap.name='control-surface';cap.userData.spatialAction=a.id;
    const edge=new T.Mesh(sides(T,{...r,front:.012,back:.001}),casing);
    key.add(cap,edge);keys.add(key);records.push({id:a.id,key,cap,edge,disabled:false});
   }
  }
  for(const rec of records){const a=actions.find(a=>a.id===rec.id);rec.disabled=Boolean(a?.disabled);rec.cap.userData.disabled=rec.disabled;}
 }
 function hit(ray){mesh.updateWorldMatrix(true,true);const base=ray.intersectObject(mesh,false)[0];const hits=ray.intersectObjects(records.map(r=>r.cap),false);const front=hits[0];
  if(front&&(!base||front.distance<=base.distance+.002)){return {hit:front,id:front.object.userData.disabled?null:front.object.userData.spatialAction,control:true};}
  return base?{hit:base,id:null,control:false}:null;
 }
 function press(id,time){if(records.some(r=>r.id===id&&!r.disabled)){pressedId=id;pressedAt=time;}}
 function update(hovered,time=NaN){for(const r of records){const isHover=!r.disabled&&hovered.includes(r.id);const travel=r.id===pressedId?root.PulmoXRSpatialExperience?.pulse(time-pressedAt)||0:0;r.key.position.z=(isHover?.003:0)+travel;}}
 const api={setActions,hit,update,press,root:mesh,snapshot:()=>({kind:'CURVED_EXTRUDED_SPATIAL_SURFACE',controls:records.length,depthM:depth,radiusM:radius})};mesh.userData.spatialSurface=api;return api;
}
function monitorHousing(T,root,{width=1,height=.665,depth=.047}={}){
 const metal=new T.MeshStandardMaterial({color:0x384249,metalness:.45,roughness:.56});
 const dark=new T.MeshStandardMaterial({color:0x20272d,metalness:.18,roughness:.8});
 const box=(w,h,d,x,y,z,mat,name)=>{const o=new T.Mesh(new T.BoxGeometry(w,h,d),mat);o.name=name;o.position.set(x,y,z);root.add(o);return o;};
 const shape=new T.Shape(),w=width+.022,h=height+.022,r=.021;
 shape.moveTo(-w/2+r,-h/2);shape.lineTo(w/2-r,-h/2);shape.quadraticCurveTo(w/2,-h/2,w/2,-h/2+r);shape.lineTo(w/2,h/2-r);shape.quadraticCurveTo(w/2,h/2,w/2-r,h/2);shape.lineTo(-w/2+r,h/2);shape.quadraticCurveTo(-w/2,h/2,-w/2,h/2-r);shape.lineTo(-w/2,-h/2+r);shape.quadraticCurveTo(-w/2,-h/2,-w/2+r,-h/2);
 const chassis=new T.Mesh(new T.ExtrudeGeometry(shape,{depth:depth-.008,bevelEnabled:true,bevelSize:.004,bevelThickness:.004,bevelSegments:2,steps:1,curveSegments:5}),dark);chassis.position.z=-depth-.014;chassis.name='monitor-rear-chassis';root.add(chassis);
 const vents=new T.InstancedMesh(new T.BoxGeometry(.004,.037,.0016),new T.MeshBasicMaterial({color:0x13181c}),24);const matrix=new T.Matrix4();for(let i=0;i<24;i++){matrix.makeTranslation((i-11.5)*.013,.208,-depth-.020);vents.setMatrixAt(i,matrix);}vents.instanceMatrix.needsUpdate=true;vents.name='rear-vent-array';root.add(vents);
 box(.24,.105,.009,0,-.16,-depth-.018,metal,'rear-mounting-plate');
 box(width+.022,.011,depth,0,(height+.011)/2,-depth/2-.014,metal,'monitor-top-rail');
 box(width+.022,.011,depth,0,-(height+.011)/2,-depth/2-.014,metal,'monitor-bottom-rail');
 for(const s of [-1,1])box(.011,height,depth,s*(width+.011)/2,0,-depth/2-.014,metal,'monitor-side-rail');
 const indicator=new T.Mesh(new T.SphereGeometry(.0035,12,8),new T.MeshBasicMaterial({color:0x67757d,toneMapped:false}));indicator.position.set(width*.486,-height*.477,.002);indicator.name='monitor-source-status';root.add(indicator);
 return {depthM:depth,videoUnchanged:true,ventCount:24,beveledChassis:true,setStatus(state){indicator.material.color.setHex(state==='live'?0xa9d6cd:state==='unavailable'?0xd3a486:0x67757d);}};
}
function createLaboratory(T,room){
 const root=new T.Group();root.name='pulmosimu-architectural-laboratory';room.add(root);
 const materials={
 porcelain:new T.MeshStandardMaterial({color:0x9da5a7,roughness:.86,metalness:0}),
 wall:new T.MeshStandardMaterial({color:0x555d61,roughness:.94,metalness:0}),
 floor:new T.MeshStandardMaterial({color:0x51585a,roughness:.88,metalness:0}),
 ceiling:new T.MeshStandardMaterial({color:0x50575b,roughness:.92}),
 graphite:new T.MeshStandardMaterial({color:0x292f33,roughness:.8,metalness:.12}),
 titanium:new T.MeshStandardMaterial({color:0x879399,roughness:.48,metalness:.36}),
 textile:new T.MeshStandardMaterial({color:0x727d82,roughness:1}),
 glass:new T.MeshStandardMaterial({color:0x34454e,roughness:.31,metalness:.25}),
 luminous:new T.MeshBasicMaterial({color:0xd8e1e3,toneMapped:false}),
 seam:new T.MeshBasicMaterial({color:0x3c4549,toneMapped:false})
 };
 const span=6.95,half=span/2,height=2.75,unit=new T.BoxGeometry(1,1,1),groups=new Map(),dummy=new T.Object3D();
 function block(mat,x,y,z,w,h,d){if(!groups.has(mat))groups.set(mat,[]);groups.get(mat).push([x,y,z,w,h,d]);}
 block('floor',0,-.035,0,span,.06,span);block('ceiling',0,height,0,span,.07,span);
 for(const s of [-1,1]){
  block('wall',0,height/2,s*half,span,height,.075);block('wall',s*half,height/2,0,.075,height,span);
  block('graphite',0,.065,s*(half-.048),span,.13,.07);block('graphite',s*(half-.048),.065,0,.07,.13,span);
  // Broad, flush wall panels with a recessed shadow gap.
  for(let k=-2;k<=2;k++){block('porcelain',k*1.30,1.50,s*(half-.055),1.275,2.27,.035);block('porcelain',s*(half-.055),1.50,k*1.30,.035,2.27,1.275);}
  block('graphite',0,height-.14,s*(half-.07),span-.15,.16,.09);block('graphite',s*(half-.07),height-.14,0,.09,.16,span-.15);
  block('luminous',0,height-.215,s*(half-.125),span-.25,.025,.045);block('luminous',s*(half-.125),height-.215,0,.045,.025,span-.25);
  for(const x of [-1.72,1.72]){block('graphite',x,height-.08,s*1.35,.34,.075,1.95);block('luminous',x,height-.123,s*1.35,.27,.014,1.86);}
 }
 // Subtle floor material joins instead of a tracking grid.
 for(let i=-2;i<=2;i++){block('seam',i*1.16,.001,0,.002,.002,span-.15);block('seam',0,.001,i*1.16,span-.15,.002,.002);}
 // Perimeter casework: no interactive storage and no claim of physical collision.
 // Interior extent restricted to |x| or |z| >= 3.00 m.
 for(const s of [-1,1]){
  const z=s*3.24;
  block('graphite',-1.69,.10,z,2.36,.16,.45);
  block('porcelain',-1.69,.49,z,2.32,.65,.42);
  block('titanium',-1.69,.84,z,2.44,.05,.48);
  for(let i=0;i<4;i++){const x=-2.56+i*.58;block('seam',x,.49,s*3.022,.006,.61,.012);block('titanium',x+.23,.72,s*3.015,.16,.018,.024);}
  // Tall service cabinet and passive equipment bay.
  block('graphite',2.34,1.02,z,1.22,1.94,.44);
  block('porcelain',2.04,1.03,s*3.025,.57,1.88,.035);block('glass',2.65,1.44,s*3.025,.54,.77,.035);
  block('porcelain',2.65,.53,s*3.025,.54,.95,.035);block('titanium',2.29,1.08,s*3.002,.025,.23,.028);
  for(let i=0;i<3;i++)block('titanium',2.65,1.16+i*.22,s*3.013,.45,.016,.02);
 }
 // Acoustic treatment and an architectural, non-operable door on side walls.
 for(const s of [-1,1]){
  block('graphite',s*3.405,1.35,-1.64,.045,2.37,1.80);
  for(let i=0;i<20;i++)block('textile',s*3.375,1.35,-2.48+i*.088,.026,2.29,.055);
  block('graphite',s*3.403,1.07,1.60,.055,2.14,1.14);
  block('porcelain',s*3.365,1.06,1.60,.045,2.07,1.05);
  block('glass',s*3.334,1.49,1.60,.014,.48,.32);
  block('titanium',s*3.324,.98,1.25,.036,.027,.15);
 }
 let instances=0;
 for(const [key,records] of groups){const mesh=new T.InstancedMesh(unit,materials[key],records.length);mesh.name='architecture-'+key;records.forEach(([x,y,z,w,h,d],i)=>{dummy.position.set(x,y,z);dummy.scale.set(w,h,d);dummy.updateMatrix();mesh.setMatrixAt(i,dummy.matrix);});mesh.instanceMatrix.needsUpdate=true;mesh.frustumCulled=false;root.add(mesh);instances+=records.length;}
 // A single identification plaque, physically flush with the wall.
 const labelCanvas=document.createElement('canvas');labelCanvas.width=1024;labelCanvas.height=256;const c=labelCanvas.getContext('2d');
 c.fillStyle='#262d31';c.fillRect(0,0,1024,256);c.fillStyle='#ebeeeb';c.font='500 70px system-ui';c.fillText('PulmoSIMU',45,104);c.fillStyle='#c4d0d4';c.font='400 26px system-ui';c.fillText('RESPIRATORY ENGINEERING LABORATORY',48,163);c.fillStyle='#92a4ad';c.font='400 21px system-ui';c.fillText('PROCEDURAL SIMULATION  /  COMPUTATIONAL RESEARCH',48,209);
 const texture=new T.CanvasTexture(labelCanvas);texture.encoding=T.sRGBEncoding;
 const sign=new T.Mesh(new T.PlaneGeometry(1.54,.385),new T.MeshBasicMaterial({map:texture,toneMapped:false}));sign.name='physical-wall-identification';sign.position.set(0,2.25,-half+.089);root.add(sign);
 return {root,snapshot:()=>({kind:'AUTHORED_VIRTUAL_LABORATORY',spanM:span,heightM:height,physicalRoomReconstructed:false,finish:'PORCELAIN_GRAPHITE_TITANIUM',instancedDetails:true,architectureInstances:instances,architectureBatches:groups.size+1,perimeterCasework:true,operableDoors:false,physicalCollisionValidated:false,additionalSceneLights:0})};
}
root.PulmoXRSpatialGeometry=Object.freeze({point,patch,sides,create,monitorHousing,createLaboratory});
})(typeof window!=='undefined'?window:globalThis);
