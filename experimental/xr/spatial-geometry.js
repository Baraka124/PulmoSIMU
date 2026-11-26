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
 wall:new T.MeshStandardMaterial({color:0x343a3d,roughness:.95,metalness:0,side:T.DoubleSide}),
 floor:new T.MeshStandardMaterial({color:0x383d40,roughness:.94,metalness:0}),
 ceiling:new T.MeshStandardMaterial({color:0x272c30,roughness:.9,side:T.DoubleSide}),
 frame:new T.MeshStandardMaterial({color:0x252a2e,roughness:.66,metalness:.25}),
 inset:new T.MeshStandardMaterial({color:0x555b5e,roughness:.88,metalness:.04}),
 luminous:new T.MeshBasicMaterial({color:0xc0cbd0,toneMapped:false})
 };
 const box=(w,h,d,x,y,z,m,name)=>{const o=new T.Mesh(new T.BoxGeometry(w,h,d),materials[m]);o.position.set(x,y,z);o.name=name;root.add(o);return o;};
 // Four actual wall surfaces, ceiling and acoustic/wall treatment; no
 // physical-room geometry is inferred, and the shell is hidden in MR.
 const span=6.95,half=span/2,height=2.75;
 box(span,.06,span,0,-.035,0,'floor','matte-floor');
 box(span,.07,span,0,height,0,'ceiling','ceiling');
 for(const s of [-1,1]){
  box(span,height,.075,0,height/2,s*half,'wall','wall-ns');
  box(.075,height,span,s*half,height/2,0,'wall','wall-ew');
  box(span,.11,.085,0,.055,s*(half-.04),'frame','wall-base-ns');
  box(.085,.11,span,s*(half-.04),.055,0,'frame','wall-base-ew');
  box(2.35,1.43,.055,0,1.49,s*(half-.09),'inset','acoustic-panel-ns');
  box(.055,1.43,2.35,s*(half-.09),1.49,0,'inset','acoustic-panel-ew');
  box(.085,.035,4.7,s*1.9,height-.065,0,'luminous','ceiling-linear-diffuser');
 }
 // Restrained architectural detail, instanced to limit draw calls. Wall
 // treatment remains within the authored enclosure, outside the work area.
 const matrix=new T.Matrix4();
 function batch(name,geometry,material,positions){const o=new T.InstancedMesh(geometry,material,positions.length);positions.forEach((p,i)=>{matrix.makeTranslation(...p);o.setMatrixAt(i,matrix);});o.instanceMatrix.needsUpdate=true;o.name=name;root.add(o);return o;}
 const slats=[];for(const s of [-1,1])for(let i=0;i<19;i++)slats.push([(i-9)*.115,1.49,s*(half-.13)]);
 batch('acoustic-wall-fins',new T.BoxGeometry(.021,1.41,.018),materials.frame,slats);
 const slatsEW=[];for(const s of [-1,1])for(let i=0;i<19;i++)slatsEW.push([s*(half-.13),1.49,(i-9)*.115]);
 batch('acoustic-side-fins',new T.BoxGeometry(.018,1.41,.021),materials.frame,slatsEW);
 const joints=[];for(let i=-2;i<=2;i++){joints.push([i*1.17,.001,0]);}
 const jointMat=new T.MeshBasicMaterial({color:0x2c3134,toneMapped:false});batch('floor-material-joints',new T.BoxGeometry(.003,.002,span-.12),jointMat,joints);
 // Ceiling cove is baked visual illumination, not an additional scene light.
 // Vertex-color gradients make the wall wash visible even in low light.
 const washGeo=new T.PlaneGeometry(span-.16,.48,1,5);const pos=washGeo.attributes.position,cols=[];
 for(let i=0;i<pos.count;i++){const t=(pos.getY(i)+.24)/.48;const value=.19+.20*Math.pow(t,1.7);cols.push(value,value*1.025,value*1.045);}
 washGeo.setAttribute('color',new T.Float32BufferAttribute(cols,3));
 const washMat=new T.MeshBasicMaterial({vertexColors:true,side:T.DoubleSide,toneMapped:false});
 for(const side of [-1,1]){const north=new T.Mesh(washGeo,washMat);north.position.set(0,height-.27,side*(half-.06));north.name='ceiling-wall-wash';root.add(north);const east=new T.Mesh(washGeo,washMat);east.rotation.y=Math.PI/2;east.position.set(side*(half-.06),height-.27,0);east.name='ceiling-side-wash';root.add(east);}
 const labelCanvas=document.createElement('canvas');labelCanvas.width=1024;labelCanvas.height=192;
 const ctx=labelCanvas.getContext('2d');ctx.fillStyle='#2a3034';ctx.fillRect(0,0,1024,192);ctx.font='500 42px system-ui';ctx.fillStyle='#e5e9eb';ctx.fillText('PulmoSIMU',44,75);ctx.font='400 29px system-ui';ctx.fillStyle='#aebac0';ctx.fillText('RESPIRATORY SIMULATION  /  RESEARCH LABORATORY',44,138);
 const texture=new T.CanvasTexture(labelCanvas);texture.encoding=T.sRGBEncoding;
 const sign=new T.Mesh(new T.PlaneGeometry(1.8,.3375),new T.MeshBasicMaterial({map:texture,toneMapped:false}));sign.position.set(0,2.36,-half+.055);root.add(sign);
 // Fixed neutral room orientation cues; existing room lighting is unchanged.
 return {root,snapshot:()=>({kind:'AUTHORED_VIRTUAL_LABORATORY',spanM:span,heightM:height,physicalRoomReconstructed:false,finish:'MATTE_ACOUSTIC_AND_MACHINED_SURFACES',instancedDetails:true,bakedWallWash:true,additionalSceneLights:0})};
}
root.PulmoXRSpatialGeometry=Object.freeze({point,patch,sides,create,monitorHousing,createLaboratory});
})(typeof window!=='undefined'?window:globalThis);
