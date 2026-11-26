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
 add(patch(T,{...opts,v:0,h:deckStart,z:.001}),face,'spatial-information-display');
 if(footerStart<1)add(patch(T,{...opts,v:footerStart,h:1-footerStart,z:.001}),face,'spatial-footer-display');
 add(patch(T,{...opts,z:-depth}),back,'spatial-back');
 add(sides(T,{...opts,front:0,back:-depth}),casing,'spatial-machined-edge');
 const keys=new T.Group();keys.name='spatial-physical-controls';mesh.add(keys);
 let signature='',records=[];
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
 function update(hovered){for(const r of records){const isHover=!r.disabled&&hovered.includes(r.id);r.key.position.z=isHover?.003:0;}}
 const api={setActions,hit,update,root:mesh,snapshot:()=>({kind:'CURVED_EXTRUDED_SPATIAL_SURFACE',controls:records.length,depthM:depth,radiusM:radius})};mesh.userData.spatialSurface=api;return api;
}
function monitorHousing(T,root,{width=1,height=.665,depth=.047}={}){
 const metal=new T.MeshStandardMaterial({color:0x384249,metalness:.45,roughness:.56});
 const dark=new T.MeshStandardMaterial({color:0x20272d,metalness:.18,roughness:.8});
 const box=(w,h,d,x,y,z,mat,name)=>{const o=new T.Mesh(new T.BoxGeometry(w,h,d),mat);o.name=name;o.position.set(x,y,z);root.add(o);return o;};
 box(width+.022,height+.022,depth,0,0,-depth/2-.014,dark,'monitor-rear-chassis');
 box(width+.022,.011,depth,0,(height+.011)/2,-depth/2-.014,metal,'monitor-top-rail');
 box(width+.022,.011,depth,0,-(height+.011)/2,-depth/2-.014,metal,'monitor-bottom-rail');
 for(const s of [-1,1])box(.011,height,depth,s*(width+.011)/2,0,-depth/2-.014,metal,'monitor-side-rail');
 return {depthM:depth,videoUnchanged:true};
}
root.PulmoXRSpatialGeometry=Object.freeze({point,patch,sides,create,monitorHousing});
})(typeof window!=='undefined'?window:globalThis);
