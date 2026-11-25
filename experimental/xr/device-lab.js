/* XR24–26: bounded reference optics, source-constrained shaft and
 * working-channel visual mechanism. No measured material coefficients,
 * tissue forces, actual sampling, fluid removal or clinical device validation. */
(function(root){
'use strict';
const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
const finite=Number.isFinite;
function triggerIntent({atBranch,side,tool='none'}){
 if(atBranch&&['left','right'].includes(side))return 'BRANCH';
 if(side!=='right')return 'NONE';
 return tool==='none'?'CLINICAL':'ACCESSORY';
}
function allowNativeDisplay({opticsActive,radialEnabled,nativeOptIn}){return Boolean(nativeOptIn&&!opticsActive&&!radialEnabled)}
const OPTICS=Object.freeze({distortion:.055,focusNearMm:2,focusFarMm:50});
function distort(x,y,k=OPTICS.distortion){const r=x*x+y*y;return [x*(1+k*r),y*(1+k*r)]}
function linearDepth(z,near,far){return near*far/(far-(far-near)*clamp(z,0,1))}
function blurMm(mm){return clamp(Math.max(0,(2-mm)/2,(mm-50)/50),0,1)}
function opticalMaterial(THREE,target){
 const mat=new THREE.ShaderMaterial({
  uniforms:{image:{value:target.texture},depthMap:{value:target.depthTexture},aspect:{value:16/9},
   texel:{value:new THREE.Vector2(1/target.width,1/target.height)},nearPlane:{value:.001},farPlane:{value:8},
   mmPerWorld:{value:100},distortion:{value:OPTICS.distortion},enabled:{value:1}},
  vertexShader:'varying vec2 pUV; void main(){pUV=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
  fragmentShader:`
   varying vec2 pUV; uniform sampler2D image,depthMap; uniform vec2 texel;
   uniform float aspect,nearPlane,farPlane,mmPerWorld,distortion,enabled;
   void main(){
    vec2 p=pUV*2.-1.; vec2 q=vec2(p.x,p.y/aspect);
    float r2=dot(q,q); vec2 uv=.5+.5*p*(1.+distortion*r2*enabled);
    if(any(lessThan(uv,vec2(0.)))||any(greaterThan(uv,vec2(1.)))){gl_FragColor=vec4(.008,.008,.008,1.);return;}
    vec3 col=texture2D(image,uv).rgb;
    float z=texture2D(depthMap,uv).r;
    float mm=nearPlane*farPlane/max(.0000001,farPlane-(farPlane-nearPlane)*z)*mmPerWorld;
    float coc=clamp(max((2.-mm)/2.,(mm-50.)/50.),0.,1.)*enabled;
    vec2 d=texel*(1.+2.5*coc);
    vec3 blurred=(texture2D(image,clamp(uv+vec2(d.x,0.),0.,1.)).rgb+
       texture2D(image,clamp(uv-vec2(d.x,0.),0.,1.)).rgb+
       texture2D(image,clamp(uv+vec2(0.,d.y),0.,1.)).rgb+
       texture2D(image,clamp(uv-vec2(0.,d.y),0.,1.)).rgb)*.25;
    col=mix(col,blurred,coc*.55);
    // Bounded static response, not measured auto-exposure or lens calibration.
    float lum=dot(col,vec3(.2126,.7152,.0722));
    col*=mix(1.,clamp(1.12-.23*lum,.87,1.12),enabled);
    col*=1.-.13*clamp(r2,0.,1.)*enabled;
    gl_FragColor=vec4(col,1.);
    #include <encodings_fragment>
   }`,
   side:THREE.DoubleSide,depthTest:false,depthWrite:false,toneMapped:false
 });
 return Object.freeze({material:mat,update({near,far,scale,mmPerUnit,enabled=true}){
  const u=mat.uniforms;u.nearPlane.value=near;u.farPlane.value=far;u.mmPerWorld.value=mmPerUnit/Math.max(scale,1e-6);
  u.aspect.value=target.width/target.height;u.texel.value.set(1/target.width,1/target.height);u.enabled.value=enabled?1:0;
 }});
}
function ancestry(segments,id){const out=[];for(let i=0;i<64&&id!=null&&!out.includes(id);i++){
 if(!segments[id])break;out.unshift(id);id=Object.values(segments).find(s=>s.children?.includes(id))?.id;}return out}
function shaftSamples(THREE,{segments,curves,id,t,mmPerUnit,maxMm=600}){
 const ids=ancestry(segments,id);const points=[];let distance=0;
 for(const key of ids){const curve=curves[key];if(!curve)continue;
  const last=key===id?clamp(t,0,1):1;
  const n=Math.max(1,Math.ceil(curve.getLength()*mmPerUnit*last/2));
  for(let i=0;i<=n;i++){const p=curve.getPoint(last*i/n);const prev=points[points.length-1];
   if(prev&&p.distanceTo(prev)<1e-7)continue;points.push(p);
  }}
 const out=[];for(let i=points.length-1;i>=0;i--){
  if(out.length){const d=points[i].distanceTo(out[out.length-1])*mmPerUnit;if(distance+d>maxMm){
   const remain=maxMm-distance;if(remain>0)out.push(out[out.length-1].clone().lerp(points[i],remain/d));distance=maxMm;break;}distance+=d;}
  out.push(points[i]);}
 return {points:out.reverse(),lengthMm:distance};
}
function curvature(a,b,c,mmPerUnit){
 const u=b.clone().sub(a),v=c.clone().sub(b);
 const len=(u.length()+v.length())*.5*mmPerUnit;
 return len>1e-6?u.angleTo(v)/len:0;
}
function assessShaft(points,{field,radius,mmPerUnit,curvatureLimit=.055}){
 if(!points.length||!field||!(radius>0)||!(mmPerUnit>0))return {state:'UNAVAILABLE',minMarginMm:null,maxCurvature:null};
 let margin=Infinity,k=0;
 for(let i=0;i<points.length;i++){
  margin=Math.min(margin,(-field(points[i].toArray())-radius)*mmPerUnit);
  if(i>0&&i<points.length-1)k=Math.max(k,curvature(points[i-1],points[i],points[i+1],mmPerUnit));
 }
 return {state:margin<0?'GEOMETRIC_INTERFERENCE':k>curvatureLimit?'BENDING_LIMIT_EXCEEDED':'REFERENCE_ENVELOPE_CLEAR',minMarginMm:margin,maxCurvature:k,forceValidated:false};
}
function createShaft(THREE,parent){
 const mat=new THREE.MeshStandardMaterial({color:0x25333b,roughness:.64,metalness:.05});
 const mesh=new THREE.Mesh(new THREE.BufferGeometry(),mat);mesh.visible=false;mesh.name='XR25 source constrained inserted shaft';parent.add(mesh);
 let key=null,evidence={state:'UNAVAILABLE'},lastUpdate=-Infinity;
 function update(args){
  const next=[args.id,Math.round(args.t*200),args.diameterMm,args.mmPerUnit].join('/');
  if(next===key)return evidence;
  const now=Number.isFinite(args.timeMs)?args.timeMs:0;
  if(now>=lastUpdate&&now-lastUpdate<100)return evidence;
  key=next;lastUpdate=now;
  const s=shaftSamples(THREE,args);const radius=args.diameterMm/2/args.mmPerUnit;
  let solved=null,display=s.points;
  if(s.points.length>=3&&args.field&&root.PulmoXRElasticRod){try{
   solved=root.PulmoXRElasticRod.solve({points:s.points.map(p=>p.toArray().map(v=>v*args.mmPerUnit)),field:p=>args.field(p.map(v=>v/args.mmPerUnit))*args.mmPerUnit,radiusMm:args.diameterMm/2});
   const candidate=solved.points.map(p=>new THREE.Vector3(...p).multiplyScalar(1/args.mmPerUnit));
   // Keep the established centerline on any sampled interference or excess length change.
   if(solved.minMarginMm>=0&&Math.abs(solved.lengthMm-solved.referenceLengthMm)<Math.max(.5,solved.referenceLengthMm*.03))display=candidate;
  }catch(error){solved={state:'SOLVER_FALLBACK',reason:String(error?.message||error).slice(0,80)};}}
  mesh.visible=display.length>=2;
  if(mesh.visible){
   const curve=new THREE.CatmullRomCurve3(display,false,'centripetal');
   const g=new THREE.TubeGeometry(curve,Math.min(180,Math.max(8,s.points.length*2)),radius,10,false);
   mesh.geometry.dispose();mesh.geometry=g;
  }
  const {points:discardedPoints,...rodEvidence}=solved||{};
  evidence={...assessShaft(display,{field:args.field,radius,mmPerUnit:args.mmPerUnit}),insertedLengthMm:s.lengthMm,elasticRod:{...rodEvidence,displayed:display!==s.points,model:'DISCRETE_ELASTIC_CENTERLINE',hardwareCommand:false}};
  return evidence;
 }
 return {mesh,update,hide(){mesh.visible=false;key=null;lastUpdate=-Infinity},snapshot(){return evidence}};
}
const TOOLS=Object.freeze([{id:'none',label:'EMPTY CHANNEL',od:0},{id:'forceps',label:'BIOPSY FORCEPS',od:1.8},{id:'brush',label:'SAMPLING BRUSH',od:1.5},{id:'suction',label:'SUCTION INDICATOR',od:0}]);
function createChannel(){
 let index=0,extension=0,open=false,suction=false;
 function reset(){index=0;extension=0;open=false;suction=false}
 function snapshot(){return Object.freeze({tool:TOOLS[index].id,label:TOOLS[index].label,odMm:TOOLS[index].od,extensionMm:extension,jawsOpen:open,suction,actualSampling:false,actualFluidRemoval:false})}
 function cycle(){resetExtension();index=(index+1)%TOOLS.length;return snapshot()}
 function resetExtension(){extension=0;open=false;suction=false}
 function extend(delta,channelMm=2){const t=TOOLS[index];if(!['forceps','brush'].includes(t.id)||t.od>channelMm||!finite(delta))return snapshot();extension=clamp(extension+delta,0,12);if(extension<3)open=false;return snapshot()}
 function action(){const t=TOOLS[index].id;if(t==='forceps'&&extension>=3)open=!open;else if(t==='suction')suction=!suction;return snapshot()}
 return {reset,snapshot,cycle,extend,action};
}
function createAccessory(THREE,scene){
 const g=new THREE.Group();g.name='XR26 working channel visual';g.visible=false;scene.add(g);
 const steel=new THREE.MeshStandardMaterial({color:0x9aaab1,roughness:.42,metalness:.65});
 const polymer=new THREE.MeshStandardMaterial({color:0x3a4550,roughness:.8});
 const mesh=(geo,mat,parent=g)=>{const m=new THREE.Mesh(geo,mat);parent.add(m);return m};
 const stem=mesh(new THREE.CylinderGeometry(1,1,1,12),polymer);stem.rotation.x=Math.PI/2;
 const jaws=new THREE.Group();g.add(jaws);
 const jawA=mesh(new THREE.BoxGeometry(1,1,1),steel,jaws),jawB=mesh(new THREE.BoxGeometry(1,1,1),steel,jaws);
 const brush=mesh(new THREE.CylinderGeometry(1,1,1,16),polymer);brush.rotation.x=Math.PI/2;
 function update(camera,state,worldPerMm){
  g.visible=['forceps','brush'].includes(state.tool)&&state.extensionMm>0;
  if(!g.visible)return;
  camera.updateMatrixWorld(true);g.position.copy(camera.getWorldPosition(new THREE.Vector3()));g.quaternion.copy(camera.getWorldQuaternion(new THREE.Quaternion()));
  const u=worldPerMm,e=state.extensionMm,rad=state.odMm*.5;
  const x=1.05*u,y=-.9*u;
  stem.position.set(x,y,-e*u*.5);stem.scale.set(rad*u,e*u,rad*u);
  jaws.visible=state.tool==='forceps'&&e>=3;brush.visible=state.tool==='brush'&&e>=2;
  jaws.position.set(x,y,-e*u);const spread=state.jawsOpen?.65:.12;
  for(const [j,sign] of [[jawA,1],[jawB,-1]]){j.scale.set(.55*u,.65*u,2.0*u);j.position.set(sign*spread*u,0,-.8*u);j.rotation.y=sign*(state.jawsOpen?.42:0)}
  brush.position.set(x,y,-e*u);brush.scale.set(rad*u,2*u,rad*u);
 }
 return {group:g,update};
}
root.PulmoXRDeviceLab=Object.freeze({OPTICS,triggerIntent,allowNativeDisplay,distort,linearDepth,blurMm,opticalMaterial,ancestry,shaftSamples,assessShaft,createShaft,createChannel,createAccessory});
})(typeof window!=='undefined'?window:globalThis);
