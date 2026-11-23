/* PulmoSIMU A1 — source-linked anatomic inspection. Local-only uploads.
 * No patient study bytes are uploaded or persisted. No inferred bronchi,
 * disease labels, fissures or geometry-linked physiological claims.
 */
(function(){
'use strict';
const $=id=>document.getElementById(id);
const state={volume:null,qa:null,structureQA:null,gltf:null,model:null,meshNames:[],ctSha:null,meshSha:null,cross:[256,256,75],surfaceOpacity:1,
  window:'lung',clip:false,clipping:[],fileNames:[],numSlices:0};
const huWindows={lung:[-600,1500],mediastinum:[40,400],bone:[350,1700]};
const THREE=window.THREE;
const has3d=Boolean(THREE&&THREE.GLTFLoader&&THREE.OrbitControls);
function setMessage(text,kind='note'){
  const node=$('status-text');node.textContent=text;node.dataset.state=kind;
}
function short(n){return Number(n).toFixed(2)}
function escapeHtml(text){return String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
const fileLabels=['glb','nii','json'];
// Known A1 research mesh has failed source-RAS anatomical laterality QA.
// This fingerprint also catches the older private manifest which predates A2.
const A2_FLAGGED_GLBSHA='5e4457003b45f9d7f78e87239f935c4299e4c3b947a9a3ffda0e273216228a0f';
function a2LateralityConflict(){
  return state.meshSha===A2_FLAGGED_GLBSHA ||
    state.qa?.mesh_sha256===A2_FLAGGED_GLBSHA ||
    state.qa?.quality_gates?.laterality_labels==='FAIL_A2_SOURCE_RAS_CONFLICT';
}
const a2Warning='A2 QUALITY GATE FAILED: the original GLB RIGHT and LEFT labels conflict with source NIfTI RAS laterality (85.2% and 99.0% opposite-side mesh area). Source-image review required. These are mesh names, NOT verified anatomical sides. Bronchoscope and clinical coupling remain blocked.';
const A2_CANDIDATE_SHA='8f8bea9a95fc21a44ca10244c91c249d2836539bf119973bcb18c614c9886d2f';
function a2CandidateReviewPending(){
 return state.meshSha===A2_CANDIDATE_SHA ||
   state.qa?.mesh_sha256===A2_CANDIDATE_SHA ||
   state.qa?.quality_gates?.laterality_labels==='CANDIDATE_CORRECTED_RAS_INDEPENDENT_REVIEW_PENDING';
}
const a2CandidateWarning='A2 VERSIONED CANDIDATE: mesh node names follow NIfTI RAS laterality but segmentation is NOT independently validated. CT-voxel QA found 16.3 mL overlap between candidate lungs; 14.8% of left-mesh surface AREA (1.06% of left-mesh voxel VOLUME) crosses RAS midline. Five lobes, vessels, airway lumen, clinical coupling and patient-specific bronchoscopy remain BLOCKED.';
function updateResearchGates(){
  if(!window.PulmoSourceAnatomyGate?.assess)return;
  const status=window.PulmoSourceAnatomyGate.assess({
    manifest:state.qa,ctSha:state.ctSha,meshSha:state.meshSha,
    structureQA:state.structureQA
  });
  $('source-gates').textContent=status.researchStageStatus.map(s=>
    s.id+' / '+s.name+' — '+s.status.replaceAll('_',' ')
  ).join('\n')+
  '\nClinical anatomy validated: NO · CT-specific bronchoscopy: '+status.CTLinkedBronchoscopyStatus;
}
function applyQa(){
  $('source-state').textContent=a2LateralityConflict()
    ?'A2 LATERALITY FAIL · UNVALIDATED'
    :a2CandidateReviewPending()?'A2 CANDIDATE · CLINICAL QA PENDING'
    :state.qa?String(state.qa.validation||'UNVALIDATED').replaceAll('_',' '):'No manifest loaded';
  if(state.qa){
    $('source-shape').textContent=(state.qa.source_shape_vox||[]).join(' × ')||'—';
    $('source-spacing').textContent=(state.qa.spacing_mm||[]).map(x=>Number(x).toFixed(3)).join(' / ')+' mm';
    $('volume-stats').textContent=Object.entries(state.qa.structures||{}).map(([side,s])=>
      side+' '+(s.airspace_volume_l_approx||'—')+' L').join(' · ')||'No volume summary';
    $('qa-details').textContent=(a2LateralityConflict()?[a2Warning]:
      a2CandidateReviewPending()?[a2CandidateWarning]:[])
      .concat(state.qa.research_flags||state.qa.known_limitations||state.qa.review_notes||[]).join('\n• ');
  }else{
    $('source-shape').textContent='—';$('source-spacing').textContent='—';
    $('volume-stats').textContent='No source QA loaded';
    $('qa-details').textContent=a2LateralityConflict()?a2Warning:
      a2CandidateReviewPending()?a2CandidateWarning:
      'No segmentation review or source registration has been established.';
  }
  validate();updateResearchGates();
}
function validate(){
  const status=state.volume&&state.qa?window.PulmoNiftiSource.matchQA(state.volume,state.qa):null;
  const ctHashOK=Boolean(state.qa&&state.ctSha&&state.qa.source_sha256===state.ctSha);
  const meshHashOK=Boolean(state.qa&&state.meshSha&&state.qa.mesh_sha256===state.meshSha);
  const aligned=Boolean(status?.valid&&ctHashOK&&meshHashOK&&state.qa?.model_from_ras_mm&&state.model);
  $('registration').textContent=!status?'Awaiting NIfTI + manifest':!status.valid?'Mismatch · '+status.reason:
    !ctHashOK?'CT SHA-256 not verified':!meshHashOK?'GLB SHA-256 not verified':
    a2LateralityConflict()
      ?'File hashes matched · A2 LATERALITY FAIL · tissue segmentation UNVALIDATED'
      :a2CandidateReviewPending()
      ?'Exact files aligned · A2 candidate labels reviewed computationally · segmentation/clinical QA PENDING'
      :'Exact study files and header matched · tissue segmentation UNVALIDATED';
  $('registration').dataset.state=a2LateralityConflict()?'error':
    a2CandidateReviewPending()?'pending':aligned?'valid':status?'error':'pending';
  const enabled=aligned;
  $('clip-toggle').disabled=!enabled;
  $('cross-3d').disabled=!enabled;
  $('section-plane').disabled=!enabled;
  if(!enabled){state.clip=false;$('clip-toggle').checked=false;}
  updateMarker();applyClipping();updateResearchGates();
}
let scene=null,renderer=null,camera=null,orbit=null,modelGroup=null,marker=null,
  scenePlane=null,workLight=null,originalMaterials=[];
function start3d(){
  if(!has3d){setMessage('3D libraries could not load. The CT slice viewer remains available.','error');return;}
  renderer=new THREE.WebGLRenderer({canvas:$('three-canvas'),antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.75));
  renderer.setClearColor(0x111d25,1);
  renderer.outputEncoding=THREE.sRGBEncoding;
  renderer.localClippingEnabled=true;
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(38,1,.001,20);camera.position.set(.30,.20,.58);
  orbit=new THREE.OrbitControls(camera,renderer.domElement);
  orbit.enableDamping=true;orbit.dampingFactor=.085;orbit.enablePan=true;
  orbit.minDistance=.1;orbit.maxDistance=2.5;
  scene.add(new THREE.HemisphereLight(0xe2ecf1,0x253541,.90));
  const key=new THREE.DirectionalLight(0xfff6ed,1.6);key.position.set(-.2,.5,.7);scene.add(key);
  const rim=new THREE.DirectionalLight(0x8cbbcb,.65);rim.position.set(.4,.2,-.5);scene.add(rim);
  modelGroup=new THREE.Group();scene.add(modelGroup);
  marker=new THREE.Group();const core=new THREE.Mesh(new THREE.SphereGeometry(.0035,16,12),
    new THREE.MeshBasicMaterial({color:0xe4c08d,depthTest:false}));marker.add(core);
  for(const axis of [new THREE.Vector3(1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,1)]){
    const ends=[axis.clone().multiplyScalar(-.018),axis.clone().multiplyScalar(.018)];
    const g=new THREE.BufferGeometry().setFromPoints(ends);
    const l=new THREE.Line(g,new THREE.LineBasicMaterial({color:0xe4c08d,depthTest:false}));l.renderOrder=10;marker.add(l);
  }
  marker.visible=false;marker.renderOrder=10;scene.add(marker);
  window.addEventListener('resize',resize);
  if(window.ResizeObserver)new ResizeObserver(resize).observe($('three-stage'));
  resize();
  function loop(){requestAnimationFrame(loop);orbit.update();renderer.render(scene,camera);}
  loop();
}
function resize(){if(!renderer||!camera)return;
  const el=$('three-stage'),w=Math.max(240,el.clientWidth),h=Math.max(220,el.clientHeight);
  renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();
}
function clear3d(){if(!modelGroup)return;
  for(const child of [...modelGroup.children])modelGroup.remove(child);
  state.model=null;state.meshNames=[];originalMaterials=[];
}
function viewReset(){if(!state.model||!orbit)return;
  const box=new THREE.Box3().setFromObject(state.model);
  const center=box.getCenter(new THREE.Vector3());const size=box.getSize(new THREE.Vector3());
  const extent=Math.max(size.x,size.y,size.z),distance=Math.max(.35,extent*2.2);
  camera.position.copy(center).add(new THREE.Vector3(distance*.38,distance*.18,distance));
  orbit.target.copy(center);orbit.update();camera.near=.0005;camera.far=10;camera.updateProjectionMatrix();
}
function updateStructureVisibility(){if(!state.model)return;
  const choice=$('structure-select').value;
  state.model.traverse(obj=>{
    if(!obj.isMesh)return;
    if(choice==='both'){obj.visible=true;return;}
    if(choice.startsWith('mesh:')){
      obj.visible=Number(choice.slice(5))===obj.userData.pulmoSurfaceIndex;return;
    }
    const name=obj.name.toLowerCase();
    obj.visible=choice==='right'?name.includes('right'):name.includes('left');
  });
}
function applySurfaceOpacity(){
  const slider=$('surface-opacity');
  if(!slider)return;
  const value=Number(slider.value)/100;
  state.surfaceOpacity=value;
  $('opacity-readout').textContent=Math.round(100*value)+'%';
  for(const m of originalMaterials){
    if(!m)return;
    if(!Number.isFinite(m.userData.pulmoBaseOpacity)){
      m.userData.pulmoBaseOpacity=m.opacity??1;
      m.userData.pulmoBaseTransparent=m.transparent;
      m.userData.pulmoBaseDepthWrite=m.depthWrite;
    }
    m.opacity=m.userData.pulmoBaseOpacity*value;
    m.transparent=value<.999||m.userData.pulmoBaseTransparent;
    m.depthWrite=value>=.999?m.userData.pulmoBaseDepthWrite:false;
    m.needsUpdate=true;
  }
}
async function sha256(buffer){
  if(!globalThis.crypto?.subtle)throw new Error('SHA-256 unavailable: use HTTPS or a localhost server for verified alignment');
  const hash=await crypto.subtle.digest('SHA-256',buffer);
  return [...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('');
}
async function loadModel(file){
  if(!has3d)throw new Error('Three.js GLTF library unavailable; verify internet access.');
  if(!/\.glb$/i.test(file.name))throw new Error('Choose a binary .glb model');
  if(file.size>160*1024*1024)throw new Error('GLB too large for browser inspection');
  const bytes=await file.arrayBuffer();
  state.meshSha=await sha256(bytes);
  const gltf=await new Promise((resolve,reject)=>new THREE.GLTFLoader().parse(bytes,'',resolve,reject));
  clear3d();
  state.gltf=gltf;state.model=gltf.scene;modelGroup.add(gltf.scene);
  const meshes=[];
  gltf.scene.traverse(obj=>{
    if(!obj.isMesh)return;
    meshes.push(obj);
    obj.frustumCulled=true;
    // Preserve authored GLB materials, vertex colour data, geometry and units.
    const materials=Array.isArray(obj.material)?obj.material:[obj.material];
    for(const m of materials){if(!m)continue;originalMaterials.push(m);m.clippingPlanes=[];m.clipShadows=false;}
  });
  if(!meshes.length)throw new Error('GLB has no surface meshes');
  state.meshNames=meshes.map(m=>m.name);
  const select=$('structure-select');
  while(select.options.length>3)select.remove(select.options.length-1);
  meshes.forEach((mesh,i)=>{
    mesh.userData.pulmoSurfaceIndex=i;
    select.add(new Option('Isolate · '+(mesh.name||('Structure '+(i+1))), 'mesh:'+i));
  });
  select.value='both';
  $('surface-opacity').disabled=false;
  $('surface-opacity').value='100';applySurfaceOpacity();
  $('mesh-count').textContent=meshes.length+' source mesh'+(meshes.length===1?'':'es');
  $('mesh-faces').textContent=meshes.reduce((v,m)=>v+(m.geometry.index?m.geometry.index.count/3:m.geometry.attributes.position.count/3),0).toLocaleString()+' triangles';
  $('three-placeholder').hidden=true;
  $('structure-select').disabled=false;
  viewReset();applyQa();updateStructureVisibility();
  setMessage(a2LateralityConflict()?
    'A2 LATERALITY QUALITY FAILURE: original surface labels are reversed relative to CT RAS. No patient-specific use.':
    a2CandidateReviewPending()?
    'A2 candidate: source RAS labels corrected, but anatomical mask and 16.3 mL bilateral overlap are unvalidated. Do not use for CT-linked bronchoscopy.':
    'Local GLB loaded. Geometry inspection is not clinical segmentation validation.',
    a2LateralityConflict()?'error':a2CandidateReviewPending()?'note':'ok');
}
async function loadNifti(file){
  if(!/\.nii$/i.test(file.name))throw new Error('Choose an uncompressed NIfTI .nii file');
  if(file.size>800*1024*1024)throw new Error('CT file exceeds local browser size limit');
  setMessage('Reading local 3D NIfTI volume…');
  const bytes=await file.arrayBuffer();
  state.ctSha=await sha256(bytes);
  state.volume=window.PulmoNiftiSource.readNifti(bytes);
  state.cross=state.volume.dims.map(n=>Math.floor(n/2));
  $('ct-state').textContent='Loaded locally · '+(file.size/1024/1024).toFixed(1)+' MiB';
  $('source-shape').textContent=state.volume.dims.join(' × ');
  $('source-spacing').textContent=state.volume.spacing.map(x=>x.toFixed(3)).join(' / ')+' mm';
  $('slice-axial').max=state.volume.dims[2]-1;
  $('slice-axial').value=state.cross[2];
  renderSlices();validate();
  setMessage('NIfTI read locally. Compare all segmentation boundaries against the original image.','ok');
}
async function loadManifest(file){
  if(!/\.json$/i.test(file.name))throw new Error('Choose a reconstruction manifest .json file');
  const data=JSON.parse(await file.text());
  if(!Array.isArray(data.source_shape_vox)||!Array.isArray(data.sform_RAS_mm)||
     String(data.validation||'').toUpperCase().includes('CLINICAL')){
    throw new Error('This is not a supported research CT geometry manifest.');
  }
  state.qa=data;applyQa();renderSlices();
  setMessage(a2LateralityConflict()?a2Warning:
    a2CandidateReviewPending()?a2CandidateWarning:
    'Local manifest loaded. Registration and tissue segmentation are separate validations.',
    a2LateralityConflict()?'error':'note');
}
function imageBuffer(plane){
  const volume=state.volume;
  let [w,h]=[512,512];let position=()=>[0,0,0];let cross=[0,0];
  const [i,j,k]=state.cross,sp=volume.spacing;
  if(plane==='axial'){
    w=volume.dims[0];h=volume.dims[1];position=(x,y)=>[x,y,k];cross=[i,j];
  }else if(plane==='coronal'){
    w=volume.dims[0];h=Math.max(2,Math.round(volume.dims[2]*sp[2]/sp[0]));
    position=(x,y)=>[x,j,Math.min(volume.dims[2]-1,Math.round((1-y/(h-1))*(volume.dims[2]-1)))];
    cross=[i,(1-k/(volume.dims[2]-1))*(h-1)];
  }else{
    w=volume.dims[1];h=Math.max(2,Math.round(volume.dims[2]*sp[2]/sp[1]));
    position=(x,y)=>[i,x,Math.min(volume.dims[2]-1,Math.round((1-y/(h-1))*(volume.dims[2]-1)))];
    cross=[j,(1-k/(volume.dims[2]-1))*(h-1)];
  }
  return {w,h,position,cross};
}
function renderSlice(plane){
  if(!state.volume)return;
  const cv=$('mpr-'+plane);const ctx=cv.getContext('2d',{alpha:false});
  const {w,h,position,cross}=imageBuffer(plane);
  cv.width=w;cv.height=h;
  const img=ctx.createImageData(w,h);const px=img.data;
  const [level,width]=huWindows[state.window];const min=level-width/2,max=level+width/2;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const p=position(x,y),value=state.volume.voxel(...p);
    const v=Math.max(0,Math.min(255,Math.round(255*(value-min)/(max-min))));
    const z=4*(x+y*w);px[z]=v;px[z+1]=v;px[z+2]=v;px[z+3]=255;
  }
  ctx.putImageData(img,0,0);
  ctx.lineWidth=Math.max(.7,w/512);ctx.strokeStyle='rgba(222,183,131,.85)';
  ctx.beginPath();ctx.moveTo(cross[0]+.5,0);ctx.lineTo(cross[0]+.5,h);
  ctx.moveTo(0,cross[1]+.5);ctx.lineTo(w,cross[1]+.5);ctx.stroke();
  const [i,j,k]=state.cross;
  $('plane-'+plane).textContent=plane==='axial'?`k ${k} / ${state.volume.dims[2]-1}`:
    plane==='coronal'?`j ${j} / ${state.volume.dims[1]-1}`:`i ${i} / ${state.volume.dims[0]-1}`;
}
function renderSlices(){
  if(!state.volume)return;
  for(const plane of ['axial','coronal','sagittal'])renderSlice(plane);
  $('slice-axial').value=state.cross[2];
  $('slice-readout').textContent='i '+state.cross[0]+' · j '+state.cross[1]+' · k '+state.cross[2];
  const [i,j,k]=state.cross;
  $('ras-readout').textContent='RAS '+state.volume.ras(i,j,k).map(x=>x.toFixed(1)).join(' / ')+' mm';
  updateMarker();applyClipping();
}
function updateMarker(){
  if(!marker)return;
  const match=state.volume&&state.qa&&state.meshSha===state.qa.mesh_sha256&&state.ctSha===state.qa.source_sha256&&window.PulmoNiftiSource.matchQA(state.volume,state.qa).valid;
  marker.visible=Boolean(match&&state.qa.model_from_ras_mm&&state.model&&$('cross-3d').checked);
  if(!marker.visible)return;
  try{
    const ras=state.volume.ras(...state.cross);
    const xyz=window.PulmoNiftiSource.getWorldFromRAS(ras,state.qa);
    marker.position.set(...xyz);
  }catch(err){marker.visible=false;setMessage(err.message,'warn')}
}
function applyClipping(){
  if(!originalMaterials.length)return;
  const match=state.volume&&state.qa&&state.meshSha===state.qa.mesh_sha256&&state.ctSha===state.qa.source_sha256&&window.PulmoNiftiSource.matchQA(state.volume,state.qa).valid;
  const wants=Boolean($('clip-toggle').checked&&match&&state.qa.model_from_ras_mm);
  let plane=null;
  if(wants){
    const direction=$('section-plane').value;
    const axis={sagittal:0,coronal:1,axial:2}[direction];
    if(!Number.isInteger(axis))return;
    const current=state.cross.slice(),neighbour=current.slice();
    neighbour[axis]=Math.min(state.volume.dims[axis]-1,current[axis]+1);
    if(neighbour[axis]===current[axis])neighbour[axis]=Math.max(0,current[axis]-1);
    const a=window.PulmoNiftiSource.getWorldFromRAS(state.volume.ras(...current),state.qa);
    const b=window.PulmoNiftiSource.getWorldFromRAS(state.volume.ras(...neighbour),state.qa);
    const vec=new THREE.Vector3(b[0]-a[0],b[1]-a[1],b[2]-a[2]).normalize();
    plane=new THREE.Plane().setFromNormalAndCoplanarPoint(vec,new THREE.Vector3(...a));
  }
  for(const m of originalMaterials){m.clippingPlanes=plane?[plane]:[];m.needsUpdate=true;}
}
function clickSlice(plane,ev){
  if(!state.volume)return;
  const cv=$('mpr-'+plane),rect=cv.getBoundingClientRect();
  // Canvas uses object-fit:contain; ignore letterbox margins when clicked.
  const factor=Math.min(rect.width/cv.width,rect.height/cv.height);
  const width=cv.width*factor,height=cv.height*factor;
  const left=rect.left+(rect.width-width)/2,top=rect.top+(rect.height-height)/2;
  const x=Math.max(0,Math.min(1,(ev.clientX-left)/width));
  const y=Math.max(0,Math.min(1,(ev.clientY-top)/height));
  if(plane==='axial'){
    state.cross[0]=Math.round(x*(state.volume.dims[0]-1));
    state.cross[1]=Math.round(y*(state.volume.dims[1]-1));
  }else if(plane==='coronal'){
    state.cross[0]=Math.round(x*(state.volume.dims[0]-1));
    state.cross[2]=Math.round((1-y)*(state.volume.dims[2]-1));
  }else{
    state.cross[1]=Math.round(x*(state.volume.dims[1]-1));
    state.cross[2]=Math.round((1-y)*(state.volume.dims[2]-1));
  }
  renderSlices();
}
async function processFile(file){
  try{
    if(!file)return;
    if(/\.nii$/i.test(file.name))await loadNifti(file);
    else if(/\.glb$/i.test(file.name))await loadModel(file);
    else if(/\.json$/i.test(file.name))await loadManifest(file);
    else throw new Error('Only .nii, .glb and .json sources are supported');
    const label=/\.nii$/i.test(file.name)?'ct-file':/\.glb$/i.test(file.name)?'mesh-file':'manifest-file';
    $(label+'-name').textContent=file.name;
  }catch(err){setMessage(String(err.message||err),'error');console.error('PulmoSIMU source import',err);}
}
for(const [id,accept] of [['ct-file','.nii'],['mesh-file','.glb'],['manifest-file','.json']]){
  $(id).accept=accept;
  $(id).addEventListener('change',async e=>{await processFile(e.target.files?.[0]);e.target.value='';});
}
for(const p of ['axial','coronal','sagittal'])$('mpr-'+p).addEventListener('click',ev=>clickSlice(p,ev));
$('slice-axial').addEventListener('input',e=>{if(!state.volume)return;state.cross[2]=Number(e.target.value);renderSlices()});
$('window-preset').addEventListener('change',e=>{state.window=e.target.value;renderSlices()});
$('clip-toggle').addEventListener('change',()=>applyClipping());
$('cross-3d').addEventListener('change',()=>updateMarker());
$('structure-select').addEventListener('change',updateStructureVisibility);
$('reset-view').addEventListener('click',viewReset);
$('source-files').addEventListener('dragover',e=>{e.preventDefault();$('source-files').classList.add('dropping')});
$('source-files').addEventListener('dragleave',()=>$('source-files').classList.remove('dropping'));
$('source-files').addEventListener('drop',async e=>{
  e.preventDefault();$('source-files').classList.remove('dropping');
  for(const file of [...e.dataTransfer.files])await processFile(file);
});
$('offline-help').addEventListener('click',()=>{$('help-dialog').showModal()});
$('help-close').addEventListener('click',()=>$('help-dialog').close());
window.PulmoAnatomyInspector=Object.freeze({get status(){return{
  hasCT:!!state.volume,hasGLB:!!state.model,hasQA:!!state.qa,
  sourceMatched:!!(state.volume&&state.qa&&state.ctSha===state.qa.source_sha256&&state.meshSha===state.qa.mesh_sha256&&window.PulmoNiftiSource.matchQA(state.volume,state.qa).valid),
  meshNames:state.meshNames.slice(),cross:state.cross.slice()}}});
applyQa();start3d();
})();