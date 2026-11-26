/* PulmoSIMU A1 — source-linked anatomic inspection. Local-only uploads.
 * No patient study bytes are uploaded or persisted. No inferred bronchi,
 * disease labels, fissures or geometry-linked physiological claims.
 */
(function(){
'use strict';
const $=id=>document.getElementById(id);
const state={volume:null,qa:null,structureQA:null,qaSha:null,gltf:null,model:null,meshNames:[],ctSha:null,meshSha:null,cross:[256,256,75],surfaceOpacity:1,
  window:'lung',clip:false,clipping:[],fileNames:[],numSlices:0,reviewMarks:[],metric:{a:null,b:null}};
let seg=null,segManifest=null,segBytes=null,segDigest=null,segEpoch=0;
const huWindows={lung:[-600,1500],mediastinum:[40,400],bone:[350,1700]};
const THREE=window.THREE;
const has3d=Boolean(THREE&&THREE.GLTFLoader&&THREE.OrbitControls);
function setMessage(text,kind='note'){
  const node=$('status-text');node.textContent=text;node.dataset.state=kind;
}
function short(n){return Number(n).toFixed(2)}
function escapeHtml(text){return String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
const fileLabels=['glb','nii','json'];
// Known A1 mesh needs anatomical review. Earlier A2 wrongly inverted RAS X; preserve fail-closed safety until independently reviewed.
// This fingerprint also catches the older private manifest which predates A2.
const A2_FLAGGED_GLBSHA='5e4457003b45f9d7f78e87239f935c4299e4c3b947a9a3ffda0e273216228a0f';
function a2LateralityConflict(){
  return state.meshSha===A2_FLAGGED_GLBSHA ||
    state.qa?.mesh_sha256===A2_FLAGGED_GLBSHA ||
    state.qa?.quality_gates?.laterality_labels==='FAIL_A2_SOURCE_RAS_CONFLICT';
}
const a2Warning='A2 GEOMETRY REVIEW REQUIRED: earlier label-swap audit used the WRONG RAS sign. In NIfTI RAS +X is patient RIGHT. Existing right mesh has 14.8% of surface area crossing into the opposite hemisphere; left mesh 1.05%. Labels are NOT proven reversed. Independent CT segmentation review required; bronchoscopy and clinical coupling remain blocked.';
const A2_CANDIDATE_SHA='8f8bea9a95fc21a44ca10244c91c249d2836539bf119973bcb18c614c9886d2f';
function a2CandidateReviewPending(){
 return state.meshSha===A2_CANDIDATE_SHA ||
   state.qa?.mesh_sha256===A2_CANDIDATE_SHA ||
   state.qa?.quality_gates?.laterality_labels==='CANDIDATE_CORRECTED_RAS_INDEPENDENT_REVIEW_PENDING';
}
const a2CandidateWarning='A2 CANDIDATE INVALID: old GLB was relabelled using a sign-inverted RAS convention (+X was incorrectly called LEFT). Do NOT treat its left/right labels as correct, or publish this candidate. Historical voxel overlap 16.3 mL; use the newer A2.1 airspace QA instead. CT-linked bronchoscopy remains BLOCKED.';
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
    ?'A2 CT GEOMETRY · REVIEW REQUIRED'
    :a2CandidateReviewPending()?'A2 CANDIDATE · INVALID RAS LABELS'
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
      ?'File hashes matched · A2 geometry needs expert review · UNVALIDATED'
      :a2CandidateReviewPending()
      ?'Exact files aligned · INVALID old A2 RAS-relabelled candidate · do not use'
      :'Exact study files and header matched · tissue segmentation UNVALIDATED';
  $('registration').dataset.state=a2LateralityConflict()?'error':
    a2CandidateReviewPending()?'pending':aligned?'valid':status?'error':'pending';
  const enabled=aligned;
  $('clip-toggle').disabled=!enabled;
  $('cross-3d').disabled=!enabled;
  $('section-plane').disabled=!enabled;
  if(!enabled){state.clip=false;$('clip-toggle').checked=false;}
  updateMarker();applyClipping();updateResearchGates();renderReviewStatus();renderStudy();
}
let scene=null,renderer=null,camera=null,orbit=null,modelGroup=null,marker=null,
  sourceSection=null,sourceSectionKey=null,workLight=null,originalMaterials=[];
function disposeSourceSection(){
  if(sourceSection){sourceSection.dispose();sourceSection=null;}
  sourceSectionKey=null;
}
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
  disposeSourceSection();
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
  const nextMeshSha=await sha256(bytes);
  const gltf=await new Promise((resolve,reject)=>new THREE.GLTFLoader().parse(bytes,'',resolve,reject));
  if(!gltf.scene)throw Error('GLB scene unavailable');
  let meshCount=0;gltf.scene.traverse(o=>{if(o.isMesh)meshCount++});if(!meshCount)throw Error('GLB has no surface meshes');
  resetReview();clear3d();state.meshSha=nextMeshSha;
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
    'A2 QUALITY REVIEW: prior RAS sign audit was wrong. Original right surface crosses physical midline. No patient-specific use.':
    a2CandidateReviewPending()?
    'Old A2 candidate relabel used reversed RAS X sign and is INVALID. Do not use for CT-linked bronchoscopy.':
    'Local GLB loaded. Geometry inspection is not clinical segmentation validation.',
    a2LateralityConflict()?'error':a2CandidateReviewPending()?'note':'ok');
}
async function loadNifti(file){
  disposeSourceSection();
  if(!/\.nii$/i.test(file.name))throw new Error('Choose an uncompressed NIfTI .nii file');
  if(file.size>800*1024*1024)throw new Error('CT file exceeds local browser size limit');
  setMessage('Reading local 3D NIfTI volume…');
  const bytes=await file.arrayBuffer();
  const nextVolume=window.PulmoNiftiSource.readNifti(bytes),nextCtSha=await sha256(bytes);
  resetReview();resetSegmentation();state.ctSha=nextCtSha;state.volume=nextVolume;
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
  disposeSourceSection();
  if(!/\.json$/i.test(file.name))throw new Error('Choose a reconstruction manifest .json file');
  if(file.size>2*1024*1024)throw Error('Reconstruction manifest exceeds 2 MiB');
  const bytes=await file.arrayBuffer(),data=JSON.parse(new TextDecoder().decode(bytes)),qaSha=await sha256(bytes);
  if(!Array.isArray(data.source_shape_vox)||!Array.isArray(data.sform_RAS_mm)||
     String(data.validation||'').toUpperCase().includes('CLINICAL')){
    throw new Error('This is not a supported research CT geometry manifest.');
  }
  resetReview();state.qa=data;state.qaSha=qaSha;applyQa();renderSlices();
  setMessage(a2LateralityConflict()?a2Warning:
    a2CandidateReviewPending()?a2CandidateWarning:
    'Local manifest loaded. Registration and tissue segmentation are separate validations.',
    a2LateralityConflict()?'error':'note');
}

/* A2.3 source-verified manual points. This never labels mesh triangles,
 * changes CT voxels or grants patient-specific procedure authorization. */
function reviewIdentity(){
  if(!window.PulmoCTReview?.identity||a2CandidateReviewPending()||
     !state.volume||!state.model||!state.qa)return null;
  return window.PulmoCTReview.identity({
    ctSha:state.ctSha,meshSha:state.meshSha,qa:state.qa,
    headerValid:window.PulmoNiftiSource.matchQA(state.volume,state.qa).valid
  });
}
function resetReview(){
  state.reviewMarks=[];
  state.metric={a:null,b:null};
  renderReviewStatus();
}
function renderMetricStatus(){
  const study=reviewIdentity(),p=state.metric;
  for(const id of ['metric-a','metric-b'])$(id).disabled=!study;
  $('metric-clear').disabled=!p.a&&!p.b;
  if(!study){$('metric-result').textContent='Exact matching CT, GLB and manifest required';return;}
  if(p.a&&p.b){
    try{
      const out=window.PulmoCTReview.measureRAS(study,p.a,p.b);
      $('metric-result').textContent=out.distance_mm.toFixed(2)+' mm · A '+p.a.join('/')+' → B '+p.b.join('/');
    }catch(err){$('metric-result').textContent='Invalid source measurement';}
  }else $('metric-result').textContent=p.a
    ?'A '+p.a.join('/')+' · choose B'
    :'Mark two CT voxels for physical distance';
}
function renderReviewStatus(){
  renderMetricStatus();
  const matched=Boolean(reviewIdentity()),count=state.reviewMarks.length;
  $('review-add').disabled=!matched||count>=window.PulmoCTReview.MAX_MARKS;
  $('review-import').disabled=!matched;
  $('review-export').disabled=!matched||!count;
  $('review-clear').disabled=!count;
  $('review-status').textContent=matched
    ?count+' / '+window.PulmoCTReview.MAX_MARKS+' local observations · exact source pair verified'
    :'Exact matching CT, GLB and manifest required · no clinical approval';
  const list=$('review-list');list.replaceChildren();
  if(!count){
    const p=document.createElement('div');p.className='muted';
    p.textContent='No review marks. Select a source-verified CT position, then choose Mark CT point.';
    list.append(p);return;
  }
  for(const mark of state.reviewMarks){
    const row=document.createElement('div');row.className='review-row';
    const description=document.createElement('div');description.className='review-description';
    const strong=document.createElement('strong');strong.textContent=mark.region+' · '+mark.finding.replaceAll('-',' ');
    const txt=document.createElement('span');
    txt.textContent='i/j/k '+mark.voxel.join('/')+' · RAS '+mark.ras_mm.join('/')+' mm'+
      (mark.note?' · '+mark.note:'');
    description.append(strong,txt);
    const focus=document.createElement('button');focus.type='button';focus.textContent='Locate';
    focus.disabled=!matched;
    focus.addEventListener('click',()=>{
      if(!reviewIdentity())return;
      state.cross=mark.voxel.slice();renderSlices();
    });
    const remove=document.createElement('button');remove.type='button';remove.textContent='Remove';
    remove.setAttribute('aria-label','Remove '+mark.region+' observation at voxel '+mark.voxel.join(', '));
    remove.addEventListener('click',()=>{
      state.reviewMarks=state.reviewMarks.filter(m=>m.id!==mark.id);
      renderReviewStatus();renderSlices();
    });
    row.append(description,focus,remove);list.append(row);
  }
}
function drawReviewPoints(plane,ctx,w,h){
  if(!reviewIdentity())return;
  const marks=state.reviewMarks,dimensions=state.volume.dims,coord=state.cross;
  const axis={axial:2,coronal:1,sagittal:0}[plane];
  for(const mark of marks){
    if(mark.voxel[axis]!==coord[axis])continue;
    const v=mark.voxel;
    const x=plane==='sagittal'?v[1]:v[0];
    const y=plane==='axial'?v[1]:
      (1-v[2]/(dimensions[2]-1))*(h-1);
    const px=x+.5,py=y+.5;
    ctx.save();ctx.beginPath();ctx.arc(px,py,Math.max(3,w/125),0,Math.PI*2);
    ctx.fillStyle='rgba(228,192,141,.18)';ctx.fill();
    ctx.lineWidth=Math.max(1.4,w/420);ctx.strokeStyle='#e4c08d';ctx.stroke();
    ctx.font='bold '+Math.max(10,Math.round(w/48))+'px system-ui';
    ctx.fillStyle='#f5dfbe';ctx.fillText(mark.region,px+7,Math.max(11,py-7));
    ctx.restore();
  }
  for(const [key,point] of Object.entries(state.metric)){
    if(!point||point[axis]!==coord[axis])continue;
    const x=plane==='sagittal'?point[1]:point[0];
    const y=plane==='axial'?point[1]:(1-point[2]/(dimensions[2]-1))*(h-1);
    ctx.save();ctx.beginPath();ctx.arc(x+.5,y+.5,Math.max(4,w/110),0,2*Math.PI);
    ctx.lineWidth=1.8;ctx.strokeStyle=key==='a'?'#75bdca':'#e7bd87';ctx.stroke();
    ctx.font='bold '+Math.max(11,Math.round(w/45))+'px system-ui';
    ctx.fillStyle=key==='a'?'#8ed9e7':'#f3d1a5';
    ctx.fillText(key.toUpperCase(),x+9,y-9);ctx.restore();
  }
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
    if(seg&&$('seg-overlay').checked&&seg.at(p)){px[z]=Math.round(v*.58+255*.42);px[z+1]=Math.round(v*.58+178*.42);px[z+2]=Math.round(v*.58+70*.42);}
  }
  ctx.putImageData(img,0,0);
  ctx.lineWidth=Math.max(.7,w/512);ctx.strokeStyle='rgba(222,183,131,.85)';
  ctx.beginPath();ctx.moveTo(cross[0]+.5,0);ctx.lineTo(cross[0]+.5,h);
  ctx.moveTo(0,cross[1]+.5);ctx.lineTo(w,cross[1]+.5);ctx.stroke();
  drawReviewPoints(plane,ctx,w,h);
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
  updateMarker();applyClipping();renderMetricStatus();
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
  // The inspector source section is a local CT pixel plane in the registered
  // model frame, not a mesh-material WebGL clip. The old clipping path could
  // blank both lung meshes in Chromium, although the plane crossed both.
  const match=state.volume&&state.qa&&state.model&&
    state.meshSha===state.qa.mesh_sha256&&state.ctSha===state.qa.source_sha256&&
    window.PulmoNiftiSource.matchQA(state.volume,state.qa).valid;
  const wants=Boolean($('clip-toggle').checked&&match&&state.qa.model_from_ras_mm);
  if(!wants){
    if(sourceSection)sourceSection.setVisible(false);
    return;
  }
  if(!window.PulmoXRCTSpatial?.create||!modelGroup){
    $('clip-toggle').checked=false;
    setMessage('CT source plane unavailable; original surfaces preserved.','error');
    return;
  }
  try{
    const study={volume:state.volume,manifest:state.qa,ctSha:state.ctSha,
      meshSha:state.meshSha,meshVerified:true};
    const gate=window.PulmoXRCTSpatial.gate(study);
    if(!gate.ok)throw Error(gate.reason);
    const key=state.ctSha+':'+state.meshSha;
    if(sourceSection&&sourceSectionKey!==key)disposeSourceSection();
    if(!sourceSection){
      sourceSection=window.PulmoXRCTSpatial.create(THREE,modelGroup,study);
      sourceSectionKey=key;
    }
    const direction=$('section-plane').value;
    const axis={sagittal:0,coronal:1,axial:2}[direction];
    if(!Number.isInteger(axis))throw Error('Invalid CT section orientation');
    sourceSection.setIndex(direction,state.cross[axis]);
    for(const mesh of sourceSection.group.children){
      mesh.visible=mesh.userData.plane===direction;
      // Translucent actual CT intensities, legible in front of the surface.
      // Original lung materials, geometry, and camera are never modified.
      mesh.material.depthTest=false;
      mesh.material.opacity=.70;
    }
    sourceSection.setVisible(true);
  }catch(error){
    if(sourceSection)sourceSection.setVisible(false);
    $('clip-toggle').checked=false;
    setMessage('3D CT section unavailable: '+String(error.message||error),'error');
  }
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
$('section-plane').addEventListener('change',()=>applyClipping());
$('surface-opacity').addEventListener('input',()=>applySurfaceOpacity());
$('cross-3d').addEventListener('change',()=>updateMarker());
$('structure-select').addEventListener('change',updateStructureVisibility);
$('structure-qa-file').addEventListener('change',async ev=>{
  const file=ev.target.files?.[0];
  ev.target.value='';
  if(!file)return;
  try{
    const data=JSON.parse(await file.text());
    if(!data?.groups||!data?.CT_SHA256||!data?.structure_findings)
      throw Error('Expected A3.2–A3.4 source-structure QA JSON');
    if(state.qa&&data.CT_SHA256!==state.qa.source_sha256)
      throw Error('Structure-QA CT SHA-256 does not match this source manifest');
    if(data.clinical_segmentation_approval!=='PENDING_INDEPENDENT_REVIEW')
      throw Error('Unknown research structure-QA review status');
    state.structureQA=data;updateResearchGates();
    setMessage('Source masks QA loaded locally; CT-specific bronchoscopy remains gated.','note');
  }catch(err){state.structureQA=null;updateResearchGates();
    setMessage('Source structure QA rejected: '+String(err.message||err),'error');
  }
});

for(const [id,key] of [['metric-a','a'],['metric-b','b']]){
 $(id).addEventListener('click',()=>{
   if(!reviewIdentity())return;
   state.metric[key]=state.cross.slice();
   renderMetricStatus();renderSlices();
   setMessage('CT source distance point '+key.toUpperCase()+' set in exact voxel coordinates.','note');
 });
}
$('metric-clear').addEventListener('click',()=>{
 state.metric={a:null,b:null};renderMetricStatus();renderSlices();
});
$('review-add').addEventListener('click',()=>{
 try{
   const study=reviewIdentity();
   if(!study)throw Error('CT, GLB and manifest must exactly match before marking');
   const mark=window.PulmoCTReview.newMark(study,state.cross.slice(),
     $('review-region').value,$('review-finding').value,$('review-note').value);
   if(state.reviewMarks.some(x=>x.id===mark.id))throw Error('Duplicate review marker ID');
   state.reviewMarks.push(mark);$('review-note').value='';
   renderReviewStatus();renderSlices();
   setMessage('Source-bound review point added. Export JSON to preserve the session.','ok');
 }catch(err){setMessage(String(err.message||err),'error');}
});
$('review-clear').addEventListener('click',()=>{
  resetReview();renderSlices();setMessage('Session review marks cleared. Source data is unchanged.','note');
});
$('review-export').addEventListener('click',()=>{
 try{
   const study=reviewIdentity();
   const bundle=window.PulmoCTReview.exportReview(study,state.reviewMarks);
   const json=new Blob([JSON.stringify(bundle,null,2)+'\n'],{type:'application/json'});
   const href=URL.createObjectURL(json),a=document.createElement('a');
   a.href=href;a.download='PulmoSIMU_CT_Review_'+study.source_sha256.slice(0,12)+'.json';
   document.body.append(a);a.click();a.remove();
   setTimeout(()=>URL.revokeObjectURL(href),500);
   setMessage('Review JSON exported locally. No study CT or GLB data included.','ok');
 }catch(err){setMessage('Review export failed: '+String(err.message||err),'error');}
});
$('review-import').addEventListener('change',async ev=>{
 const file=ev.target.files?.[0];ev.target.value='';
 if(!file)return;
 try{
   if(file.size>512*1024)throw Error('Review JSON exceeds 512 KiB');
   const study=reviewIdentity();
   if(!study)throw Error('Exact matching source is required before import');
   const marks=window.PulmoCTReview.importReview(JSON.parse(await file.text()),study);
   state.reviewMarks=marks;renderReviewStatus();renderSlices();
   setMessage('Manual CT review imported for the matching exact source pair.','ok');
 }catch(err){setMessage('Review import rejected: '+String(err.message||err),'error');}
});

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
  sourceSectionVisible:Boolean(sourceSection?.snapshot().visible),
  sourceSectionMode:sourceSection?.snapshot().visible?'source-voxel-plane':'off',
  sectionPlane:$('section-plane').value,
  sourceMeshesVisible:state.model?(()=>{
    let n=0;state.model.traverse(o=>{if(o.isMesh&&o.visible)n++});return n;
  })():0,
  sourceMaterialClippingActive:originalMaterials.some(m=>m?.clippingPlanes?.length>0),
  sourceMatched:!!(state.volume&&state.qa&&state.ctSha===state.qa.source_sha256&&state.meshSha===state.qa.mesh_sha256&&window.PulmoNiftiSource.matchQA(state.volume,state.qa).valid),
  meshNames:state.meshNames.slice(),cross:state.cross.slice(),
   reviewCount:state.reviewMarks.length,reviewReady:Boolean(reviewIdentity()),
   metricReady:!!(reviewIdentity()&&state.metric.a&&state.metric.b)}}});

function currentStudy(){
 return window.PulmoStudy.build({volume:state.volume,ctSha:state.ctSha,meshSha:state.meshSha,qa:state.qa,qaSha:state.qaSha,meshLoaded:!!state.model,headerMatched:!!(state.volume&&state.qa&&window.PulmoNiftiSource.matchQA(state.volume,state.qa).valid)});
}
function renderStudy(){
 const summary=$('study-summary');if(!summary)return;
 $('study-result').textContent='';
 try{
  const s=currentStudy();$('study-status').textContent=s.surface?'CT + reconstruction matched':'CT frame verified · surface not linked';
  summary.textContent='SHA-256  '+s.source.sha256+'\n'+s.source.dimensions.join(' × ')+' voxels · '+s.spatial.spacing_mm.map(x=>x.toFixed(3)).join(' / ')+' mm\nVoxel ↔ RAS ↔ LPS · roundtrip '+s.spatial.roundtrip_max_vox.toExponential(2)+' voxels\n'+(s.surface?'Exact GLB and reconstruction manifest linked (unreviewed).':'No matched surface attached.')+'\nPatient-specific procedure blocked.';
  $('study-export').disabled=false;$('study-import').disabled=false;
 }catch(e){$('study-status').textContent='Study not ready';summary.textContent=e.message;$('study-export').disabled=true;$('study-import').disabled=true;}
}
$('study-export').addEventListener('click',()=>{
 try{const s=currentStudy(),url=URL.createObjectURL(new Blob([JSON.stringify(s,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='PulmoSIMU_Study_'+s.source.sha256.slice(0,12)+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),500);$('study-result').textContent='Manifest exported locally; no image bytes.';}catch(e){$('study-result').textContent=e.message;}
});
$('study-import').addEventListener('change',async ev=>{
 const f=ev.target.files?.[0];ev.target.value='';if(!f)return;
 try{if(f.size>64*1024)throw Error('Study manifest exceeds 64 KiB');const imported=JSON.parse(await f.text());window.PulmoStudy.verify(imported,currentStudy());$('study-result').textContent='Exact local package match. Clinical review remains pending.';}catch(e){$('study-result').textContent='Rejected: '+e.message;}
});


function updateSegUI(message){
 $('seg-status').textContent=message;
 for(const id of ['seg-add','seg-remove','seg-clear','seg-export'])$(id).disabled=!seg;
 const m=seg?.metrics();
 $('seg-metrics').textContent=m?seg.structure+' · '+m.edited_voxels.toLocaleString()+' voxels · '+m.volume_ml.toFixed(3)+' mL · +'+m.added+' / −'+m.removed+' proposed edits · agreement to original (Dice): '+(m.dice_to_original===null?'undefined':m.dice_to_original.toFixed(4))+' · not accuracy against ground truth.':'Load CT, then matching mask manifest and raw mask.';
}
function resetSegmentation(){segEpoch++;seg=null;segManifest=null;segBytes=null;segDigest=null;updateSegUI('No mask loaded');}
function trySeg(){
 seg=null;
 try{if(!segManifest||!segBytes){updateSegUI('Awaiting manifest and binary mask');return;}
  seg=window.PulmoSegmentation.bind(currentStudy(),segManifest,segBytes,segDigest);
  updateSegUI('Exact source-linked mask · unreviewed');
 }catch(e){updateSegUI('Rejected: '+e.message);}finally{if(state.volume)renderSlices();}
}
$('seg-manifest').addEventListener('change',async ev=>{
 const file=ev.target.files?.[0];ev.target.value='';if(!file)return;
 const epoch=++segEpoch;seg=null;segManifest=null;updateSegUI('Reading mask manifest');renderSlices();
 try{if(file.size>64*1024)throw Error('Manifest exceeds 64 KiB');const m=JSON.parse(await file.text());if(epoch!==segEpoch)return;segManifest=m;trySeg();}catch(e){if(epoch===segEpoch)updateSegUI('Rejected: '+e.message);}
});
$('seg-mask').addEventListener('change',async ev=>{
 const file=ev.target.files?.[0];ev.target.value='';if(!file)return;
 const epoch=++segEpoch;seg=null;segBytes=null;segDigest=null;updateSegUI('Reading mask');renderSlices();
 try{if(!state.volume)throw Error('Load CT first');if(file.size>250000000)throw Error('Mask exceeds local size limit');const bytes=await file.arrayBuffer(),digest=await sha256(bytes);if(epoch!==segEpoch)return;segBytes=new Uint8Array(bytes);segDigest=digest;trySeg();}catch(e){if(epoch===segEpoch)updateSegUI('Rejected: '+e.message);}
});
for(const [id,value] of [['seg-add',1],['seg-remove',0]])$(id).addEventListener('click',()=>{if(!seg)return;try{seg.set(state.cross.slice(),value);updateSegUI('Local correction proposal · unreviewed');renderSlices();}catch(e){updateSegUI(e.message);}});
$('seg-overlay').addEventListener('change',renderSlices);
$('seg-clear').addEventListener('click',()=>{if(seg){seg.clear();updateSegUI('Original mask restored');renderSlices();}});
$('seg-export').addEventListener('click',()=>{if(!seg)return;const url=URL.createObjectURL(new Blob([JSON.stringify(seg.patch(),null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='PulmoSIMU_MaskPatch_'+seg.digest.slice(0,12)+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),500);});
applyQa();renderReviewStatus();start3d();
})();