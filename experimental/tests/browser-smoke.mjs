/**
 * Browser-level integration gate, not a clinical or visual fidelity certificate.
 * Run after installing playwright + Chromium and starting a local HTTP server.
 */
import assert from 'node:assert/strict';
import {chromium} from 'playwright';

const host=process.env.PULMOSIMU_TEST_ORIGIN||'http://127.0.0.1:8765';
const browser=await chromium.launch({headless:true,args:['--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader']});
const errors=[];
const page=await browser.newPage({viewport:{width:1440,height:900},deviceScaleFactor:1});
page.on('pageerror',error=>errors.push(error.message));
page.on('console',message=>{if(message.type()==='error'&&!message.text().includes('WebGL'))errors.push(message.text())});
try {
 await page.goto(host+'/experimental/lab.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>document.getElementById('networkStatus')?.textContent?.includes('SI-unit'),null,{timeout:45000});
 assert.ok(await page.locator('#referenceObservatory').isVisible(),'Numerical observatory visible by default');
 assert.ok(await page.locator('#lungCanvas').isHidden(),'No fake anatomical mesh presented in numerical reference');
 assert.equal(await page.locator('#regionalBars .region-row').count(),5,'Five mathematical region outputs');
 assert.equal(await page.locator('#regionalProfileTitle').innerText(),'Ventilation allocation');
 assert.notEqual(await page.locator('#tidal').innerText(),'—','R3 breathing model mounted');
 assert.notEqual(await page.locator('#terminals').innerText(),'—','Airway graph loaded');
 assert.equal(await page.locator('#atlasSegmentCount').innerText(),'3','RUL segment allocation present');
 assert.match(await page.locator('#atlasFissures').innerText(),/oblique fissure/);
 assert.match(await page.locator('#atlasVessels').innerText(),/pulmonary artery/);
 await page.locator('#focusRegion').selectOption('LLL');
 await page.locator('#scenario').selectOption('asthma');
 // Focus-region selection now tracks the canonical reference lobe and its
 // anatomical inspector; left lower lobe has four commonly defined units.
 assert.equal(await page.locator('#atlasSegmentCount').innerText(),'4');

 await page.locator('#scenario').selectOption('asthma');
 await page.locator('#therapy').selectOption('anti_il5');
 await page.locator('#phenotype').selectOption('eosinophilic');
 await page.locator('#days').evaluate(el=>{el.value='80';el.dispatchEvent(new Event('input',{bubbles:true}))});
 await page.waitForTimeout(350);
 assert.match(await page.locator('#pathwayNote').innerText(),/eosinophil/i,'R5 pathway readout');
 assert.notEqual(await page.locator('#deltaInfl').innerText(),'—','R5 counterfactual comparison');
 await page.locator('#scenario').selectOption('lung_cancer');
 await page.locator('#focusRegion').selectOption('LLL');
 await page.waitForTimeout(350);
 await page.locator('#overlayTumor').click();
 assert.equal(await page.locator('#overlayLabel').innerText(),'TUMOR');
 assert.ok(await page.locator('#selected').isVisible());
 await page.locator('#isolateRegion').click();
 assert.match(await page.locator('#isolateRegion').innerText(),/Show all regions/);
 assert.equal(await page.locator('#regionalBars .region-row').count(),1,'Numerical region isolation');
 await page.locator('#isolateRegion').click();
 assert.equal(await page.locator('#regionalBars .region-row').count(),5);
 assert.ok(await page.locator('#resetCamera').isHidden(),'Source camera controls hidden without a source');
 // Generate a synthetic GLB and PNEUMO·VR metadata pair entirely within Chromium.
 // This is a browser integration fixture, NOT patient imaging data.
 await page.evaluate(async()=>{
   const THREE=await import('three');
   const {GLTFExporter}=await import('three/addons/exporters/GLTFExporter.js');
   const scene=new THREE.Scene();
   scene.add(new THREE.Mesh(new THREE.BoxGeometry(.6,.6,.6),new THREE.MeshStandardMaterial({color:0xb0b0b0})));
   const binary=await new Promise((resolve,reject)=>new GLTFExporter().parse(scene,resolve,reject,{binary:true}));
   const meta={version:'pneumovr-1.1',transform:{centre_mm:[0,0,0],scale:2},
     segments:[
       {id:10,parent:-1,name:'Trachea',gen:0,pts:[[0,.2,0],[0,0,0]],r0:.08,r1:.07},
       {id:21,parent:10,name:'Branch',gen:1,pts:[[0,0,0],[.2,-.2,.1]],r0:.06,r1:.04}
     ]};
   function assign(id,file){const el=document.getElementById(id),dt=new DataTransfer();dt.items.add(file);el.files=dt.files}
   assign('ctMetaFile',new File([JSON.stringify(meta)],'synthetic_meta.json',{type:'application/json'}));
   assign('ctMeshFile',new File([binary],'synthetic_airway.glb',{type:'model/gltf-binary'}));
 });
 await page.locator('#loadCt').click();
 await page.waitForFunction(()=>document.getElementById('ctStatus')?.textContent?.includes('loaded locally'),null,{timeout:30000});
 assert.match(await page.locator('#anatomyProvenance').innerText(),/CT AIRWAY/);
 assert.ok(await page.locator('#lungCanvas').isVisible(),'Source-backed 3D canvas only after GLB load');
 assert.ok(await page.locator('#referenceObservatory').isHidden(),'Numerical board replaced by supplied source');
 assert.ok(await page.locator('#anatomicalAtlas').isHidden(),'reference atlas hidden during CT-source inspection');
 assert.match(await page.locator('#networkStatus').innerText(),/NOT computed from/);
 assert.equal(await page.locator('#ctBranchCount').innerText(),'2','CT-derived branch inspection');
 assert.match(await page.locator('#ctUnitScale').innerText(),/500\.0 mm\/unit/);
 // A deliberately mismatched CT metadata frame must be rejected while
 // preserving the last successful local scene.
 await page.evaluate(async()=>{
   const el=document.getElementById('ctMetaFile');
   const previous=JSON.parse(await el.files[0].text());
   for(const segment of previous.segments)segment.pts=segment.pts.map(p=>p.map(v=>v+50));
   const dt=new DataTransfer();dt.items.add(new File([JSON.stringify(previous)],'mismatch.json',{type:'application/json'}));
   el.files=dt.files;
 });
 await page.locator('#loadCt').click();
 await page.waitForFunction(()=>document.getElementById('ctStatus')?.textContent?.includes('rejected'),null,{timeout:15000});
 assert.match(await page.locator('#anatomyProvenance').innerText(),/CT AIRWAY/,'Rejected replacement leaves existing source active');
 await page.locator('#restoreReference').click();
 assert.match(await page.locator('#anatomyProvenance').innerText(),/NUMERICAL REFERENCE/);
 assert.ok(await page.locator('#anatomicalAtlas').isVisible(),'reference terminology restored');
 assert.ok(await page.locator('#lungCanvas').isHidden(),'No artificial anatomy after returning to numerical mode');
 assert.match(await page.locator('#ctStatus').innerText(),/cleared/);
 // Surface-atlas fixture: authored geometric primitives for integration only.
 await page.evaluate(async()=>{
   const THREE=await import('three');
   const {GLTFExporter}=await import('three/addons/exporters/GLTFExporter.js');
   const scene=new THREE.Scene();
   const a=new THREE.Mesh(new THREE.BoxGeometry(120,160,75),new THREE.MeshStandardMaterial());
   a.name='fixture_RUL';a.position.x=65;scene.add(a);
   const b=new THREE.Mesh(new THREE.CylinderGeometry(10,12,130,12),new THREE.MeshStandardMaterial());
   b.name='fixture_RPA';b.position.x=-50;scene.add(b);
   const binary=await new Promise((resolve,reject)=>new GLTFExporter().parse(scene,resolve,reject,{binary:true}));
   const manifest={
     schema:'pulmosimu-surface-atlas-v1',datasetId:'browser-test-only',
     source:'Synthetic automated fixture',license:'Test-generated',
     rightsStatement:'Synthetic data solely for automated checks',
     coordinateFrame:'fixture-local',unit:'mm',patientSpecific:false,
     structures:[
       {id:'RUL',kind:'lobe',objectName:'fixture_RUL',geometryOrigin:'authored',
        evidence:'Generated primitive, not anatomy',sourceValidated:false},
       {id:'RPA',kind:'vessel',objectName:'fixture_RPA',geometryOrigin:'authored',
        evidence:'Generated primitive, not vasculature',sourceValidated:false}
     ]
   };
   const assign=(id,file)=>{const dt=new DataTransfer();dt.items.add(file);document.getElementById(id).files=dt.files;};
   assign('surfaceManifest',new File([JSON.stringify(manifest)],'surface_manifest.json',{type:'application/json'}));
   assign('surfaceMesh',new File([binary],'synthetic_surfaces.glb',{type:'model/gltf-binary'}));
 });
 await page.locator('#loadSurfaceAtlas').click();
 await page.waitForFunction(()=>document.getElementById('surfaceStatus')?.textContent?.includes('loaded locally'),null,{timeout:30000});
 assert.match(await page.locator('#anatomyProvenance').innerText(),/EXTERNAL SURFACE ATLAS/);
 assert.ok(await page.locator('#lungCanvas').isVisible(),'Atlas source uses imported 3D only');
 assert.equal(await page.locator('#surfaceLobeCount').innerText(),'1/5');
 assert.equal(await page.locator('#surfaceVesselCount').innerText(),'1');
 assert.equal(await page.locator('#surfaceUnit').innerText(),'mm');
 assert.ok(await page.locator('#anatomicalAtlas').isHidden());
 assert.match(await page.locator('#networkStatus').innerText(),/NOT computed from displayed geometry/);
 await page.locator('#surfaceVessels').click();
 assert.match(await page.locator('#surfaceVessels').innerText(),/Show vessels/);
 await page.locator('#surfaceVessels').click();
 await page.locator('#surfaceLobes').click();
 assert.match(await page.locator('#surfaceLobes').innerText(),/Show lobar surfaces/);
 await page.locator('#surfaceLobes').click();
 // Unsupported patient-specific manifests must be refused without replacing the loaded scene.
 await page.evaluate(async()=>{
   const el=document.getElementById('surfaceManifest');
   const bad=JSON.parse(await el.files[0].text());bad.patientSpecific=true;
   const dt=new DataTransfer();dt.items.add(new File([JSON.stringify(bad)],'invalid_manifest.json',{type:'application/json'}));el.files=dt.files;
 });
 await page.locator('#loadSurfaceAtlas').click();
 await page.waitForFunction(()=>document.getElementById('surfaceStatus')?.textContent?.includes('rejected'),null,{timeout:15000});
 assert.match(await page.locator('#anatomyProvenance').innerText(),/EXTERNAL SURFACE ATLAS/);
 await page.locator('#restoreReference').click();
 assert.match(await page.locator('#anatomyProvenance').innerText(),/NUMERICAL REFERENCE/);
 assert.ok(await page.locator('#anatomicalAtlas').isVisible());
 await page.screenshot({path:'/tmp/pulmosimu-experimental-smoke.png',fullPage:true});
 assert.deepEqual(errors,[],'No uncaught browser errors');
 console.log('PASS numerical laboratory: no invented lobe meshes; R3, R5 and source-backed inspections retained');
 // An ordinary old VR bookmark must never show the schematic ellipsoids.
 const redirect=await browser.newPage();
 await redirect.goto(host+'/experimental/vr.html',{waitUntil:'domcontentloaded'});
 await redirect.waitForURL(/index\.html\?xr=procedural.*XR09-UNIFIED/,{timeout:30000});
 assert.ok(!redirect.url().includes('archive=1'),'Legacy entry points to original engine');
 await redirect.close();
 const xr=await browser.newPage({viewport:{width:1260,height:850}});
 const xrErrors=[];
 xr.on('pageerror',err=>xrErrors.push(err.message));
 try{
  await xr.goto(host+'/experimental/vr.html?archive=1',{waitUntil:'domcontentloaded'});
  await xr.waitForFunction(()=>document.getElementById('status')?.textContent?.includes('SOURCE BRANCHES'),null,{timeout:45000});
  assert.ok(await xr.locator('#stageCanvas').isVisible());
  assert.equal(await xr.locator('#selectedId').innerText(),'RUL');
  const count=Number(await xr.locator('#branchCount').innerText());
  assert.ok(count>0,'Canonical reference airway tree loaded');
  assert.ok(await xr.locator('#enterVR button').count()>0,'WebXR button or supported fallback present');
  await xr.locator('#modeAirway').click();
  assert.equal(await xr.locator('#modeAirway').getAttribute('aria-pressed'),'true');
  assert.ok(await xr.locator('#routeControls').isVisible(),'Airway guide reveals route controls');
  await xr.locator('#routeProgress').fill('65');
  assert.equal(await xr.locator('#routePercent').innerText(),'65%');
  assert.equal(await xr.evaluate(()=>window.pulmoXRDebug()?.mode),'path');
  await xr.locator('#isolateLobe').click();
  assert.equal(await xr.locator('#isolateLobe').getAttribute('aria-pressed'),'true');
  assert.equal(await xr.evaluate(()=>window.pulmoXRDebug()?.isolated),true);
  const originalBranch=await xr.evaluate(()=>window.pulmoXRDebug()?.workbench?.branchId);
  await xr.locator('#nextBranch').click();
  const branchAfter=await xr.evaluate(()=>window.pulmoXRDebug()?.workbench?.branchId);
  assert.ok(Number.isInteger(originalBranch)&&Number.isInteger(branchAfter)&&originalBranch!==branchAfter,
   'Next airway branch must navigate the canonical reference source');
  assert.notEqual(await xr.locator('#branchName').innerText(),'—');
  await xr.locator('#modeSection').click();
  assert.equal(await xr.locator('#modeSection').getAttribute('aria-pressed'),'true');
  assert.ok(await xr.locator('#sectionControls').isVisible(),'Section mode provides a separate reference position');
  await xr.locator('#sectionPosition').fill('40');
  assert.equal(await xr.locator('#sectionPercent').innerText(),'+40%');
  assert.equal(await xr.evaluate(()=>window.pulmoXRDebug()?.workbench?.sectionFraction),.4);
  assert.equal(await xr.evaluate(()=>window.pulmoXRDebug()?.workbench?.sectionActive),true);
  assert.deepEqual(await xr.evaluate(()=>window.pulmoXRDebug()?.sectionRenderer),{
    clippingEnabled:true,clippingActive:true,lobeClipping:true,airwayClipping:true
  },'Real clipping planes must be attached to rendered reference meshes');
  await xr.locator('#modeAnatomy').click();
  assert.ok(await xr.locator('#routeControls').isHidden(),'Exploration does not display route controls');
  assert.equal(await xr.evaluate(()=>window.pulmoXRDebug()?.mode),'anatomy');
  assert.equal(await xr.evaluate(()=>window.pulmoXRDebug()?.workbench?.sectionActive),false);
  assert.equal(await xr.evaluate(()=>window.pulmoXRDebug()?.sectionRenderer?.clippingActive),false);
  await xr.locator('#panelToggle').click();
  assert.equal(await xr.locator('#panelToggle').getAttribute('aria-pressed'),'false');
  await xr.locator('#resetView').click();
  assert.equal(await xr.locator('#selectedId').innerText(),'RUL');
  assert.equal(await xr.locator('#isolateLobe').getAttribute('aria-pressed'),'false');
  assert.equal(await xr.evaluate(()=>window.pulmoXRDebug()?.mode),'anatomy');
  assert.deepEqual(xrErrors,[],'No WebXR viewer script errors');
  console.log('PASS Quest viewer: reference anatomy, airway guide, progress, isolation, spatial controls and desktop fallback');
 }finally{await xr.close()}
 // XR-07 verifies the ORIGINAL procedural application mounts the WebXR adapter
 // without changing its original bronchoscopic, pathology or desktop state.
 const proceduralXR=await browser.newPage({viewport:{width:1360,height:850}});
 const proceduralErrors=[];
 proceduralXR.on('pageerror',err=>proceduralErrors.push(err.message));
 try{
  await proceduralXR.goto(host+'/index.html?xr=procedural',{waitUntil:'domcontentloaded'});
  await proceduralXR.waitForFunction(()=>typeof window.pulmoProceduralXRDebug==='function',null,{timeout:45000});
  const status=await proceduralXR.evaluate(()=>window.pulmoProceduralXRDebug());
  assert.equal(status.source,'original-pneumovr-scene');
  await proceduralXR.waitForFunction(()=>window.PulmoXRReferencePhysicsStatus==='reference-ready',
    null,{timeout:30000});
  const solver=await proceduralXR.evaluate(()=>{
    const engine=window.PulmoXRReferencePhysics;
    if(!engine)return null;
    const sample=engine.step(.08);
    return {sample,after:engine.read()};
  });
  assert.equal(solver?.sample?.validation?.solver,'R3 linear RC network');
  assert.equal(solver?.sample?.validation?.clinicalValidated,false);
  assert.equal(solver?.sample?.validation?.meshMotionCoupled,false);
  assert.equal(solver?.after?.airwayCount,18);
  assert.ok(Number.isFinite(solver?.after?.pressurePa),
    'Original scene can run the same SI-unit R3 reference physics without a second solver');

  assert.equal(status.mode,'anatomy');
  assert.ok(status.originalAnatomyMeshes>=15,'Original lobed lungs and bronchial anatomy, not the five schematic fallback lobes');
  assert.ok(status.originalSegments>=15,'Original procedural bronchial tree is connected');
  assert.ok(status.originalPathologyCount>=3,'Original pathology objects are retained');
  assert.equal(status.sourceAudit,null,'Reference-only scene cannot claim CT co-registration');
  assert.equal(status.sourceScaleEligible,false,'Reference scene cannot claim physical scene scale');
  assert.ok(status.clearance?.wallContactSimulated===false,'Lumen estimate is not tissue mechanics');
  const coframe=await proceduralXR.evaluate(()=>{
    const metadata={version:'pneumovr-1.1',transform:{centre_mm:[0,0,0],scale:4},
     segments:[{id:5,parent:-1,r0:.06,r1:.05,pts:[[0,.3,0],[0,0,0]]}]};
    const bounds={min:[-.4,-.2,-.3],max:[.4,.7,.3]};
    return window.PulmoXR08.assessRegistration(metadata,bounds);
  });
  assert.equal(coframe.status,'coframed','Browser classic-script CT registration contract available');
  assert.equal(coframe.sceneUnitMetres,.25);
  assert.equal(coframe.modelToPatientRegistrationValidated,false);
  const researchBridge=await proceduralXR.evaluate(()=>{
    const bridge=window.PulmoCTResearchBridge;
    if(!bridge)return null;
    const sha='5b234ac92a9a5d58d15f3f43144d31612d10ff6e60777d8c273cd52b36b13087';
    const sample={geometry_class:'PRELIMINARY_CT_DERIVED_AIRSPACE_MASK',
      source_sha256:sha,mesh_sha256:sha,model_from_ras_mm:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
    return {
      validResearch:bridge.isResearchManifest(sample),
      validA2:bridge.isResearchManifest({...sample,
        geometry_class:'A2_1_CONSERVATIVE_BILATERAL_CT_AIRSPACE_REVIEW'}),
      rejectsAirway:bridge.isResearchManifest({...sample,segments:[{id:0}]}),
      sourceControls:!!document.getElementById('ct-source-desk'),
      sourceInitiallyHidden:!document.body.classList.contains('ct-source-mode')
    };
  });
  assert.equal(researchBridge?.validResearch,true);
  assert.equal(researchBridge?.validA2,true);
  assert.equal(researchBridge?.rejectsAirway,false);
  assert.equal(researchBridge?.sourceControls,true);
  assert.equal(researchBridge?.sourceInitiallyHidden,true);
  console.log('PASS CT research scene is separate from patient airway metadata and reference procedure');
  const atlasCheck=await proceduralXR.evaluate(()=>{
    const T=window.THREE;
    const stage=new T.Group(),source=new T.Group();
    stage.add(source);
    const geom=new T.BufferGeometry();
    geom.setAttribute('position',new T.Float32BufferAttribute([
      90,60,-310, 90,120,-360, 90,80,-470,
      -90,60,-330, -90,60,-460, -90,70,-400
    ],3));
    geom.setIndex([0,1,2,3,4,5]);
    geom.computeVertexNormals();
    const material=new T.MeshStandardMaterial({color:0xd2c4b1});
    const mesh=new T.Mesh(geom,material);source.add(mesh);
    const atlas=window.PulmoCTLobeTerritories.create(T,source,stage,{
      model_from_ras_mm:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    });
    const initial=atlas.enable();
    const newMat=mesh.material;
    const newGeom=mesh.geometry;
    const update=atlas.update({rightMajor:18,rightMinor:12,selected:'RUL'});
    atlas.disable();
    const result={fiveLabels:Object.keys(window.PulmoCTLobeTerritories.names).length,initial:initial.mode,
      next:update.mode,vertices:update.vertices,derivedNewMesh:newGeom!==geom,
      derivedNewMaterial:newMat!==material,originalGeometry:mesh.geometry===geom,
      originalMaterial:mesh.material===material};
    atlas.dispose();geom.dispose();material.dispose();
    return result;
  });
  assert.equal(atlasCheck.fiveLabels,5);
  assert.equal(atlasCheck.initial,'editable-atlas-territories');
  assert.equal(atlasCheck.next,'editable-atlas-territories');
  assert.equal(atlasCheck.vertices,6);
  assert.equal(atlasCheck.derivedNewMesh,true);
  assert.equal(atlasCheck.derivedNewMaterial,true);
  assert.equal(atlasCheck.originalGeometry,true);
  assert.equal(atlasCheck.originalMaterial,true);
  console.log('PASS A2.2 atlas renders five source-space territory labels without altering source topology');
  const originalCount=status.originalSegments;
  await proceduralXR.evaluate(()=>{
    window._applyPatientMeta({version:'pneumovr-1.1',
      transform:{centre_mm:[0,0,0],scale:4},
      segments:[{id:5,parent:-1,r0:.06,r1:.05,pts:[[0,.3,0],[0,0,0]]}]});
  });
  const afterReject=await proceduralXR.evaluate(()=>window.pulmoProceduralXRDebug());
  assert.equal(afterReject.originalSegments,originalCount,
   'Without a valid GLB, patient metadata must be rejected before touching original procedure graph');
  assert.equal(afterReject.sourceAudit,null);
  assert.equal(await proceduralXR.locator('#pulmo-procedural-xr').count(),1,'Dedicated immersive original-engine entry is mounted');
  const xr16=await proceduralXR.evaluate(()=>window.pulmoProceduralXRDebug());
  assert.equal(xr16.xrCapabilityEvidence.desktop,'available');
  assert.equal(xr16.xrCapabilityEvidence.worldAnchor,'NOT_IMPLEMENTED');
  assert.equal(xr16.xrCapabilityEvidence.physicalOcclusion,'NOT_IMPLEMENTED');
  assert.equal(xr16.xrCapabilityEvidence.patientSpecificBronchoscopy,'NOT_VALIDATED');
  assert.deepEqual(xr16.monitorResolution,[800,450],'XR16 starts with balanced same-renderer source video');
  assert.equal(xr16.procedureStation.layout,'stacked');
  assert.equal(xr16.procedureStation.fovDegrees,100);
  assert.equal(xr16.procedureStation.insertionMultiplier,1);
  assert.equal(xr16.instrumentInputs.tipH,0,'No invented instrument movement at rest');
  console.log('PASS XR16: original Quest scene loads independent larger monitor and bounded procedure station');
  assert.equal(await proceduralXR.locator('#pulmo-passthrough-ar').count(),1,
    'Actual, separate AR passthrough entry is mounted');
  const arCapability=await proceduralXR.evaluate(()=>({
    available:!!navigator.xr,visible:document.querySelector('#pulmo-passthrough-ar').style.display,
    alpha:window.PulmoXRProcedureGuide?.describe?true:false
  }));
  assert.equal(arCapability.alpha,true,'Source-grounded procedural guidance available');
  if(!arCapability.available)assert.equal(arCapability.visible,'none',
    'Desktop browser must never offer fake AR passthrough without immersive-ar');
  const guided=await proceduralXR.evaluate(()=>window.PulmoXRProcedureGuide.describe({
    segmentId:0,segment:{name:'Trachea',gen:0,children:[1,4]},
    segments:{1:{name:'Left main bronchus'},4:{name:'Right main bronchus'}},
    position:.99,atBranch:true,clinicalMode:'nav',monitorStatus:'live'}));
  assert.equal(guided.branches.length,2);
  assert.equal(guided.branches[0].enabled,true);
  assert.equal(guided.monitor,'LIVE SOURCE VIEW');
  assert.match(guided.provenance,/REFERENCE SIMULATION/);
  console.log('PASS XR14 guided source branch state and feature-gated immersive AR entry');
  assert.equal(await proceduralXR.locator('#sbtn').count(),1,'Original desktop splash and clinical workflow are preserved');
  assert.deepEqual(proceduralErrors,[],'No original procedural XR adapter initialization errors');
  console.log('PASS original procedural PNEUMO XR: authentic source anatomy, pathology, scope bridge and desktop compatibility');
 }finally{await proceduralXR.close()}
 // XR17A launcher status is feature-detected, without claiming room
 // registration from immersive-ar support.
 const capabilityLauncher=await browser.newPage({viewport:{width:1050,height:760}});
 try{
  await capabilityLauncher.goto(host+'/quest-short/',{waitUntil:'domcontentloaded'});
  await capabilityLauncher.waitForFunction(()=>window.pulmoXRLauncherStatus?.()?.probed===true,
    null,{timeout:15000});
  const cap=await capabilityLauncher.evaluate(()=>window.pulmoXRLauncherStatus());
  assert.equal(cap.desktop,'available');
  assert.equal(cap.worldAnchor,'NOT_IMPLEMENTED');
  assert.equal(cap.physicalOcclusion,'NOT_IMPLEMENTED');
  assert.equal(await capabilityLauncher.locator('#desktop-launch').count(),1);
  assert.ok((await capabilityLauncher.locator('#vr-capability').textContent()).length>7);
  assert.ok((await capabilityLauncher.locator('#mr-capability').textContent()).length>7);
  if(!cap.webxrAvailable){
   assert.equal(cap.sessions.vr,'unavailable');
   assert.equal(cap.sessions.mr,'unavailable');
   assert.match(await capabilityLauncher.locator('#mr-capability').textContent(),/unavailable|No mixed/i);
  }
  console.log('PASS XR17A launcher: truthful browser VR/MR capability and direct desktop fallback');
 }finally{await capabilityLauncher.close()}
 // A normal Chrome/Edge browser must enter the original procedural
 // workstation without WebXR or webcam permission dialogs.
 const desktop=await browser.newPage({viewport:{width:1300,height:860}});
 try{
  await desktop.goto(host+'/index.html?xr=procedural&entry=desktop',{waitUntil:'domcontentloaded'});
  await desktop.waitForFunction(()=>document.getElementById('app')?.style.display==='flex',null,{timeout:45000});
  assert.ok(await desktop.locator('#splash').isHidden(),'Desktop entry bypasses headset-only launch UI');
  assert.equal((await desktop.locator('#sdot-l').textContent()).trim(),'DESKTOP',
   'Desktop entry must not request camera hand tracking');
  assert.equal(await desktop.locator('#pulmo-procedural-xr').count(),1,
   'Original procedural XR is available as a headset-specific option, not prerequisite');
  console.log('PASS direct desktop workstation: original clinical reference engine, no Quest or camera required');
 }finally{await desktop.close()}
 // XR16 final: the evidence lab must work from an ordinary browser and
 // must NEVER infer clinical validity from a synthetic test bench fixture.
 const validationPage=await browser.newPage({viewport:{width:1300,height:900}});
 try{
  await validationPage.goto(host+'/experimental/validation-lab.html',{waitUntil:'domcontentloaded'});
  await validationPage.waitForFunction(()=>!!window.PulmoXRInstrumentBench?.assess
    && !!window.PulmoXRClinicalReview?.assess);
  assert.ok(await validationPage.getByText('Clinical review and physical instrument verification.').isVisible());
  assert.equal(await validationPage.locator('#benchBadge').innerText(),'BENCH · AWAITING TRACE');
  // Isolated packet verifier coexists with the measured-trace and clinical
  // reviewer gates; it does not promote itself to simulated scope control.
  await validationPage.waitForFunction(()=>!!window.PulmoXRInstrumentTelemetry?.parseFrame);
  await validationPage.locator('#inspectPacket').click();
  assert.match(await validationPage.locator('#packetReport').textContent(),/UNCALIBRATED/);
  await validationPage.locator('#zeroPacket').click();
  assert.match(await validationPage.locator('#packetReport').textContent(),/BENCH_ONLY/);
  await validationPage.locator('#inspectPacket').click();
  assert.match(await validationPage.locator('#packetReport').textContent(),/STALE_OR_REPLAYED_SEQUENCE/);

  const samples=Array.from({length:110},(_,i)=>({
    tMs:i*60,
    cmdInsertionMm:i*.25,measInsertionMm:i*.25+.8,
    cmdTipHDeg:15,measTipHDeg:16,
    cmdTipVDeg:-10,measTipVDeg:-9,
    cmdRollDeg:i*3,measRollDeg:i*3+1,
    tracking:true
  }));
  const trace={
    version:'pulmosimu-instrument-trace/1',
    provenance:{deviceModel:'Synthetic test fixture',instrumentId:'Synthetic QA A1',
      referenceSensor:'Synthetic reference sensor',operator:'Automated browser test',
      measuredAt:'2026-11-24T12:00:00Z',measurementSource:'external-bench'},
    units:{time:'ms',insertion:'mm',angles:'deg'},samples
  };
  await validationPage.locator('#trace').setInputFiles({
    name:'synthetic-browser-fixture.json',mimeType:'application/json',
    buffer:Buffer.from(JSON.stringify(trace))
  });
  await validationPage.locator('#evaluate').click();
  await validationPage.waitForFunction(
    ()=>document.querySelector('#benchReport')?.textContent?.includes('BENCH_ENGINEERING_TARGET_MET'),
    null,{timeout:15000}
  );
  assert.match(await validationPage.locator('#benchReport').textContent(),/BENCH_ENGINEERING_TARGET_MET/);
  assert.match(await validationPage.locator('#benchReport').textContent(),/"clinicalValidated": false/);
  await validationPage.locator('#review').click();
  assert.match(await validationPage.locator('#clinicalReport').textContent(),/INCOMPLETE_REVIEW_PACKET/);
  assert.match(await validationPage.locator('#clinicalReport').textContent(),/"clinicalValidated": false/);
  console.log('PASS XR16F browser evidence lab: measured trace fixture, clinician gate, no false clinical certification');
 }finally{await validationPage.close()}

 // Retired launcher routes must never open the competing XR-04/06 viewer.
 const legacyEntry=await browser.newPage({viewport:{width:1100,height:760}});
 try{
   await legacyEntry.goto(host+'/quest4.html?focus=LLL',{waitUntil:'domcontentloaded'});
   await legacyEntry.waitForURL(url=>url.pathname.endsWith('/index.html')&&
     url.searchParams.get('xr')==='procedural'&&url.searchParams.get('focus')==='LLL',
     {timeout:20000});
   await legacyEntry.waitForFunction(()=>window.pulmoProceduralXRDebug?.()?.referenceFocus==='LLL',
     null,{timeout:45000});
   assert.equal(await legacyEntry.locator('#pulmo-procedural-xr').count(),1);
   console.log('PASS retired XR04 launcher now routes into the original-engine XR09 workspace');
 }finally{await legacyEntry.close()}
 const questEntry=await browser.newPage({viewport:{width:1050,height:760}});
 try{
   await questEntry.goto(host+'/quest.html?focus=RUL',{waitUntil:'domcontentloaded'});
   await questEntry.waitForURL(url=>url.pathname.endsWith('/index.html')&&url.searchParams.get('xr')==='procedural',
     {timeout:20000});
   await questEntry.waitForFunction(()=>window.pulmoProceduralXRDebug?.()?.referenceFocus==='RUL',
     null,{timeout:45000});
   const result=await questEntry.evaluate(()=>{
     const T=window.THREE;
     const globe=new T.Mesh(new T.SphereGeometry(1,16,12),
       new T.MeshBasicMaterial({side:T.DoubleSide}));
     const hit=window.PulmoXRSurface.contact(T,[globe],new T.Vector3(.98,0,0),{radius:.16});
     const fakeBox=window.PulmoXRSurface.contact(T,[globe],new T.Vector3(.95,.95,.95),{radius:.16});
     const far=window.PulmoXRSurface.contact(T,[globe],new T.Vector3(2,0,0),{radius:.16});
     return {hit,fakeBox,far};
   });
   assert.equal(result.hit.hit,true,'Real mesh surface should be grabbable close to contact');
   assert.equal(result.fakeBox.hit,false,'Empty volume inside a bounding box is not a physical surface');
   assert.equal(result.far.hit,false,'Distant gestures cannot grab anatomy');
   const remote=await questEntry.evaluate(()=>{
     const T=window.THREE;
     const root=new T.Group();
     const sphere=new T.Mesh(new T.SphereGeometry(1,32,16),
       new T.MeshBasicMaterial({side:T.DoubleSide}));
     sphere.position.set(0,0,-1.5);root.add(sphere);
     const pointer=new T.Group();pointer.position.set(0,0,0);
     const front=window.PulmoXRRemoteTarget.pick(T,pointer,[root]);
     pointer.position.set(.95,.95,0);
     const emptyCorner=window.PulmoXRRemoteTarget.pick(T,pointer,[root]);
     pointer.position.set(0,0,0);pointer.rotation.y=Math.PI;
     const lookingAway=window.PulmoXRRemoteTarget.pick(T,pointer,[root]);
     pointer.rotation.y=0;
     const tooFar=window.PulmoXRRemoteTarget.pick(T,pointer,[root],{maxDistance:.30});
     const noSource=window.PulmoXRRemoteTarget.pick(T,pointer,[]);
     return {
       front:{hit:front.hit,reason:front.reason,distance:front.distanceM,
         pointZ:front.point?.z},
       emptyCorner:{hit:emptyCorner.hit,reason:emptyCorner.reason},
       lookingAway:{hit:lookingAway.hit,reason:lookingAway.reason},
       tooFar:{hit:tooFar.hit,reason:tooFar.reason},
       noSource:{hit:noSource.hit,reason:noSource.reason}
     };
   });
   assert.equal(remote.front.hit,true,'Quest remote trigger/grip must select actual source triangle 1.5m away');
   assert.equal(remote.front.reason,'triangle-ray');
   // The actual triangle hit may be the front or rear surface at a mesh seam;
   // the contract is a bounded *real* triangle, not a guessed bounding box.
   assert.ok(remote.front.distance>0&&remote.front.distance<=3.5,
     'Ray hits a triangle within the selected 3.5m range: '+JSON.stringify(remote));
   assert.ok(remote.front.pointZ<-.25&&remote.front.pointZ>-3.0,
     'Hit point is on the actual transformed sphere surface');
   assert.equal(remote.emptyCorner.hit,false,'Ray through an empty bounding-box corner cannot grab');
   assert.equal(remote.lookingAway.hit,false,'Pointing away never moves anatomy');
   assert.equal(remote.tooFar.hit,false,'Bounded distant selection must honour reach limit');
   assert.equal(remote.noSource.hit,false,'No anatomy source means no grab');
   const backface=await questEntry.evaluate(()=>{
     const T=window.THREE,mat=new T.MeshBasicMaterial({side:T.BackSide});
     const mesh=new T.Mesh(new T.SphereGeometry(.35,24,12),mat);
     mesh.position.set(0,0,-1.1);
     const p=new T.Group();
     const front=window.PulmoXRRemoteTarget.pick(T,p,[mesh],{maxDistance:2});
     const state=mat.side;
     return {hit:front.hit,distance:front.distanceM,sideRestored:state===T.BackSide};
   });
   assert.equal(backface.hit,true,'Outward-facing and reversed imported triangles both selectable');
   assert.ok(backface.distance>.70&&backface.distance<.80,
      'Ray terminates at NEAREST source triangle instead of far lung wall: '+JSON.stringify(backface));
   assert.equal(backface.sideRestored,true,'Source-authored material side is never modified');
   const routing=await questEntry.evaluate(()=>{
     const T=window.THREE;
     const frontMenu=new T.Mesh(new T.PlaneGeometry(.82,.48),
       new T.MeshBasicMaterial({side:T.DoubleSide}));
     frontMenu.position.set(0,0,-.75);frontMenu.updateMatrixWorld(true);
     const ray=new T.Raycaster(new T.Vector3(),new T.Vector3(0,0,-1),0,3.5);
     const hit=ray.intersectObject(frontMenu,false)[0];
     const anatomy=new T.Mesh(new T.SphereGeometry(.23,16,12),
       new T.MeshBasicMaterial({side:T.DoubleSide}));
     anatomy.position.set(0,0,-1.4);
     const p=new T.Group();
     const actual=window.PulmoXRRemoteTarget.pick(T,p,[anatomy]);
     const front=window.PulmoXRSpatialTarget.choose({
       panel:{distance:hit.distance,action:'reset'},anatomy:actual,mode:'anatomy'});
     const behind=window.PulmoXRSpatialTarget.choose({
       panel:{distance:2.2,action:'reset'},anatomy:actual,mode:'anatomy'});
     const blank=window.PulmoXRSpatialTarget.choose({
       panel:{distance:hit.distance,action:null},anatomy:actual,mode:'anatomy'});
     return {front:{kind:front.kind,action:front.action},behind:behind.kind,
       blank:{kind:blank.kind,action:blank.action},anatomyHit:actual.hit};
   });
   assert.equal(routing.anatomyHit,true);
   assert.equal(routing.front.kind,'panel');
   assert.equal(routing.front.action,'reset');
   assert.equal(routing.behind,'anatomy');
   assert.equal(routing.blank.kind,'panel');
   assert.equal(routing.blank.action,null);
   const spatial=await questEntry.evaluate(()=>{
     const T=window.THREE,room=new T.Group();
     const floor=new T.Mesh(new T.CircleGeometry(3.8,32),
       new T.MeshStandardMaterial({transparent:true,opacity:1}));
     const grid=new T.Group(),plinth=new T.Group(),ring=new T.Group();
     room.add(floor,grid,plinth,ring);
     const presence=window.PulmoXRStagePresence.create(T,room,{floor,grid,plinth,rings:[ring]});
     const original=presence.snapshot();
     const singleLine=presence.root.children.filter(o=>o.isLineSegments).length;
     presence.setMode('focus');
     const focused={mode:presence.snapshot().mode,grid:grid.visible,
       plinth:plinth.visible,ring:ring.visible,floorOpacity:floor.material.opacity};
     presence.setMode('studio');
     const studio={mode:presence.snapshot().mode,grid:grid.visible,
       plinth:plinth.visible,ring:ring.visible,floorOpacity:floor.material.opacity};
     return {original,focused,studio,singleLine};
   });
   assert.equal(spatial.original.mode,'studio');
   assert.equal(spatial.original.datumSegments,24,'World references use 12 vertical + 12 rim segments');
   assert.equal(spatial.singleLine,1,'Architecture is a single LineSegments object');
   assert.equal(spatial.focused.grid,false);
   assert.equal(spatial.focused.plinth,false);
   assert.equal(spatial.focused.ring,false);
   assert.ok(spatial.focused.floorOpacity<1);
   assert.equal(spatial.studio.grid,true);
   assert.equal(spatial.studio.plinth,true);
   assert.equal(spatial.studio.ring,true);
   assert.equal(spatial.studio.floorOpacity,1);
   console.log('PASS XR13 world stage, single-draw-call architecture and focus/studio mode switching');
   console.log('PASS XR12 actual Three plane, source geometry and no-click-through hit arbitration');
   console.log('PASS XR11 front/back ray targeting and source material preservation');
   console.log('PASS XR10 remote triangle selection with real Three.js geometry');
   console.log('PASS canonical Quest entry and source mesh-triangle near contact');
 }finally{await questEntry.close()}
 const studio=await browser.newPage({viewport:{width:1440,height:900}});
 studio.on('pageerror',error=>console.error('STUDIO PAGE ERROR',error.message));
 try{
   await studio.goto(host+'/studio.html',{waitUntil:'domcontentloaded'});
   assert.equal(await studio.locator('#workspaceTitle').innerText(),'Procedural simulation');
   // Guided session is a separate panel and R3 actually runs.
   assert.equal(await studio.locator('#guidePanel').isHidden(),true);
   await studio.locator('#guideToggle').click();
   assert.equal(await studio.locator('#guidePanel').isVisible(),true);
   assert.equal(await studio.locator('#guideToggle').getAttribute('aria-expanded'),'true');
   await studio.locator('#guideDone').click();
   assert.match(await studio.locator('#guideProgress').innerText(),/1 manually reviewed/);
   await studio.locator('#guideNext').click();
   assert.match(await studio.locator('#guideTitle').innerText(),/Rehearse reference bronchoscopy/);
   await studio.locator('#guideRunR3').click();
   await studio.waitForFunction(()=>document.getElementById('guidePhysics')?.textContent?.includes('Peak inspiratory flow'),null,{timeout:30000});
   const report=await studio.evaluate(()=>window.PulmoStudioGuidedSession?.report());
   assert.equal(report.schema,'pulmosimu-guided-research-session-v1');
   assert.equal(report.studyDataIncluded,false);
   assert.equal(report.r3Reference?.branchCount,18);
   assert.equal(report.r3Reference?.clinicalValidated,false);
   await studio.locator('#guideClose').click();
   assert.equal(await studio.locator('#guidePanel').isHidden(),true);
   await studio.locator('#modeMechanisms').click();
   assert.equal(await studio.locator('#workspaceTitle').innerText(),'Mechanistic laboratory');
   await studio.waitForFunction(()=>document.getElementById('workspaceFrame')?.getAttribute('src')==='./experimental/lab.html');
   await studio.frameLocator('#workspaceFrame').locator('#networkStatus').waitFor({state:'visible',timeout:45000});
   await studio.waitForFunction(()=>document.getElementById('workspaceFrame')?.contentDocument?.getElementById('networkStatus')?.textContent?.includes('SI-unit'),null,{timeout:45000});
   await studio.frameLocator('#workspaceFrame').locator('#focusRegion').selectOption('LLL');
   await studio.waitForTimeout(500);
   console.log('FOCUS DIAGNOSTIC',JSON.stringify(await studio.evaluate(()=>({
      context:document.getElementById('workspaceContext').textContent,
      path:document.getElementById('workspaceFrame').getAttribute('src'),
      selected:document.getElementById('workspaceFrame').contentDocument?.getElementById('focusRegion')?.value
   }))));
   await studio.waitForFunction(()=>document.getElementById('workspaceContext')?.textContent?.includes('REFERENCE LLL'),null,{timeout:15000});
   await studio.locator('#modeProcedure').click();
   await studio.frameLocator('#workspaceFrame').locator('#status').waitFor({state:'visible',timeout:30000});
   await studio.waitForFunction(()=>document.getElementById('workspaceFrame')?.contentDocument?.getElementById('status')?.textContent?.includes('Reference focus: LLL'),null,{timeout:15000});
   await studio.locator('#modeMechanisms').click();
   await studio.frameLocator('#workspaceFrame').locator('#focusRegion').waitFor({state:'visible',timeout:45000});
   await studio.waitForFunction(()=>document.getElementById('workspaceFrame')?.contentDocument?.getElementById('focusRegion')?.value==='LLL',null,{timeout:15000});
   assert.equal(await studio.frameLocator('#workspaceFrame').locator('#focusRegion').inputValue(),'LLL');
   await studio.locator('#modeXR').click();
   assert.equal(await studio.locator('#workspaceTitle').innerText(),'Immersive procedural laboratory');
   await studio.waitForFunction(()=>document.getElementById('workspaceFrame')
     ?.contentWindow?.pulmoProceduralXRDebug?.()?.referenceFocus==='LLL',null,{timeout:45000});
   assert.ok((await studio.locator('#workspaceFrame').getAttribute('src')).includes('/index.html?xr=procedural'),
     'Studio must embed the original anatomical engine rather than old schematic viewer');
   const questLink=await studio.locator('#launchXR').getAttribute('href');
   assert.ok(questLink.includes('focus=LLL')&&questLink.includes('index.html'),
     'Direct Quest entry carries only reference lobe context into original procedure');
   assert.equal(await studio.frameLocator('#workspaceFrame').locator('#pulmo-procedural-xr').count(),1);
   // Do not claim that XR silently transfers new procedure or patient state.
   await studio.locator('#modeMechanisms').click();
   await studio.frameLocator('#workspaceFrame').locator('#focusRegion').waitFor({state:'visible',timeout:45000});
   assert.equal(await studio.frameLocator('#workspaceFrame').locator('#focusRegion').inputValue(),'LLL');
   await studio.locator('#modeProcedure').click();
   assert.equal(await studio.locator('#workspaceTitle').innerText(),'Procedural simulation');
   assert.equal(await studio.locator('#modeProcedure').getAttribute('aria-pressed'),'true');
   assert.equal(await studio.locator('#workspaceFrame').getAttribute('src'),'./procedural-workspace.html');
   // CT anatomy is a separate, source-only review surface. It must neither
   // import nor change the original procedural airway/navigation state.
   await studio.locator('#modeSource').click();
   assert.equal(await studio.locator('#workspaceTitle').innerText(),'CT anatomical source inspector');
   await studio.frameLocator('#workspaceFrame').locator('#source-files')
     .waitFor({state:'visible',timeout:45000});
   const source=studio.frameLocator('#workspaceFrame');
   for(const id of ['#ct-file','#mesh-file','#manifest-file','#mpr-axial',
     '#mpr-coronal','#mpr-sagittal','#clip-toggle','#cross-3d'])
     assert.equal(await source.locator(id).count(),1,'Missing source-only anatomy control '+id);
   assert.equal(await source.locator('#clip-toggle').isDisabled(),true,
     'No source manifest and locally matched NIfTI/GLB means no 3D registration');
   assert.equal(await source.locator('#cross-3d').isDisabled(),true);
   for(const control of ['#section-plane','#surface-opacity','#source-gates','#structure-qa-file'])
     assert.equal(await source.locator(control).count(),1,
       'Missing A3.1-A4 source inspector '+control);
   assert.equal(await source.locator('#section-plane').isDisabled(),true);
   assert.equal(await source.locator('#surface-opacity').isDisabled(),true);
   for(const id of ['#metric-a','#metric-b','#metric-clear','#metric-result'])
     assert.equal(await source.locator(id).count(),1,'Missing CT metric '+id);
   assert.equal(await source.locator('#metric-a').isDisabled(),true);
   assert.equal(await source.locator('#metric-b').isDisabled(),true);
   // A2.3 review is strictly source-verified: without the three exact local
   // sources, none of the annotation/import/export controls can mutate state.
   for(const control of ['#review-region','#review-finding','#review-note',
     '#review-add','#review-import','#review-export','#review-clear','#review-list'])
     assert.equal(await source.locator(control).count(),1,
       'Missing CT-local review control '+control);
   assert.equal(await source.locator('#review-add').isDisabled(),true);
   assert.equal(await source.locator('#review-import').isDisabled(),true);
   assert.equal(await source.locator('#review-export').isDisabled(),true);
   assert.equal(await source.locator('#review-clear').isDisabled(),true);
   assert.match(await source.locator('#review-status').innerText(),/Exact matching CT, GLB and manifest required/);
   assert.equal(await source.locator('body').evaluate(()=>
     window.PulmoCTReview?.SCHEMA),'pulmosimu-ct-point-review-v1');
   const clinicalGate=await source.locator('body').evaluate(()=>
     window.PulmoSourceAnatomyGate?.assess({}).CTLinkedBronchoscopyAllowed);
   assert.equal(clinicalGate,false,'CT procedures require independent source airway review');
   await studio.locator('#modeProcedure').click();
   assert.equal(await studio.locator('#workspaceTitle').innerText(),'Procedural simulation');
   console.log('PASS A1 CT anatomy: source-only import, locked unverified registration, original procedure preserved');
   console.log('PASS integrated Studio: original procedural XR and separate research mechanisms share canonical context only');
 } finally {await studio.close();}
 const procedural=await browser.newPage({viewport:{width:1440,height:900}});
 try{
   await procedural.goto(host+'/procedural-workspace.html',{waitUntil:'domcontentloaded'});
   await procedural.frameLocator('#frame').locator('#sbtn').waitFor({state:'attached',timeout:45000});
   await procedural.waitForFunction(()=>document.getElementById('frame')?.contentDocument?.getElementById('pl-ui-style')!==null,null,{timeout:15000});
   assert.equal(await procedural.locator('#modeFocus').getAttribute('aria-pressed'),'true');
   const childDoc=procedural.frameLocator('#frame');
   assert.ok(await childDoc.locator('#c-ext').count()>0);
   assert.ok(await childDoc.locator('#c-bro').count()>0);
   await procedural.locator('#modeInspector').click();
   assert.equal(await procedural.locator('#modeInspector').getAttribute('aria-pressed'),'true');
   assert.ok(await childDoc.locator('body.pl-inspector').count()>0);
   await procedural.locator('#modeGestures').click();
   assert.ok(await childDoc.locator('body.pl-gestures').count()>0);
   await procedural.locator('#modeFocus').click();
   assert.ok(await childDoc.locator('body.pl-focus').count()>0);
   await procedural.locator('#viewDual').click();
   assert.equal(await procedural.locator('#viewDual').getAttribute('aria-pressed'),'true');
   await procedural.locator('#viewExternal').click();
   assert.equal(await procedural.locator('#viewExternal').getAttribute('aria-pressed'),'true');
   const fitAvailable=await procedural.locator('#frame').evaluate(frame=>typeof frame.contentWindow.plFitAnatomy==='function');
   assert.ok(fitAvailable,'Original procedural camera is exposed through non-destructive framing helper');
   console.log('PASS procedural quality layer: focus mode, inspector, gestures, preserved canvases, view controls');
 } finally {await procedural.close();}
 const legacy=await browser.newPage({viewport:{width:1360,height:820}});
 try {
   await legacy.goto(host+'/',{waitUntil:'domcontentloaded',timeout:45000});
   assert.match(await legacy.title(),/PNEUMO/i,'Original surgical application title');
   assert.ok(await legacy.locator('#c-ext').count()>0,'Original external 3D procedure canvas exists');
   assert.ok(await legacy.locator('#c-bro').count()>0,'Original bronchoscopic procedure canvas exists');
   const spatialSource=await legacy.evaluate(()=>{
     const THREE=window.THREE,api=window.PulmoXRCTSpatial;
     if(!THREE||!api||!window.PulmoNiftiSource)throw Error('E2 source XR scripts not loaded');
     const affine=[[1,0,0,-4],[0,-1,0,2],[0,0,2,-20],[0,0,0,1]];
     const volume={
       dims:[4,3,2],affine,spacing:[1,1,2],slope:1,intercept:-1024,
       ras:(i,j,k)=>[i-4,2-j,2*k-20],voxel:(i,j,k)=>i*100+j*20+k*10-1024
     };
     const manifest={
       source_shape_vox:volume.dims,sform_RAS_mm:affine,
       HU_stored_slope:1,HU_stored_intercept:-1024,
       source_sha256:'a'.repeat(64),mesh_sha256:'b'.repeat(64),
       model_from_ras_mm:[[.001,0,0,0],[0,0,.001,.39],[0,-.001,0,.12],[0,0,0,1]]
     };
     const study={volume,manifest,ctSha:'a'.repeat(64),meshSha:'b'.repeat(64),meshVerified:true};
     const sceneGroup=new THREE.Group();
     const initial=sceneGroup.children.length;
     const view=api.create(THREE,sceneGroup,study);
     view.setVisible(true);
     const before=view.snapshot();
     const after=view.step('axial',1);
     const drawn=view.group.children.length;
     const labels=view.group.children.map(x=>x.name).sort();
     view.dispose();
     const domain=window.PulmoXRScience.classify({modelRole:'patient',ctStudy:study,referenceSolverReady:true});
     return {initial,drawn,labels,before:before.sections.axial,
       after:after.sections.axial,remaining:sceneGroup.children.length,
       physics:domain.respiratoryPhysics,procedure:domain.bronchoscopy};
   });
   assert.equal(spatialSource.initial,0);
   assert.equal(spatialSource.drawn,3,'Source-backed anatomical CT has three real voxel sections');
   assert.equal(spatialSource.remaining,0,'CT spatial resources disposed without changing original scene');
   assert.notEqual(spatialSource.before,spatialSource.after,'CT slice control advances actual voxel index');
   assert.equal(spatialSource.physics,'NOT_ATTACHED');
   assert.equal(spatialSource.procedure,'CT_SPECIFIC_PROCEDURE_BLOCKED');
   console.log('PASS E1–E4 XR: source MPR three-plane rendering, cleanup, source-science safety');
   console.log('PASS original surgical application: expected canvases and page preserved');
 } finally {await legacy.close();}

} finally {
 await browser.close();
}
