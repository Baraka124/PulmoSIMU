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
 await redirect.waitForURL(url=>url.pathname==='/',{timeout:30000});
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
  assert.equal(xr16.mrPlacementEvidence.state,'idle',
    'Physical MR placement must never run in ordinary desktop canvas mode');
  assert.equal(xr16.mrPlacementEvidence.userConfirmed,false);
  assert.equal(xr16.mrPlacementEvidence.physicalDepthOcclusion,false);
  assert.equal(xr16.XR17CInstrumentTrace.loaded,false);
  assert.equal(xr16.XR17CInstrumentTrace.deviceToScopeCoupled,false);
  assert.equal(xr16.XR17CDisplayPath.compositorQuadActive,false);
  assert.equal(xr16.XR17CTrackedControllerProxy.enabled,false);
  assert.equal(await proceduralXR.locator('#pulmo-xr-trace-file').count(),1);
  const scopeBeforeImport=xr16.scopeBranch;
  const fakeBenchFixture={
    version:'pulmosimu-instrument-trace/1',
    provenance:{deviceModel:'Synthetic software test fixture',
      instrumentId:'TEST-ONLY-001',referenceSensor:'SYNTHETIC-TEST-ENCODER',
      operator:'Automated CI fixture',measuredAt:'2026-11-25T09:00:00.000Z',
      measurementSource:'external-bench'},
    units:{time:'ms',insertion:'mm',angles:'deg'},
    samples:Array.from({length:60},(_,i)=>({
      tMs:100*i,cmdInsertionMm:2*i,measInsertionMm:2*i+1,
      cmdTipHDeg:4,measTipHDeg:4.5,cmdTipVDeg:-2,measTipVDeg:-2,
      cmdRollDeg:10,measRollDeg:11,tracking:true
    }))
  };
  await proceduralXR.locator('#pulmo-xr-trace-file').setInputFiles({
    name:'xr17c-synthetic-ci-fixture.json',mimeType:'application/json',
    buffer:Buffer.from(JSON.stringify(fakeBenchFixture))
  });
  await proceduralXR.waitForFunction(()=>window.pulmoProceduralXRDebug?.()?.XR17CInstrumentTrace.loaded===true,
    null,{timeout:20000});
  const postImport=await proceduralXR.evaluate(()=>window.pulmoProceduralXRDebug());
  assert.equal(postImport.XR17CInstrumentTrace.total,60);
  assert.equal(postImport.XR17CInstrumentTrace.clinicalValidated,false);
  assert.equal(postImport.XR17CInstrumentTrace.deviceToScopeCoupled,false);
  assert.equal(postImport.scopeBranch,scopeBeforeImport,
    'Independent measured trace import must never move original scope');
  assert.equal(postImport.XR17CTrackedControllerProxy.physicalBronchoscopeConnected,false);
  console.log('PASS XR17C browser: local measured-trace import, safe display and tracked-proxy boundaries');
  assert.equal(await proceduralXR.evaluate(()=>typeof window.PulmoXRMRPlacement?.create),'function');
  const mrProbe=await proceduralXR.evaluate(async()=>{
    const m=window.PulmoXRMRPlacement.create();
    await m.start({
      requestReferenceSpace:async()=>({type:'viewer'}),
      requestHitTestSource:async()=>({cancel(){}})
    });
    const frame={getHitTestResults:()=>[{
      getPose:()=>({transform:{position:{x:.65,y:0,z:-.8},orientation:{x:0,y:0,z:0,w:1}}})
    }]};
    const ref={kind:'reference'};
    const target=m.scan(frame,ref);
    const unconfirmed=m.snapshot().userConfirmed;
    const accepted=m.confirm().accepted;
    const placed=m.scan(frame,ref);
    const safe=m.snapshot();
    m.reset();
    return {hit:target?.kind,unconfirmed,accepted,placed:placed?.kind,
      userConfirmed:safe.userConfirmed,anchorStatus:safe.anchorStatus,
      stateAfterReset:m.snapshot().state};
  });
  assert.equal(mrProbe.hit,'candidate');
  assert.equal(mrProbe.unconfirmed,false);
  assert.equal(mrProbe.accepted,true);
  assert.equal(mrProbe.placed,'placed');
  assert.equal(mrProbe.userConfirmed,true);
  assert.equal(mrProbe.anchorStatus,'not-anchored');
  assert.equal(mrProbe.stateAfterReset,'idle');
  console.log('PASS XR17B browser: actual hit result required, explicit confirmation, no fake anchor');
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
  assert.equal(await capabilityLauncher.locator('body.ps-launch').count(),1);
  for(const width of [390,760,1440]){
    await capabilityLauncher.setViewportSize({width,height:900});
    const q=await capabilityLauncher.evaluate(()=>({scroll:document.documentElement.scrollWidth,viewport:innerWidth,links:[...document.querySelectorAll('.mode')].map(a=>({width:a.getBoundingClientRect().width,height:a.getBoundingClientRect().height}))}));
    assert.ok(q.scroll<=q.viewport+1,JSON.stringify(q));
    assert.ok(q.links.every(a=>a.width>240&&a.height>100));
  }
  await capabilityLauncher.screenshot({path:'/tmp/pulmosimu-ui01-launcher.png',fullPage:true});
  console.log('PASS UI01 responsive launcher and original XR links');
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
  assert.equal(await desktop.locator('#pulmo-launch-dock #pulmo-procedural-xr').count(),1);
  assert.equal(await desktop.locator('#pulmo-launch-dock #pulmo-xr-session-report').count(),1);
  assert.ok(await desktop.locator('#pulmo-launch-dock').isVisible());
  await desktop.locator('.ps-dock-tools>summary').click();
  assert.ok(await desktop.locator('#pulmo-xr-trace-import').isVisible());
  await desktop.locator('.ps-dock-tools>summary').click();
  const ui=await desktop.evaluate(()=>{
   const U=window.PulmoXRSpatialUI,canvas=document.createElement('canvas');canvas.width=1024;canvas.height=600;const c=canvas.getContext('2d');U.frame(c,1024,600);U.header(c,1024,'PulmoSIMU / Spatial laboratory','Bronchoscopy workstation');
   const title=U.text(c,'Extremely long source and instrument label that should be fitted rather than overflow the spatial panel',42,180,330,{size:28,min:21});
   const fit=c.measureText(title).width;
   U.control(c,{x:50,y:324,w:440,h:70,title:'PROCEDURE STATION',hovered:true,available:true,action:'view'});
   return {fit,ink:c.getImageData(100,350,1,1).data[3],corner:c.getImageData(0,0,1,1).data[3]};
  });
  assert.ok(ui.fit<=330.01&&ui.ink===255&&ui.corner===0,JSON.stringify(ui));
  await desktop.screenshot({path:'/tmp/pulmosimu-ui01-desktop.png'});
  console.log('PASS UI01 desktop dock, measured canvas label and frame rendering');
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
  await validationPage.locator('#configureMapping').click();
  assert.match(await validationPage.locator('#mappingReport').textContent(),/AWAITING_INPUT/);
  await validationPage.locator('#armMapping').click();
  await validationPage.locator('#mapPacket').click();
  assert.match(await validationPage.locator('#mappingReport').textContent(),/BENCH_MAPPING_ONLY/);
  await validationPage.locator('#mapPacket').click();
  assert.match(await validationPage.locator('#mappingReport').textContent(),/SAFETY_STOP/);
  assert.match(await validationPage.locator('#mappingReport').textContent(),/REPLAYED_SAMPLE/);
  console.log('PASS consolidated physical and clinical verification lab with fail-closed mapping');
 }finally{await validationPage.close()}

 // Legacy quest/quest4 HTML files are removed. Vercel redirects bookmarks to /.
 const legacyEntry=await browser.newPage({viewport:{width:1100,height:760}});
 try{
   await legacyEntry.goto(host+'/index.html?entry=vr&focus=LLL',{waitUntil:'domcontentloaded'});
   await legacyEntry.waitForFunction(()=>window.pulmoProceduralXRDebug?.()?.referenceFocus==='LLL',
     null,{timeout:45000});
   assert.equal(await legacyEntry.locator('#pulmo-procedural-xr').count(),1);
   console.log('PASS canonical Quest entry keeps original source focus');
 }finally{await legacyEntry.close()}
 const questEntry=await browser.newPage({viewport:{width:1050,height:760}});
 try{
   await questEntry.goto(host+'/index.html?entry=vr&focus=RUL',{waitUntil:'domcontentloaded'});
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
   assert.ok((await studio.locator('#workspaceFrame').getAttribute('src')).includes('/index.html?entry=vr'),
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
   assert.match(await legacy.title(),/PulmoSIMU/i,'Consolidated original research workstation title');
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
   const lumenEvidence=await legacy.evaluate(()=>{
     if(typeof window.ensureUnifiedLumen!=='function')throw Error('XR22 lumen builder unavailable');
     const mesh=window.ensureUnifiedLumen();
     if(!mesh)throw Error('XR22 actual source lumen fell back: '+JSON.stringify(lumenStatus));
     mesh.updateWorldMatrix(true,false);
     const tests=[];
     for(const parent of SEGS){
       if(!parent.children.length)continue;
       for(const childId of parent.children){
         const curve=pathCurves[childId];
         const a=pathCurves[parent.id].getPoint(1).applyMatrix4(mesh.matrixWorld);
         const b=curve.getPoint(.08).applyMatrix4(mesh.matrixWorld);
         const delta=b.clone().sub(a),len=delta.length();
         const ray=new THREE.Raycaster(a,delta.normalize(),.00001,len*.92);
         tests.push({parent:parent.id,child:childId,wallHits:ray.intersectObject(mesh,false).length});
       }
     }
     return {triangles:mesh.geometry.attributes.position.count/3,visible:mesh.visible,
       status:lumenStatus.state,step:lumenStatus.step,tests};
   });
   assert.equal(lumenEvidence.status,'UNIFIED_REFERENCE_SURFACE');
   assert.equal(lumenEvidence.visible,false,'Lumen is isolated from the external anatomical scene');
   assert.ok(lumenEvidence.triangles>1000&&lumenEvidence.triangles<=240000);
   assert.ok(lumenEvidence.tests.length>10);
   assert.ok(lumenEvidence.tests.every(t=>t.wallHits===0),'Actual source junctions are obstructed: '+JSON.stringify(lumenEvidence.tests));
   console.log('PASS XR22 actual reference lumen: '+lumenEvidence.triangles+' triangles / '+lumenEvidence.tests.length+' open junction paths / step '+lumenEvidence.step);
   const instrumentEvidence=await legacy.evaluate(()=>{
     const THREE=window.THREE,M=window.PulmoXRVideoScope;
     const t=new THREE.Vector3(0,0,-1);
     const a=M.poseBasis(THREE,t,0,.6,0,32),b=M.poseBasis(THREE,t,Math.PI/2,.6,0,32);
     const device=M.createVisual(THREE);
     device.update({h:.6,v:-.4,roll:.5,diameterMm:4.2});
     let meshes=0,finite=true;
     device.group.traverse(o=>{if(o.isMesh){meshes++;finite=finite&&o.position.toArray().every(Number.isFinite)&&o.scale.toArray().every(Number.isFinite)}});
     return {meshes,finite,orthogonality:Math.abs(a.direction.dot(a.up)),unit:a.direction.length(),rollChangesDirection:a.direction.distanceTo(b.direction),horizontalFov:M.REF.horizontalFovDeg,depth:M.REF.depthOfFieldMm};
   });
   assert.ok(instrumentEvidence.meshes>40&&instrumentEvidence.finite);
   assert.ok(instrumentEvidence.orthogonality<1e-6&&Math.abs(instrumentEvidence.unit-1)<1e-6);
   assert.ok(instrumentEvidence.rollChangesDirection>.2);
   assert.equal(instrumentEvidence.horizontalFov,110);
   console.log('PASS XR23 device assembly and roll-coupled camera basis');
   const respiratoryEvidence=await legacy.evaluate(()=>{
     const p=SEGS[0].pts[0],mesh=window.ensureUnifiedLumen();
     const old=scope.t;scope.t=.65;setReferenceMotion(1);updateScope();
     const initial=camScope.position.clone(),anchor=p.clone().applyMatrix4(mGrp.matrixWorld);
     const originalParent=mesh.parent;
     setReferenceMotion(1.05);updateScope();
     const moved=camScope.position.distanceTo(initial);
     const anchorError=p.clone().applyMatrix4(motionGrp.matrixWorld).distanceTo(anchor);
     const attached=mesh.parent===motionGrp;
     const scale=motionGrp.scale.x;
     setReferenceMotion(1);updateScope();
     const restored=camScope.position.distanceTo(initial);
     const parentRestored=mesh.parent===originalParent;
     scope.t=old;updateScope();
     return {moved,anchorError,attached,scale,restored,parentRestored};
   });
   assert.ok(respiratoryEvidence.moved>1e-5);
   assert.ok(respiratoryEvidence.anchorError<1e-6);
   assert.equal(respiratoryEvidence.attached,true);
   assert.equal(respiratoryEvidence.scale,1.05);
   assert.ok(respiratoryEvidence.restored<1e-6);
   assert.equal(respiratoryEvidence.parentRestored,true);
   console.log('PASS XR29 original lumen and camera motion, anchor and reversible restoration');
   const deviceIntegration=await legacy.evaluate(()=>{
     const D=window.PulmoXRDeviceLab,T=window.THREE;
     const renderer=rBro,old=renderer.getRenderTarget();
     if(!renderer.capabilities.isWebGL2)return {depthSupported:false};
     const raw=new T.WebGLRenderTarget(64,64,{depthBuffer:true});
     raw.depthTexture=new T.DepthTexture(64,64,T.UnsignedIntType);
     const out=new T.WebGLRenderTarget(64,64);
     const camera=new T.PerspectiveCamera(60,1,.01,10);
     const source=new T.Scene(),plane=new T.Mesh(new T.PlaneGeometry(5,5),new T.MeshBasicMaterial({color:0xb87962}));
     plane.position.z=-1;source.add(plane);
     const optics=D.opticalMaterial(T,raw),screen=new T.Scene(),quad=new T.Mesh(new T.PlaneGeometry(2,2),optics.material);
     screen.add(quad);const ortho=new T.OrthographicCamera(-1,1,1,-1,.01,10);ortho.position.z=1;
     const pixel=new Uint8Array(4),off=new Uint8Array(4);
     try{
       renderer.setRenderTarget(raw);renderer.clear();renderer.render(source,camera);
       optics.update({near:.01,far:10,scale:1,mmPerUnit:100,enabled:true});
       renderer.setRenderTarget(out);renderer.clear();renderer.render(screen,ortho);renderer.readRenderTargetPixels(out,32,32,1,1,pixel);
       optics.update({near:.01,far:10,scale:1,mmPerUnit:100,enabled:false});
       renderer.clear();renderer.render(screen,ortho);renderer.readRenderTargetPixels(out,32,32,1,1,off);
       return {depthSupported:true,on:[...pixel],off:[...off],branch:D.triggerIntent({atBranch:true,side:'right',tool:'forceps'}),native:D.allowNativeDisplay({opticsActive:true,radialEnabled:false,nativeOptIn:true})};
     }finally{renderer.setRenderTarget(old);raw.dispose();out.dispose();plane.geometry.dispose();plane.material.dispose();quad.geometry.dispose();optics.material.dispose();}
   });
   if(deviceIntegration.depthSupported){
     assert.ok(deviceIntegration.on[0]>20&&deviceIntegration.on[3]===255,JSON.stringify(deviceIntegration));
     assert.ok(deviceIntegration.on.some((v,i)=>v!==deviceIntegration.off[i]),'Optical toggle did not change rendered pixels');
     assert.equal(deviceIntegration.branch,'BRANCH');assert.equal(deviceIntegration.native,false);
     console.log('PASS XR32 optical shader actual offscreen pixels and integrated routing');
   }else console.log('XR32 optical pixel test unavailable in non-WebGL2 browser; source fallback only');
   const integratedEvidence=await legacy.evaluate(()=>{
     const U=window.PulmoXRIntegratedUpgrade,T=window.THREE;
     const head=[1,1.4,-.8],spec=U.workspaceTransform({head,forward:[0,0,-1],right:[1,0,0],room:[0,0,0],command:'turn-left'});
     const q=new T.Quaternion().setFromAxisAngle(new T.Vector3(0,1,0),spec.yaw);
     const pivot=new T.Vector3(...head).applyQuaternion(q).add(new T.Vector3(...spec.translation)).distanceTo(new T.Vector3(...head));
     const mesh=window.ensureUnifiedLumen();
     if(!mesh)throw Error('Original reference lumen unavailable');
     const original=scope.t;scope.t=.25;setReferenceMotion(1.03);updateScope();
     const camera=camScope;camera.updateMatrixWorld(true);
     const worldPerMm=motionGrp.getWorldScale(new T.Vector3()).x/mmPerUnit;
     const toSource=p=>motionGrp.worldToLocal(new T.Vector3(...p).applyMatrix4(camera.matrixWorld)).toArray();
     const center=toSource([0,0,0]);
     const field=p=>window.PulmoXRUnifiedLumen.fieldAt(lumenPieces,p);
     const distance=center.reduce((v,x,i)=>v+(x-scope.pos.toArray()[i])**2,0);
     const measured=U.accessoryClearance({lengthMm:2,diameterMm:1.8,worldPerMm,sourceMmPerUnit:mmPerUnit,toSource,field});
     setReferenceMotion(1);scope.t=original;updateScope();
     return {pivot,centerError:Math.sqrt(distance),fieldFinite:Number.isFinite(field(center)),margin:measured.marginMm,decision:measured.state,missing:U.accessoryClearance({}).state};
   });
   assert.ok(integratedEvidence.pivot<1e-8);
   assert.ok(integratedEvidence.centerError<1e-5,JSON.stringify(integratedEvidence));
   assert.ok(integratedEvidence.fieldFinite&&Number.isFinite(integratedEvidence.margin));
   assert.equal(integratedEvidence.missing,'UNAVAILABLE');
   console.log('PASS AG01 workspace rotation pivot and source/deformed-frame accessory field');
   const ag03=await legacy.evaluate(()=>{
     const U=window.PulmoXRIntegratedUpgrade,T=window.THREE;
     const p=U.boundedPanelPosition([0,1.3,0],[4,1.3,0]);
     const moved=U.limitedStep([0,1.3,0],p,.05,1.8);
     const regional=U.createRegionalEvidence();
     regional.step({tSeconds:0,regionalFlowM3PerS:{RUL:.0002,LLL:.0001}});
     regional.step({tSeconds:.05,regionalFlowM3PerS:{RUL:.0002,LLL:.0001}});
     return {distance:new T.Vector3(...p).distanceTo(new T.Vector3(0,1.3,0)),step:new T.Vector3(...moved).distanceTo(new T.Vector3(0,1.3,0)),regional:regional.snapshot().regions.RUL.netM3,environment:U.observeEnvironment(null),contact:U.contactEstimate(-.3,4)};
   });
   assert.ok(Math.abs(ag03.distance-2.1)<1e-8);
   assert.ok(Math.abs(ag03.step-.09)<1e-8);
   assert.ok(Math.abs(ag03.regional-.00001)<1e-10);
   assert.equal(ag03.environment.physicalOcclusion,false);
   assert.equal(ag03.contact.forceN,null);
   console.log('PASS AG03 spatial window bounds, regional integration and honest environment/contact evidence');
   const spatialPhysical=await legacy.evaluate(()=>{
     const T=window.THREE,G=window.PulmoXRSpatialGeometry;
     const canvas=document.createElement('canvas');canvas.width=256;canvas.height=128;const texture=new T.CanvasTexture(canvas);
     const root=new T.Mesh(new T.PlaneGeometry(1,.5),new T.MeshBasicMaterial({map:texture}));
     const shell=G.create(T,root,{width:1,height:.5,radius:1.6,texture,deckStart:.5});
     const actions=[{id:'first',u:.05,v:.60,w:.4,h:.30},{id:'second',u:.55,v:.60,w:.4,h:.30,disabled:true}];shell.setActions(actions);
     root.updateMatrixWorld(true);
     function hit(u,v){const p=G.point(u,v,1,.5,1.6,.012);return shell.hit(new T.Raycaster(new T.Vector3(p[0],p[1],1),new T.Vector3(0,0,-1)));}
     return {first:hit(.25,.75)?.id,disabled:hit(.75,.75)?.id,blank:hit(.50,.75)?.id,raised:hit(.25,.75)?.hit.point.z-G.point(.25,.75,1,.5,1.6,0)[2],evidence:shell.snapshot()};
   });
   assert.equal(spatialPhysical.first,'first');assert.equal(spatialPhysical.disabled,null);assert.equal(spatialPhysical.blank,null);assert.ok(spatialPhysical.raised>.008);assert.equal(spatialPhysical.evidence.controls,2);
   console.log('PASS UI02 curved physical control ray hits, disabled keys and substrate separation');
   const labEvidence=await legacy.evaluate(()=>{const T=window.THREE,G=window.PulmoXRSpatialGeometry,r=new T.Group();const lab=G.createLaboratory(T,r);return {objects:lab.root.children.length,geometry:lab.root.children.every(o=>o.isMesh),evidence:lab.snapshot()}});
   assert.ok(labEvidence.objects>=15&&labEvidence.geometry);assert.equal(labEvidence.evidence.physicalRoomReconstructed,false);
   console.log('PASS UI03 authored laboratory geometry and explicit virtual-only evidence');
   const ui04=await legacy.evaluate(()=>{const U=window.PulmoXRSpatialExperience,T=window.THREE,G=window.PulmoXRSpatialGeometry;const root=new T.Mesh(new T.PlaneGeometry(1,.5),new T.MeshBasicMaterial());const texture=new T.CanvasTexture(document.createElement('canvas'));const s=G.create(T,root,{width:1,height:.5,texture});const action={id:'focus',u:.1,v:.6,w:.3,h:.2};s.setActions([action]);const p=G.point(.25,.7,1,.5,1.8,.012);const ray=new T.Raycaster(new T.Vector3(p[0],p[1],1),new T.Vector3(0,0,-1));const before=s.hit(ray).hit.point.z;s.press('focus',100);s.update([],185);const after=s.hit(ray).hit.point.z;return {delta:before-after,id:s.hit(ray).id,anatomy:U.layoutRail('anatomy').map(a=>a.id),procedure:U.layoutRail('procedure').map(a=>a.id),near:U.readability({heightM:.48,canvasHeight:600,fontPx:21,distanceM:.7}),far:U.readability({heightM:.48,canvasHeight:600,fontPx:21,distanceM:2})};});
   assert.ok(ui04.delta>.006&&ui04.id==='focus',JSON.stringify(ui04));assert.ok(ui04.procedure.includes('dock-channel')&&!ui04.anatomy.includes('dock-channel'));assert.ok(ui04.near.estimatedGlyphArcmin>ui04.far.estimatedGlyphArcmin);assert.equal(ui04.near.physicalReadabilityVerified,false);
   console.log('PASS UI04 context rail, physical key travel and angular size evidence');
   const ui05=await legacy.evaluate(()=>{const T=window.THREE,G=window.PulmoXRSpatialGeometry,U=window.PulmoXRSpatialUI;
     const r=new T.Group(),lab=G.createLaboratory(T,r),mon=new T.Group(),housing=G.monitorHousing(T,mon);housing.setStatus('live');
     const canvas=document.createElement('canvas');canvas.width=100;canvas.height=100;const ctx=canvas.getContext('2d');const icon=U.symbol(ctx,'scope',10,10,64,'#fff');const pixels=ctx.getImageData(0,0,100,100).data;let alpha=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])alpha++;
     const indicator=mon.getObjectByName('monitor-source-status');return {icon,alpha,instanced:lab.root.children.filter(o=>o.isInstancedMesh).length,ventCount:mon.getObjectByName('rear-vent-array').count,indicator:indicator.material.color.getHex(),beveled:housing.beveledChassis,source:housing.videoUnchanged,environment:lab.snapshot(),noLights:lab.root.children.every(o=>!o.isLight)};
   });
   assert.ok(ui05.icon&&ui05.alpha>100&&ui05.instanced>=3&&ui05.ventCount===24&&ui05.beveled&&ui05.source&&ui05.noLights);assert.equal(ui05.indicator,0xa9d6cd);assert.equal(ui05.environment.additionalSceneLights,0);
   console.log('PASS UI05 vector icon pixels, instanced architectural details and medical monitor status');
   const ui06=await legacy.evaluate(()=>{const U=window.PulmoXRSpatialExperience;
     const blocked={right:0,y:0,depth:1,scale:1},clear={right:1.15,y:.1,depth:1.3,scale:1};
     const protectedEnvelope=U.envelope([[-.25,-.3,.8],[.25,.3,.8]]);
     const result=U.compose({protectedEnvelope,items:[{id:'console',preferred:blocked,alternatives:[clear],width:.6,height:.4}]});
     const baseline=U.overlap(U.panelEnvelope(blocked,.6,.4),protectedEnvelope);
     const chosen=U.overlap(U.panelEnvelope(result.poses.console,.6,.4),protectedEnvelope);
     return {baseline,chosen,evidence:result};
   });
   assert.ok(ui06.baseline>0&&ui06.chosen<ui06.baseline);assert.equal(ui06.evidence.poses.console.right,1.15);assert.equal(ui06.evidence.physicalOcclusionVerified,false);
   console.log('PASS UI06 source-protected angular composition and honest visibility evidence');
   console.log('PASS original surgical application: expected canvases and page preserved');
 } finally {await legacy.close();}

} finally {
 await browser.close();
}
