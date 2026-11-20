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
 assert.ok(await page.locator('#lungCanvas').isVisible(),'3D canvas visible');
 const canvas=await page.locator('#lungCanvas').boundingBox();
 assert.ok(canvas?.width>250&&canvas.height>250,'Canvas has nonzero stage dimensions');
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
 assert.match(await page.locator('#isolateRegion').innerText(),/Show all lobes/);
 await page.locator('#toggleAirways').click();
 assert.match(await page.locator('#toggleAirways').innerText(),/Show airway/);
 await page.locator('#resetCamera').click();
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
 assert.match(await page.locator('#anatomyProvenance').innerText(),/PROCEDURAL REFERENCE/);
 assert.ok(await page.locator('#anatomicalAtlas').isVisible(),'reference topology restored');
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
 assert.match(await page.locator('#anatomyProvenance').innerText(),/PROCEDURAL REFERENCE/);
 assert.ok(await page.locator('#anatomicalAtlas').isVisible());
 await page.screenshot({path:'/tmp/pulmosimu-experimental-smoke.png',fullPage:true});
 assert.deepEqual(errors,[],'No uncaught browser errors');
 console.log('PASS experimental canvas: airway load, numerical results, disease, R5, overlays and controls');
 const xr=await browser.newPage({viewport:{width:1260,height:850}});
 const xrErrors=[];
 xr.on('pageerror',err=>xrErrors.push(err.message));
 try{
  await xr.goto(host+'/experimental/vr.html',{waitUntil:'domcontentloaded'});
  await xr.waitForFunction(()=>document.getElementById('status')?.textContent?.includes('SOURCE BRANCHES'),null,{timeout:45000});
  assert.ok(await xr.locator('#stageCanvas').isVisible());
  assert.equal(await xr.locator('#selectedId').innerText(),'RUL');
  const count=Number(await xr.locator('#branchCount').innerText());
  assert.ok(count>0,'Canonical reference airway tree loaded');
  assert.ok(await xr.locator('#enterVR button').count()>0,'WebXR button or supported fallback present');
  await xr.locator('#resetView').click();
  assert.equal(await xr.locator('#selectedId').innerText(),'RUL');
  assert.deepEqual(xrErrors,[],'No WebXR viewer script errors');
  console.log('PASS Quest viewer desktop fallback: source branches, selection state, reset and XR capability UI');
 }finally{await xr.close()}
 const studio=await browser.newPage({viewport:{width:1440,height:900}});
 studio.on('pageerror',error=>console.error('STUDIO PAGE ERROR',error.message));
 try{
   await studio.goto(host+'/studio.html',{waitUntil:'domcontentloaded'});
   assert.equal(await studio.locator('#workspaceTitle').innerText(),'Procedural simulation');
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
   await studio.locator('#modeProcedure').click();
   assert.equal(await studio.locator('#workspaceTitle').innerText(),'Procedural simulation');
   assert.equal(await studio.locator('#modeProcedure').getAttribute('aria-pressed'),'true');
   assert.equal(await studio.locator('#workspaceFrame').getAttribute('src'),'./procedural-workspace.html');
   console.log('PASS studio: procedural and mechanistic workspaces remain independent and reachable');
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
   console.log('PASS original surgical application: expected canvases and page preserved');
 } finally {await legacy.close();}

} finally {
 await browser.close();
}
