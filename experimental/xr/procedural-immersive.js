/* PulmoSIMU · Procedural Immersive XR
 * Three.js r147 projection-layer-capable classic-script adapter for the ORIGINAL PNEUMO·VR scene.
 * The original mGrp, lobar geometry, pathology and scope functions remain the
 * sole source of truth. No second anatomy, procedural state, or tissue solver.
 * Physical scale is a COMFORT DISPLAY scale, not calibrated patient anatomy.
 */
(function(global){
'use strict';
function install(ctx){
  const {THREE,scene,rBro,rExt,camScope,mGrp,vrCam,scope,CFG,
    SEGS,SEG_MAP,pathCurves,patientData,getMeshMode,getAnatomyMeshes,
    getPatientAudit,getCTStudy,getMmPerSceneUnit,getScopeDiameterMm,getReferenceFocus,
    setClinMode,getClinMode,fireAction,enterBranch:sourceEnterBranch,advance:sourceAdvance,retract:sourceRetract,hideBranchSel,
    updateScope,onResize,showToast,getExtIdle,setExtIdle}=ctx;
  if(!THREE||!scene||!rBro||!mGrp||!scope)throw Error('Procedural XR bridge missing original clinical engine');
  if(!global.PulmoXRSessionHealth?.createSessionHealth)
    throw Error('XR-09 local session health runtime missing');
  if(!global.PulmoXRRemoteTarget?.pick)
    throw Error('XR-10 triangle-backed remote aiming runtime missing');
  if(!global.PulmoXRInteractionMath?.clampScale)
    throw Error('XR-11 world-space manipulation limits missing');
  if(!global.PulmoXRSpatialTarget?.choose)
    throw Error('XR-12 deterministic menu/anatomy input router missing');
  if(!global.PulmoXRStagePresence?.create)
    throw Error('XR-13 spatial architecture runtime missing');
  if(!global.PulmoXRProcedureGuide?.describe)
    throw Error('XR-14 airway guidance runtime missing');
  if(!global.PulmoXRViewHistory?.create)
    throw Error('XR-15 reversible anatomy transform runtime missing');
  if(!global.PulmoXRBronchoscopyStation?.create)
    throw Error('XR-16 procedure station and instrument kinematics unavailable');
  if(!global.PulmoXRCapabilities?.create)
    throw Error('XR-17A capability-negotiated XR runtime unavailable');
  if(!global.PulmoXRMRPlacement?.create)
    throw Error('XR-17B spatial MR placement runtime unavailable');
  if(!global.PulmoXRInstrumentReplay?.create||!global.PulmoXRInstrumentBench?.assess)
    throw Error('XR-17C read-only measured instrument review unavailable');
  if(!global.PulmoXRDisplayPath?.create)
    throw Error('XR-17C display compositor assessment unavailable');
  if(!global.PulmoXRTrackedToolProxy?.create)
    throw Error('XR-17C right-controller tracked visual instrument proxy unavailable');
  const limits=global.PulmoXRInteractionMath;
  const health=global.PulmoXRSessionHealth.createSessionHealth();
  const station=global.PulmoXRBronchoscopyStation.create();
  const xrCapabilities=global.PulmoXRCapabilities.create({xr:navigator.xr||null});
  const mrPlacement=global.PulmoXRMRPlacement.create();
  const replay=global.PulmoXRInstrumentReplay.create({
    assess:global.PulmoXRInstrumentBench.assess
  });
  const displayPath=global.PulmoXRDisplayPath.create({
    bindingConstructor:global.XRWebGLBinding||null
  });
  const nativeQuad=global.PulmoXRNativeQuad?.create();
  const toolProxy=global.PulmoXRTrackedToolProxy.create();
  const spatialCal=global.PulmoXRSpatialCalibration.create(),spatialVisual=global.PulmoXRSpatialCalibration.createVisual(THREE,scene),planeMask=global.PulmoXRSpatialCalibration.createOccluder(THREE,scene);
  let calibrationPose=null,calibrationCaptureAt=-Infinity,calibrationSpace=null,benchAligned=false,planeMaskEnabled=false,showPassiveProxy=false,spatialFinal=null;
  function clearSpatialCalibration(){calibrationSpace?.removeEventListener?.('reset',onCalibrationSpaceReset);spatialCal.reset();spatialVisual.hide();planeMask.clear();calibrationPose=null;calibrationSpace=null;benchAligned=false;showPassiveProxy=false;}
  function onCalibrationSpaceReset(){if(benchAligned){invalidateBench();mGrp.visible=false;placed=false;}clearSpatialCalibration();}
  function observeCalibration(frame){const space=rBro.xr.getReferenceSpace();if(calibrationSpace&&space!==calibrationSpace){if(benchAligned){invalidateBench();mGrp.visible=false;placed=false;}clearSpatialCalibration();}if(space&&space!==calibrationSpace){calibrationSpace=space;space.addEventListener?.('reset',onCalibrationSpaceReset);}const source=Array.from(rBro.xr.getSession()?.inputSources||[]).find(s=>s.handedness==='right'&&!s.hand&&s.gripSpace);const p=source&&frame?.getPose(source.gripSpace,space)?.transform;calibrationPose=p&&[p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w].every(Number.isFinite)?{position:[p.position.x,p.position.y,p.position.z],orientation:[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]}:null;spatialVisual.update(calibrationPose,spatialCal,showPassiveProxy||menuPage==='spatial-cal'||benchAligned);planeMask.step(frame,space,xrPresentation==='ar'&&planeMaskEnabled);}
  function alignBench(){const j=spatialCal.jig();if(!j||xrPresentation!=='ar'||mode!=='anatomy'||getMeshMode()!=='reference'||!viewerValid){grabFeedback='MR REFERENCE ANATOMY AND VERIFIED JIG LANDMARKS REQUIRED';return false;}release(0);release(1);if(inspectMode)setInspection(false);summonGeneration++;summonTween=null;mrPlacement.preview();mrAnchorFault=false;const bounds=sourceBounds(),scale=getMmPerSceneUnit()/1000;if(!Number.isFinite(scale)||scale<=0||scale>.5)return false;boundsCache=bounds;const q=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(...j.basis.map(v=>new THREE.Vector3(...v))));mGrp.quaternion.copy(q);mGrp.scale.setScalar(scale);const center=new THREE.Vector3(...j.origin).add(new THREE.Vector3(.10,bounds.size.y*scale*.5+.025,.075).applyQuaternion(q));mGrp.position.copy(center).sub(bounds.center.clone().multiplyScalar(scale).applyQuaternion(q));mGrp.visible=true;mGrp.updateMatrixWorld(true);placed=true;benchAligned=true;scaleMode='manual-reference-jig';positionPanels();grabFeedback='MANUAL JIG ALIGNMENT / REFERENCE ONLY';return true;}
  function invalidateBench(){if(!benchAligned)return;benchAligned=false;scaleMode='unregistered-preview';if(xrPresentation==='ar')mrPlacement.preview();}

  const videoScope=global.PulmoXRVideoScope.createVisual(THREE);
  const deviceLab=global.PulmoXRDeviceLab;
  const spatialUI=global.PulmoXRSpatialUI;
  const spatialGeometry=global.PulmoXRSpatialGeometry;
  const spatialExperience=global.PulmoXRSpatialExperience;
  let researchSnapshot=null,researchCanvas=null,researchLoadEpoch=0;
  const reviewEngine=global.PulmoXRSessionReview,localRecording=reviewEngine.create();let reviewPacket=null,reviewIndex=0,reviewPlaying=false,reviewStart=0,reviewBase=0,reviewCanvas=null,reviewEpoch=0,localShare=null,localShareCode=null,shareLast=-Infinity;
  function stopLocalShare(){try{localShare?.postMessage({type:'ended',source:reviewEngine.source});localShare?.close();}catch{}localShare=null;localShareCode=null;}
  function beginLocalShare(){if(!localRecording.snapshot().running||typeof BroadcastChannel!=='function'){grabFeedback='START RECORDING / LOCAL OBSERVER SUPPORT REQUIRED';return;}stopLocalShare();try{localShareCode=Array.from(crypto.getRandomValues(new Uint8Array(6)),x=>x.toString(16).padStart(2,'0')).join('');localShare=new BroadcastChannel('pulmosimu-review-'+localShareCode);shareLast=-Infinity;grabFeedback='LOCAL OBSERVER CODE '+localShareCode;}catch{stopLocalShare();grabFeedback='LOCAL OBSERVER UNAVAILABLE';}}
  function stopRecording(why='STOPPED'){localRecording.stop(why);stopLocalShare();}
  function observeRecording(frame,time){if(!localRecording.snapshot().running)return;if(!viewerValid||mode!=='procedure'||getMeshMode()!=='reference'||xrPresentation!=='vr'){localRecording.suspend();return;}const session=rBro.xr.getSession(),space=rBro.xr.getReferenceSpace(),R=new THREE.Vector3().crossVectors(forward,new THREE.Vector3(0,1,0)).normalize();const hand=side=>{const sample=handSamples[side]?.points?.wrist;let p=sample?new THREE.Vector3(...sample):null,kind='HAND_WRIST',grip=side==='right'?handHeld:handGrasped[0];if(!p){const src=Array.from(session?.inputSources||[]).find(s=>s.handedness===side&&!s.hand&&s.gripSpace),pos=src&&frame?.getPose(src.gripSpace,space)?.transform?.position;if(!pos)return null;p=new THREE.Vector3(pos.x,pos.y,pos.z);kind='CONTROLLER_GRIP';grip=!!src.gamepad?.buttons?.[1]?.pressed;}const d=p.sub(head);return {kind,relative_m:[d.dot(R),d.y,-d.dot(forward)],grip:!!grip};};const eb=ebusProbe.snapshot(),row=localRecording.observe(time,{segment:scope.segId,t:scope.t,tip_mm:scope.pos.toArray().map(v=>v*getMmPerSceneUnit()),roll_rad:((scope.roll+Math.PI)%(2*Math.PI)+2*Math.PI)%(2*Math.PI)-Math.PI,deflection:[scope.deflH,scope.deflV],left:hand('left'),right:hand('right'),risk:hapticEvidence.risk??null,ebus:eb.mode,probe_mm:eb.extensionMm});if(row&&localShare&&time-shareLast>=500){shareLast=time;try{localShare.postMessage({type:'sample',source:reviewEngine.source,sample:row});}catch{stopLocalShare();}}if(!localRecording.snapshot().running){stopLocalShare();const packet=localRecording.packet();if(packet)loadRecordedReview(packet);}}
  function loadRecordedReview(packet){reviewPacket=packet;reviewIndex=0;reviewPlaying=false;if(!reviewCanvas){reviewCanvas=document.createElement('canvas');reviewCanvas.width=440;reviewCanvas.height=164;}if(packet)reviewEngine.draw(reviewCanvas,packet.samples,0);}
  function stepReview(delta){if(!reviewPacket)return;reviewPlaying=false;reviewIndex=Math.max(0,Math.min(reviewPacket.samples.length-1,reviewIndex+delta));reviewEngine.draw(reviewCanvas,reviewPacket.samples,reviewIndex);}
  function tickReview(time){if(!reviewPlaying||!reviewPacket)return;const rows=reviewPacket.samples,target=reviewBase+time-reviewStart;let i=reviewIndex;while(i+1<rows.length&&rows[i+1].t_ms<=target)i++;if(i!==reviewIndex){reviewIndex=i;reviewEngine.draw(reviewCanvas,rows,i);}if(i===rows.length-1)reviewPlaying=false;}

  let readingMode=false,monitorFocus=false,visualEvidence={},visualLast=-Infinity,contextActions=0;
  let compositionEvidence=null,compositionCount=0;
  const upgrade=global.PulmoXRIntegratedUpgrade;
  const mrQuality=upgrade.createTrackingQuality(),couplingEvidence=upgrade.createCouplingEvidence();
  let inspectionFilter='all',inspectionVisibilitySaved=null,workspaceMoves=0,workspaceRelayout=false,accessoryEvidence={state:'UNAVAILABLE'};
  const handGrasped=[false,false];
  let panelEdit=null,panelMoves=0,panelTarget='monitor',environmentEvidence=upgrade.observeEnvironment(null),environmentLast=-Infinity,regionalDisplay=false,regionalOverlay=null,regionalLast=-Infinity,regionalEvidenceFinal=null,contactProxy={state:'UNAVAILABLE'};
  const panelPinned=new Set();
  const regionalEvidence=upgrade.createRegionalEvidence();
  let section=null,sectionAxis='x',sectionFraction=.5,sectionSaved=[],sectionClipWas=false,panelOffset=0,accessoryLast=-Infinity,mrAnchorFault=false;
  const respiratoryMotion=global.PulmoXRRespiratoryMotion.create();
  let motionEnabled=true,motionEvidence=respiratoryMotion.reset(),motionReport=null;
  const ebusProbe=global.PulmoXREBUS.createProbe(),ebusVisual=global.PulmoXREBUS.createVisual(THREE,scene),probeGrip=global.PulmoXREmbodiment.createGrip();
  let radialEnabled=false,radialCoupling='air',radialLast=-Infinity,radialEvidence=null,radialRange=20;
  const registrationObserver=global.PulmoXRRegistration.createObserver();
  let registrationEvidence=registrationObserver.snapshot(),contactEvidence=global.PulmoXRContactInterface.evaluate();
  const procedureMetrics=global.PulmoXRProcedureMetrics.create({sampleHz:20});
  procedureMetrics.reset(SEGS);
  const insertedShaft=deviceLab.createShaft(THREE,mGrp);
  const channel=deviceLab.createChannel();
  const accessory=deviceLab.createAccessory(THREE,scene);
  let opticsEnabled=true,shaftEvidence={state:'UNAVAILABLE'};
  const shaftClutch=global.PulmoXRVideoScope.createClutch();
  let deviceVisible=true,clutchAxis=null,clutchHeld=false;
  scene.add(videoScope.group);
  let mrReticle=null,originalModelVisibility=true,mrMenuPositioned=false;
  const instrument=global.PulmoXRBronchoscopyStation;
  const originalScopeFov=camScope.fov;
  // External instrument inputs are evidence only, never scope control.
  const physicalBench=global.PulmoXRInstrumentTelemetry?.create?.()||null;
  const physicalCalibration=global.PulmoXRPhysicalCalibration?.create({parseFrame:global.PulmoXRInstrumentTelemetry.parseFrame});
  const physicalControl=physicalCalibration&&global.PulmoXRCalibratedControl?.create({calibration:physicalCalibration});
  let serialInput=null;
  const hapticLab=global.PulmoXRHapticLab,cueEngine=hapticLab.createCue(),sensorObserver=hapticLab.createObserver(global.PulmoXRInstrumentTelemetry.parseFrame),hapticDryRun=hapticLab.createDryRun();
  let hapticEvidence=hapticLab.estimate(),lastAdvanceAt=-Infinity,lastAdvanceBlocked=false,lastShaftAt=-Infinity,lastShaftSample=null,lastShaftKey=null,hapticFinal=null;
  function resetHapticContext(){cueEngine.reset();hapticDryRun.stop('CONTEXT_RESET');hapticEvidence=hapticLab.estimate();lastAdvanceAt=-Infinity;lastAdvanceBlocked=false;lastShaftAt=-Infinity;lastShaftSample=null;lastShaftKey=null;}
  function trackedLeftGrip(frame){const s=Array.from(rBro.xr.getSession()?.inputSources||[]).find(x=>x.handedness==='left'&&!x.hand&&x.gripSpace);return !!(s?.gamepad?.buttons?.[1]?.pressed&&frame?.getPose(s.gripSpace,rBro.xr.getReferenceSpace()));}
  function updateHapticLab(frame,time){const now=performance.now();const key=[scope.segId,scope.t,getScopeDiameterMm(),getMmPerSceneUnit()].join('/');if(shaftEvidence!==lastShaftSample){lastShaftSample=shaftEvidence;lastShaftAt=now;lastShaftKey=key;}const unchanged=lastShaftKey===key;const advancing=now-lastAdvanceAt<180;hapticEvidence=hapticLab.estimate({shaft:shaftEvidence,reference:getMeshMode()==='reference',tracking:viewerValid,procedure:mode==='procedure',marginAgeMs:unchanged&&Number.isFinite(lastShaftAt)?0:now-lastShaftAt,advancing,blocked:lastAdvanceBlocked});
   const session=rBro.xr.getSession(),left=Array.from(session?.inputSources||[]).find(x=>x.handedness==='left'&&!x.hand&&x.gripSpace),i=controllerSources.indexOf(left),grip=trackedLeftGrip(frame),allowed=active&&viewerValid&&mode==='procedure'&&xrPresentation==='vr'&&patientMotionAllowed()&&!probeLocked()&&!accessoryDeployed()&&!panelEdit;
   const cue=cueEngine.step({now,evidence:hapticEvidence,activeGrip:grip&&i>=0,allowed});if(cue)haptic(i,cue.intensity,cue.durationMs);
   const measured=sensorObserver.snapshot(now),cal=physicalCalibration?.snapshot(now),connected=serialInput?.status().state==='ACQUIRING';hapticDryRun.step({now,reference:getMeshMode()==='reference',tracking:viewerValid,procedure:mode==='procedure',connected,calibrated:!!calibrationLoaded&&!!physicalControl?.snapshot().armed,fresh:!!measured.motion&&!!cal?.corrected,deadman:grip,eligible:allowed&&ebusProbe.snapshot().mode==='off',evidence:hapticEvidence});
  }

  rBro.xr.enabled=true;
  if(rBro.xr.setFramebufferScaleFactor)rBro.xr.setFramebufferScaleFactor(.78);
  // Meta Quest Browser may expose fixed foveation on its XR base layer.
  // Do not modify resolution every frame or claim proven display FPS.
  function configureComfortRendering(){
    const layer=rBro.xr.getSession()?.renderState?.baseLayer;
    try{
      if(layer&&'fixedFoveation' in layer){
        layer.fixedFoveation=.35;
        return 'fixed-foveation requested 0.35; headset performance unverified';
      }
      if(typeof rBro.xr.setFoveation==='function'){
        rBro.xr.setFoveation(.35);
        return 'renderer foveation requested 0.35; headset performance unverified';
      }
      return 'fixed-foveation unsupported; frame-buffer scale 0.78';
    }catch(error){
      return 'fixed-foveation unavailable; '+String(error?.name||'browser restriction').slice(0,45);
    }
  }

  const safe=(a,b,v)=>Math.max(a,Math.min(b,v));
  const vec=(a)=>new THREE.Vector3(a[0],a[1],a[2]);
  const hands=[null,null],grips=[null,null],grab=[null,null],controllerSide=[null,null];
  // WebXR getController(index) is connection-order, NOT reliably left=0/right=1.
  const controllerSources=[null,null],laserLines=[null,null],laserTips=[null,null];
  const aimTargets=[null,null],squeezeHeld=[false,false],hoverPanels=[null,null];
  const attempts={acceptedRay:0,acceptedNear:0,rejected:0,trigger:0,grip:0,movedFrames:0,menuActions:0,thumbstickFrames:0,panelToggles:0,hapticRequests:0};
  let lastAimTime=0,grabFeedback='AIM AT LUNG / HOLD TRIGGER OR GRIP';
  let summonTween=null;
  const handProxies=[new THREE.Group(),new THREE.Group()];
  const handSeen=[false,false],handPinched=[false,false],grabSource=[null,null];
  // Render only actually tracked WebXR joints. No avatar or fabricated poses.
  const embodiment=global.PulmoXREmbodiment.createVisual(THREE,scene);
  let bodyEnabled=true,handProcedure=false,handSamples={},handMenuBlocked={left:false,right:false},parkedHandle=null,rightOrigin=null,rightDeflection=null,handHeld=false;
  const rightGrip=global.PulmoXREmbodiment.createGrip(),leftGrip=global.PulmoXREmbodiment.createGrip();
  function resetEmbodiedGrip(){rightGrip.reset();leftGrip.reset();probeGrip.reset();rightOrigin=null;rightDeflection=null;handHeld=false;parkedHandle=null;}
  function updateHandVisual(i,source,frame,space){
    const points={},radii={};let orientation=null;
    for(const n of global.PulmoXREmbodiment.names){const joint=source.hand.get(n),p=joint&&frame.getJointPose(joint,space),t=p?.transform;if(t?.position&&[t.position.x,t.position.y,t.position.z].every(Number.isFinite)){points[n]=[t.position.x,t.position.y,t.position.z];if(Number.isFinite(p.radius))radii[n]=p.radius;if(n==='wrist'&&t.orientation)orientation=[t.orientation.x,t.orientation.y,t.orientation.z,t.orientation.w];}}
    if(['left','right'].includes(source.handedness))handSamples[source.handedness]={points,radii,orientation};
  }
  function updateEmbodiedBody(frame){
    const session=rBro.xr.getSession(),space=rBro.xr.getReferenceSpace(),controllers={};
    for(const s of session?.inputSources||[]){if(!s.hand&&s.gripSpace){const p=frame?.getPose(s.gripSpace,space)?.transform?.position;if(p)controllers[s.handedness]=[p.x,p.y,p.z];}}
    embodiment.update({head:head.toArray(),right:right.toArray(),forward:forward.toArray(),floor:room.position.y,showBody:bodyEnabled&&xrPresentation==='vr',showHands:xrPresentation==='vr',samples:handSamples,controllers});
  }
  function updateBareHandScope(dt){
    const r=handSamples.right,l=handSamples.left,session=rBro.xr.getSession(),hasController=Array.from(session?.inputSources||[]).some(s=>!s.hand&&s.gamepad);
    const valid=patientMotionAllowed()&&handProcedure&&mode==='procedure'&&xrPresentation==='vr'&&viewerValid&&getMeshMode()==='reference'&&!hasController&&!physicalControl?.snapshot().armed&&!accessoryDeployed()&&!panelEdit;
    if(!valid){resetEmbodiedGrip();return false;}
    const pinch=s=>{const a=s?.points['index-finger-tip'],b=s?.points['thumb-tip'];return a&&b?Math.hypot(...a.map((x,i)=>x-b[i])):Infinity;};
    const center=s=>{const a=s?.points['index-finger-tip'],b=s?.points['thumb-tip'];return a&&b?a.map((x,i)=>(x+b[i])/2):null;};
    const rp=center(r),lp=center(l),ori=r?.orientation,oldHeld=handHeld;
    if(!parkedHandle){parkedHandle=patientReturn?patientStation.mouthWorld().lerp(head,.46).add(new THREE.Vector3(0,-.09,0)):head.clone().addScaledVector(forward,.45).addScaledVector(right,.20).add(new THREE.Vector3(0,-.38,0));videoScope.group.position.copy(parkedHandle);videoScope.group.quaternion.identity();}
    const rClosed=pinch(r)<(handHeld?.045:.027);
    const rg=rightGrip.step({valid:Boolean(rp&&ori?.every(Number.isFinite)&&!handMenuBlocked.right),closed:rClosed,near:Boolean(rp&&new THREE.Vector3(...rp).distanceTo(videoScope.group.position)<.095),point:rp,dt});
    handHeld=rg.held;scope.velocity=0;
    if(handHeld){
     const q=new THREE.Quaternion(...ori).normalize();
     if(!oldHeld){rightOrigin=q.clone();rightDeflection={h:scope.deflH,v:scope.deflV,roll:scope.roll};}
     videoScope.group.position.set(...rp);videoScope.group.quaternion.copy(q);
     if(rightOrigin&&rightDeflection&&!probeLocked()){const e=new THREE.Euler().setFromQuaternion(rightOrigin.clone().invert().multiply(q),'YXZ');scope.deflH=instrument.rateLimit(scope.deflH,safe(-1,1,rightDeflection.h+e.y/.65),dt);scope.deflV=instrument.rateLimit(scope.deflV,safe(-1,1,rightDeflection.v-e.x/.65),dt);}
     const guide=new THREE.Vector3(0,-.20,0).applyQuaternion(q).add(videoScope.group.position),axis=new THREE.Vector3(0,-1,0).applyQuaternion(q);
     const relative=lp?new THREE.Vector3(...lp).sub(videoScope.group.position).applyQuaternion(q.clone().invert()).toArray():null;
     if(ebusProbe.snapshot().mode==='radial'){leftGrip.reset();driveProbe(lp,pinch(l)<.035,!handMenuBlocked.left&&Boolean(lp),dt,q);}
     const lg=leftGrip.step({valid:Boolean(lp&&!handMenuBlocked.left&&ebusProbe.snapshot().mode!=='radial'),closed:pinch(l)<.035,near:Boolean(lp&&new THREE.Vector3(...lp).distanceTo(guide)<.075),point:relative,axis:[0,-1,0],dt});
     const length=pathCurves[scope.segId]?.getLength()*getMmPerSceneUnit();if(lg.held&&Number.isFinite(length)&&length>0){if(lg.deltaMm>0)advance(lg.deltaMm/length);else if(lg.deltaMm<0)retract(-lg.deltaMm/length);}
    }else{leftGrip.reset();probeGrip.reset();rightOrigin=null;rightDeflection=null;}
    videoScope.group.visible=deviceVisible;videoScope.update({h:scope.deflH,v:scope.deflV,roll:scope.roll,diameterMm:getScopeDiameterMm(),connected:true});return true;
  }
  // A slender, opt-in reference marker attached to the *actual* right
  // WebXR controller grip. It is not a physical bronchoscope model,
  // calibration, haptic device or anatomical collision body.
  const toolProxyOverlay=new THREE.Group();
  toolProxyOverlay.name='xr17c-6dof-controller-only-reference-marker';
  const markerMaterial=new THREE.MeshBasicMaterial({
    color:0x99d7df,transparent:true,opacity:.67,depthWrite:false});
  const referenceShaft=new THREE.Mesh(new THREE.CylinderGeometry(
    .0035,.0035,.15,12),markerMaterial);
  referenceShaft.rotation.x=Math.PI/2;
  referenceShaft.position.z=-.075;
  toolProxyOverlay.add(referenceShaft);
  const referenceTip=new THREE.Mesh(new THREE.SphereGeometry(.008,12,8),
    new THREE.MeshBasicMaterial({color:0xe5e1cc,depthWrite:false}));
  referenceTip.position.z=-.15;toolProxyOverlay.add(referenceTip);
  toolProxyOverlay.visible=false;scene.add(toolProxyOverlay);
  function updateVideoScope(frame,dt){
    if(updateBareHandScope(dt))return;
    const session=rBro.xr.getSession(),space=rBro.xr.getReferenceSpace();
    const rightSource=Array.from(session?.inputSources||[]).find(s=>s.handedness==='right'&&s.gripSpace&&!s.hand);
    const leftSource=Array.from(session?.inputSources||[]).find(s=>s.handedness==='left'&&s.gripSpace&&!s.hand);
    const rp=rightSource&&frame?.getPose(rightSource.gripSpace,space)?.transform;
    const lp=leftSource&&frame?.getPose(leftSource.gripSpace,space)?.transform;
    const valid=mode==='procedure'&&viewerValid&&Boolean(rp);
    videoScope.group.visible=valid&&deviceVisible;
    if(valid){
      videoScope.group.position.set(rp.position.x,rp.position.y,rp.position.z);
      videoScope.group.quaternion.set(rp.orientation.x,rp.orientation.y,rp.orientation.z,rp.orientation.w);
      videoScope.update({h:scope.deflH,v:scope.deflV,roll:scope.roll,diameterMm:getScopeDiameterMm(),connected:true});
    }
    if(ebusProbe.snapshot().mode==='radial'){
      shaftClutch.reset();clutchAxis=null;clutchHeld=false;scope.velocity=0;
      if(valid&&patientMotionAllowed()&&lp&&!physicalControl?.snapshot().armed&&!panelEdit)driveProbe([lp.position.x,lp.position.y,lp.position.z],Boolean(leftSource?.gamepad?.buttons?.[1]?.pressed),true,dt,videoScope.group.quaternion);else probeGrip.reset();return;
    }
    const held=valid&&patientMotionAllowed()&&Boolean(lp)&&Boolean(leftSource?.gamepad?.buttons?.[1]?.pressed)&&!physicalControl?.snapshot().armed&&!accessoryDeployed()&&!panelEdit;
    if(!held){shaftClutch.reset();clutchAxis=null;clutchHeld=false;return;}
    const rel=new THREE.Vector3(lp.position.x-rp.position.x,lp.position.y-rp.position.y,lp.position.z-rp.position.z);
    if(!clutchAxis){if(rel.length()<.12){scope.velocity=0;return;}clutchAxis=rel.clone().normalize().toArray();}
    clutchHeld=true;scope.velocity=0;
    const mm=shaftClutch.step({held,tracking:true,position:rel.toArray(),axis:clutchAxis,dt});
    const length=pathCurves[scope.segId]?.getLength()*getMmPerSceneUnit();
    if(Number.isFinite(length)&&length>0){if(mm>0)advance(mm/length);else if(mm<0)retract(-mm/length);}
  }
  function updateToolProxy(){
    const rightIndex=controllerSide.findIndex((v,i)=>
      (v==='right'||controllerSources[i]?.handedness==='right')&&
      Boolean(controllerSources[i]?.gripSpace));
    const valid=toolProxy.observe({
      rightGripTracked:rightIndex>=0,procedure:mode==='procedure',
      immersiveVR:xrPresentation==='vr'
    });
    if(valid&&rightIndex>=0){
      const target=grips[rightIndex];
      if(toolProxyOverlay.parent!==target)target.add(toolProxyOverlay);
    }
    toolProxyOverlay.visible=valid;
  }
  const contactMarkers=[];
  let mode='anatomy',active=false,placed=false,viewerValid=false,lastFrame=0;
  let scaleMode='comfort',lastPanelUpdate=0,physicsSnapshot=null;
  let physicsError=null;
  let saved=null,scopeSteering=null,endedXRGeometry=null;
  let gesture=null,worldMenu=null,scopeScreen=null,scopeTexture=null,scopeCanvas=null;
  let button=null,lastFeed=0,actionHitbox=[],summonGeneration=0;
  let menuPage='primary',qualityStatus='unassessed';
  let guidePlane=null,guideTex=null,guideCtx=null,guideHitboxes=[];
  let guideVisible=true,guideStep=0,guideLastState=null,guideActions=0;
  let xrPresentation='vr',backgroundBeforeXR=null,fogBeforeXR=null;
  // Physical acceptance prioritizes a reliable monitor; native layers remain
  // opt-in for a separate measured compositor session.
  const nativeLayersOptIn=false;
  let clearBeforeXR=null,clearAlphaBeforeXR=1,arButton=null,mrHidden=[];
  const tempMatrix=new THREE.Matrix4(),ray=new THREE.Raycaster();
  const modelBox=new THREE.Box3(),head=new THREE.Vector3(),forward=new THREE.Vector3(),right=new THREE.Vector3();
  const tmp=new THREE.Vector3(),tmp2=new THREE.Vector3(),tempBox=new THREE.Box3();
  let boundsCache=null,limitsApplied=0,scaleClamps=0,workspaceClamps=0;
  const viewHistory=global.PulmoXRViewHistory.create(24);
  let pivotMode='center',axisMode='yaw',analogChangedAt=0;
  let orientationLocked=false,orientationAnchor=null,ctSpatial=null,ctAxis='axial';
  const viewPose=()=>({
    position:mGrp.position.toArray(),quaternion:mGrp.quaternion.toArray(),
    scale:mGrp.scale.x
  });
  function beginViewChange(){
    if(!active||!placed||mode!=='anatomy')return false;
    return viewHistory.begin(viewPose());
  }
  function finishViewChange(){analogChangedAt=0;return viewHistory.finish(viewPose())}
  function stepViewHistory(direction){
    if(benchAligned)return false;
    if(patientReturn)return false;
    if(mode!=='anatomy'||grab.some(Boolean))return false;
    finishViewChange();
    const pose=direction==='undo'?viewHistory.undo(viewPose()):viewHistory.redo(viewPose());
    if(!pose)return false;
    mGrp.position.fromArray(pose.position);
    mGrp.quaternion.fromArray(pose.quaternion);
    if(orientationLocked)orientationAnchor=mGrp.quaternion.clone();
    mGrp.scale.setScalar(pose.scale);
    constrainSizeAndFocus();grabFeedback=direction.toUpperCase()+' / ANATOMY VIEW RESTORED';
    paint();return true;
  }
  const modelExtent=()=>Math.max(.0001,boundsCache?.size?.x||0,
    boundsCache?.size?.y||0,boundsCache?.size?.z||0);
  const focusWorld=()=>boundsCache?.center
    ?boundsCache.center.clone().multiplyScalar(mGrp.scale.x)
      .applyQuaternion(mGrp.quaternion).add(mGrp.position)
    :mGrp.position.clone();
  function constrainSizeAndFocus(){
    if(!boundsCache)return;
    const desiredScale=mGrp.scale.x;
    const actual=limits.clampScale(Math.max(.0001,desiredScale),modelExtent());
    if(Math.abs(actual-desiredScale)>1e-6){
      const center=focusWorld();
      mGrp.scale.setScalar(actual);
      mGrp.position.copy(center).sub(boundsCache.center.clone()
        .multiplyScalar(actual).applyQuaternion(mGrp.quaternion));
      scaleClamps++;limitsApplied++;
    }
    const focus=focusWorld();
    const range=limits.clampFocus(focus.toArray(),head.toArray());
    if(range.clamped){
      mGrp.position.add(new THREE.Vector3(...range.center).sub(focus));
      workspaceClamps++;limitsApplied++;
    }
    mGrp.updateMatrixWorld(true);
  }
  function haptic(i,intensity=.24,duration=35){
    // Capability-gated, subtle Quest feedback only. Absent actuators must
    // never block select, menu actions or movement, and do not imply real
    // anatomy/tool tissue forces.
    const gp=controllerSources[i]?.gamepad;
    try{
      const actuator=gp?.hapticActuators?.[0];
      const vibration=gp?.vibrationActuator;
      let result=null;
      if(actuator?.pulse)result=actuator.pulse(intensity,duration);
      else if(vibration?.playEffect)result=vibration.playEffect('dual-rumble',{
        duration,strongMagnitude:intensity,weakMagnitude:intensity*.6
      });
      else return;
      attempts.hapticRequests++;
      if(result?.catch)result.catch(()=>{});
    }catch{/* Unsupported Quest/browser haptics: no impact on controls. */}
  }
  function feedbackToController(i){
    if(Number.isInteger(i)&&i>=0&&i<2)haptic(i);
  }
  // Inspection transforms are display-only and are always restored before procedures.
  const lobes=(ctx.getReferenceLobes?.()||[]).map(l=>({...l,home:l.mesh.position.clone(),homeQuat:l.mesh.quaternion.clone()}));
  let inspectMode=false,spread=false,selectedLobe=null,inspectionUnit='lobe';
  const lobeGrabs=[null,null];
  let opticalIndex=0;
  const opticalModes=[{name:'BALANCED',gain:1.25,glare:1},{name:'LOW GLARE',gain:1.12,glare:.25},{name:'DEEP VIEW',gain:1.6,glare:.45}];
  function restoreInspectionVisibility(){if(inspectionVisibilitySaved){for(const [o,v] of inspectionVisibilitySaved)o.visible=v;inspectionVisibilitySaved=null;}inspectionFilter='all';}
  function filterLobes(){
    if(!inspectMode||!selectedLobe)return false;
    release(0);release(1);clearSection();const isolate=inspectionFilter!=='isolate';restoreInspectionVisibility();
    if(isolate){inspectionVisibilitySaved=lobes.map(l=>[l.mesh,l.mesh.visible]);const vis=upgrade.inspectionVisibility(lobes.map(l=>l.id),selectedLobe,'isolate');lobes.forEach((l,i)=>l.mesh.visible=inspectionVisibilitySaved[i][1]&&vis[i]);inspectionFilter='isolate';}
    return true;
  }
  function clearSection(){if(!section)return;for(const row of sectionSaved){row.mesh.material=row.material;for(const material of row.clones)material.dispose();}sectionSaved=[];rBro.localClippingEnabled=sectionClipWas;section=null;}
  function updateSection(){if(!section)return;const l=lobes.find(l=>l.id===selectedLobe);if(!l){clearSection();return;}const mesh=l.mesh;mesh.updateWorldMatrix(true,false);const bounds=section.bounds;const axis=sectionAxis;const point=bounds.getCenter(new THREE.Vector3());point[axis]=bounds.min[axis]+(bounds.max[axis]-bounds.min[axis])*sectionFraction;const normal=new THREE.Vector3();normal[axis]=1;section.plane.setFromNormalAndCoplanarPoint(normal,point).applyMatrix4(mesh.matrixWorld);}
  function toggleSection(){if(section){clearSection();return;}if(!inspectMode||!selectedLobe)return;const l=lobes.find(l=>l.id===selectedLobe);if(!l?.mesh?.geometry)return;const mesh=l.mesh;mesh.geometry.computeBoundingBox();const bounds=mesh.geometry.boundingBox?.clone();if(!bounds||bounds.isEmpty())return;release(0);release(1);sectionClipWas=rBro.localClippingEnabled;rBro.localClippingEnabled=true;section={plane:new THREE.Plane(),bounds};const material=mesh.material;const clones=(Array.isArray(material)?material:[material]).map(a=>{const b=a.clone();b.clippingPlanes=[...(a.clippingPlanes||[]),section.plane];b.needsUpdate=true;return b;});mesh.material=Array.isArray(material)?clones:clones[0];sectionSaved.push({mesh,material,clones});updateSection();}
  function resetLobes(){
    clearSection();restoreInspectionVisibility();lobeGrabs.fill(null);selectedLobe=null;spread=false;
    for(const l of lobes){l.mesh.position.copy(l.home);l.mesh.quaternion.copy(l.homeQuat);l.mesh.updateMatrixWorld(true);}
  }
  function setInspection(enabled){
    if(enabled&&benchAligned)return false;
    if(enabled&&(mode!=='anatomy'||getMeshMode()!=='reference'||!lobes.length))return false;
    release(0);release(1);resetLobes();inspectMode=enabled;
    grabFeedback=enabled?'LOBE INSPECTION / AIM AND HOLD GRIP OR PINCH':'ASSEMBLED LUNG / WHOLE ORGAN GRAB';
    return true;
  }
  function spreadLobes(unit='lobe'){
    if(!inspectMode&&!setInspection(true))return;
    release(0);release(1);resetLobes();spread=true;inspectionUnit=unit;
    for(const l of lobes){const offset=unit==='lung'?[l.id.startsWith('R')?.65:-.65,0,0]:l.offset;l.mesh.position.add(new THREE.Vector3(...offset));l.mesh.updateMatrixWorld(true);}
    grabFeedback=unit==='lung'?'SEPARATED LUNGS / GRAB EITHER SIDE':'EXPLODED VIEW / GRAB INDIVIDUAL LOBES';
  }
  const surfaceTarget=()=> (inspectMode?lobes.map(l=>l.mesh):getAnatomyMeshes()).filter(o=>o&&o.visible&&o.isObject3D);
  const trackedController=i=>controllerSources[i]?.gripSpace?grips[i]:hands[i];
  const effectiveGrip=i=>grabSource[i]==='hand'?handProxies[i]:trackedController(i);
  const release=(i,source=null,input=null)=>{
    if(panelEdit?.i===i&&(!source||source==='controller')&&(!input||input==='grip'))stopPanelEdit();
    if(source&&grabSource[i]&&grabSource[i]!==source)return;
    if(input&&grab[i]&&grab[i].input!==input)return;
    const endedGrab=grab[i];
    lobeGrabs[i]=null;
    grab[i]=null;grabSource[i]=null;
    if(endedGrab&&!grab.some(Boolean))finishViewChange();
    if(endedGrab){
      // Release is deterministic: the model stays where it was placed.
      // No automatic summon, resize, teleport or camera movement on tap.
      grabFeedback='RELEASED '+(endedGrab.method||'ANATOMY')+' / POSITION PRESERVED';
      if(source==='controller')haptic(i,.10,22);
    }
    if(scopeSteering?.i===i&&(!source||source==='controller'))scopeSteering=null;
    if(grab[0]&&!grab[1])capture(0);
    if(grab[1]&&!grab[0])capture(1);
    gesture=null;
  };
  // Actual source centerlines and scope position, independent of the optical feed.
  const routeOverlay=new THREE.Group();routeOverlay.name='xr21-source-airway-location';routeOverlay.visible=false;mGrp.add(routeOverlay);
  let routeVisible=true,routeSegment=null;
  const routeLines=new Map();
  for(const seg of SEGS){
    const curve=pathCurves[seg.id];if(!curve)continue;
    const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(24)),
      new THREE.LineBasicMaterial({color:0x7294a4,transparent:true,opacity:.42,depthTest:false,depthWrite:false}));
    line.renderOrder=32;routeOverlay.add(line);routeLines.set(seg.id,line);
  }
  const routeTip=new THREE.Mesh(new THREE.SphereGeometry(.038,12,8),
    new THREE.MeshBasicMaterial({color:0xe9d7a3,depthTest:false,depthWrite:false}));
  routeTip.renderOrder=33;routeOverlay.add(routeTip);
  function updateRouteOverlay(){
    routeOverlay.visible=mode==='procedure'&&routeVisible&&getMeshMode()==='reference';
    if(!routeOverlay.visible)return;
    if(routeSegment!==scope.segId){
      routeSegment=scope.segId;
      const route=global.PulmoXRProcedureGuide.route({segments:SEG_MAP,segmentId:scope.segId});
      for(const [id,line] of routeLines){line.material.color.setHex(id===scope.segId?0x9ce8d9:route.ids?.includes(id)?0x83b9c3:0x7294a4);line.material.opacity=id===scope.segId?1:route.ids?.includes(id)?.8:.35;}
    }
    const curve=pathCurves[scope.segId];if(curve)routeTip.position.copy(curve.getPoint(Math.max(0,Math.min(1,scope.t))));
  }
  function setMode(next){
    if(next==='procedure'&&benchAligned){grabFeedback='RELEASE BENCH ALIGNMENT BEFORE PROCEDURE';paint();return false;}
    if(!['anatomy','procedure'].includes(next))return false;
    // A4: the actual original bronchoscope traverses SEGS (a reference
    // tree), not the uploaded CT GLB. Never silently call that patient-specific.
    if(next==='procedure'&&xrPresentation==='ar'&&!placed){
      grabFeedback='PLACE REFERENCE ANATOMY BEFORE ENTERING BRONCHOSCOPY';
      paint();return false;
    }
    if(next==='procedure'&&getMeshMode()!=='reference'){
      grabFeedback='CT-SPECIFIC BRONCHOSCOPY BLOCKED / NO REGISTERED AIRWAY';
      showToast('The CT mesh has no reviewed airway centreline. Return to the procedural reference before bronchoscopy.','warn');
      paint();return false;
    }
    if(mode==='anatomy'&&next!=='anatomy'){finishViewChange();if(inspectMode)setInspection(false);}
    summonGeneration++;
    const previousMode=mode;if(next!=='procedure')leavePatientStation();resetEmbodiedGrip();
    stopPanelEdit();panelPinned.clear();mode=next;grab[0]=grab[1]=null;grabSource.fill(null);
    gesture=null;scopeSteering=null;summonTween=null;
    grabFeedback='AIM AT LUNG / HOLD TRIGGER OR GRIP';
    if(next!=='procedure'){if(localRecording.snapshot().running){stopRecording('PROCEDURE_EXIT');const packet=localRecording.packet();if(packet)loadRecordedReview(packet);}lastShaftAt=-Infinity;lastShaftSample=shaftEvidence;lastShaftKey=null;hapticDryRun.stop('PROCEDURE_EXIT');hapticEvidence=hapticLab.estimate();lastAdvanceAt=-Infinity;ebusProbe.reset();probeGrip.reset();ebusVisual.hide();radialEnabled=false;radialEvidence=null;ctx.setReferenceMotion?.(1);motionEvidence=respiratoryMotion.snapshot(false);procedureMetrics.suspend();videoScope.group.visible=false;shaftClutch.reset();clutchAxis=null;clutchHeld=false;insertedShaft.hide();channel.reset();accessory.group.visible=false;}
    if(previousMode!==next){monitorFocus=false;menuPage=next==='procedure'?'station':xrPresentation==='ar'&&!placed?'mr':'primary';actionHitbox=panelLayouts[menuPage];hoverPanels.fill(null);}
    if(scopeScreen)scopeScreen.visible=next==='procedure';
    if(next!=='procedure'){ebusScreen.visible=false;endoscopyTower.hide();ultrasoundTower.hide();}
    if(guidePlane)guidePlane.visible=next==='procedure'&&guideVisible;
    if(next==='anatomy'){scope.velocity=0;if(active&&physicalControl?.snapshot().armed&&previousMode==='procedure')physicalControl.stop('PROCEDURE_EXIT');}
    updateRouteOverlay();
    if(next==='procedure'){
      camScope.fov=station.snapshot().fovDegrees;
      camScope.updateProjectionMatrix();
      positionPanels();
    }else if(active&&viewerValid)positionPanels();
    paint();return true;
  }
  function headPose(frame){
    const ref=rBro.xr.getReferenceSpace();
    if(!frame||!ref)return false;
    const p=frame.getViewerPose(ref)?.transform;
    if(!p)return false;
    if(![p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w].every(Number.isFinite))return false;
    head.set(p.position.x,p.position.y,p.position.z);
    const q=new THREE.Quaternion(p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w).normalize();
    forward.set(0,0,-1).applyQuaternion(q).setY(0);
    if(forward.lengthSq()<.01)forward.set(0,0,-1);
    forward.normalize();
    right.set(1,0,0).applyQuaternion(q).setY(0);
    if(right.lengthSq()<.01)right.set(1,0,0);
    right.normalize();viewerValid=true;return true;
  }
  function sourceBounds(){
    // Measure the actual original geometry, excluding effects, background,
    // particles, floor, menu or XR controllers.
    modelBox.makeEmpty();
    mGrp.updateMatrixWorld(true);
    const rootInverse=new THREE.Matrix4().copy(mGrp.matrixWorld).invert();
    const local=new THREE.Matrix4();
    const objects=surfaceTarget();
    for(const root of objects){
      root.traverse(obj=>{
        if(!obj.isMesh||!obj.geometry)return;
        if(!obj.geometry.boundingBox)obj.geometry.computeBoundingBox();
        if(!obj.geometry.boundingBox)return;
        obj.updateMatrixWorld(true);
        local.multiplyMatrices(rootInverse,obj.matrixWorld);
        modelBox.union(obj.geometry.boundingBox.clone().applyMatrix4(local));
      });
    }
    if(modelBox.isEmpty())return {center:new THREE.Vector3(),size:new THREE.Vector3(2,2,1)};
    return {center:modelBox.getCenter(new THREE.Vector3()),size:modelBox.getSize(new THREE.Vector3())};
  }
  // Reachable seated default: the anatomy is close enough for a real grip,
  // while remote laser grabs still work at larger distances.
  function placeOnMRSurface(point){
    if(xrPresentation!=='ar'||!viewerValid)return false;
    if(!point||![point.x,point.y,point.z].every(Number.isFinite))return false;
    const target=new THREE.Vector3(point.x,point.y,point.z);
    const reach=head.distanceTo(target);
    if(reach<.25||reach>2.8){
      grabFeedback='SURFACE OUTSIDE WORK AREA / REPOSITION';
      mrPlacement.reposition();paint();return false;
    }
    release(0);release(1);summonGeneration++;summonTween=null;
    const bounds=sourceBounds();boundsCache=bounds;
    const scale=limits.clampScale(
      Math.min(.68/Math.max(.0001,bounds.size.x),
        .64/Math.max(.0001,bounds.size.y)),modelExtent());
    mGrp.quaternion.identity();mGrp.scale.setScalar(scale);
    // Upright, bounded reference anatomy stands above a real hit surface.
    // Do not rotate the lung by the XR plane normal or infer clinical scale.
    const center=target.clone().add(new THREE.Vector3(0,
      Math.max(.05,bounds.size.y*scale*.5+.025),0));
    mGrp.position.copy(center).sub(bounds.center.clone().multiplyScalar(scale));
    mGrp.visible=true;mGrp.updateWorldMatrix(true,false);
    placed=true;scaleMode='mr-surface-comfort';
    grabFeedback='MR SURFACE CONFIRMED / '+mrPlacement.snapshot().anchorStatus.toUpperCase();
    positionPanels();paint();return true;
  }
  function frameGroup(distance=.76,width=.52){
    if(benchAligned){positionPanels();return true;}
    if(!viewerValid)return false;if(patientReturn){positionPanels();return true;}stopPanelEdit();panelPinned.clear();
    if(placed){finishViewChange();beginViewChange();}
    summonGeneration++;summonTween=null;
    release(0);release(1);
    if(placed){finishViewChange();beginViewChange();}
    const bounds=sourceBounds();boundsCache=bounds;
    const scale=limits.clampScale(Math.min(width/Math.max(.0001,bounds.size.x),
      .65/Math.max(.0001,bounds.size.y)),modelExtent());
    const center=head.clone().addScaledVector(forward,distance).add(new THREE.Vector3(0,-.20,0));
    mGrp.quaternion.identity();
    if(orientationLocked)orientationAnchor=mGrp.quaternion.clone();
    mGrp.scale.setScalar(scale);scaleMode='comfort';
    mGrp.position.copy(center).sub(bounds.center.clone().multiplyScalar(scale));
    mGrp.updateMatrixWorld(true);placed=true;
    constrainSizeAndFocus();positionPanels();
    finishViewChange();return true;
  }
  function setSourceScale(){
    const audit=getPatientAudit?.();
    if(!audit||getMeshMode()!=='patient'||audit.status!=='coframed'||
       !audit.sourceScaleEligible||mode!=='anatomy'||grab.some(Boolean))
      return false;
    const sourceScale=audit.sceneUnitMetres;
    if(!Number.isFinite(sourceScale)||sourceScale<.005||sourceScale>2)return false;
    const bounds=sourceBounds();boundsCache=bounds;
    // A deliberate source-scale request must not produce a room-filling organ.
    const visibleWidth=sourceScale*modelExtent();
    if(visibleWidth>limits.CONFIG.maximumSpanM||visibleWidth<limits.CONFIG.minimumSpanM){
      showToast('Physical source scale outside comfortable view; retaining bounded size','warn');
      return false;
    }
    beginViewChange();
    const local=bounds.center;
    const currentCenter=local.clone().multiplyScalar(mGrp.scale.x)
      .applyQuaternion(mGrp.quaternion).add(mGrp.position);
    mGrp.scale.setScalar(sourceScale);
    mGrp.position.copy(currentCenter)
      .sub(local.clone().multiplyScalar(sourceScale).applyQuaternion(mGrp.quaternion));
    scaleMode='source-derived';
    constrainSizeAndFocus();finishViewChange();
    return true;
  }
  function clearanceState(){
    const seg=SEG_MAP[scope.segId];
    if(!seg||!global.PulmoXR08)return null;
    try{
      return global.PulmoXR08.assessLumenClearance({
        segment:seg,t:scope.t,scopeDiameterMm:getScopeDiameterMm(),
        mmPerSceneUnit:getMmPerSceneUnit()
      });
    }catch{return null}
  }
  function moveWorkspace(command){
    if(!active||!viewerValid||!placed||xrPresentation!=='vr'||mode!=='anatomy'||grab.some(Boolean)||handGrasped.some(Boolean)||workspaceRelayout||panelEdit)return false;
    const ref=rBro.xr.getReferenceSpace(),Ctor=global.XRRigidTransform;
    if(!Ctor||!ref?.getOffsetReferenceSpace||!rBro.xr.setReferenceSpace)return false;
    const start=head.toArray(),origin=room.position.toArray(),obstacles=[{center:plinth.getWorldPosition(new THREE.Vector3()).toArray(),radius:.78}];
    let spec;if(command==='viewpoint'){const target=[origin[0],start[1],origin[2]+1.6];spec=upgrade.targetOffset({head:start,target,room:origin,obstacles});}else{spec=upgrade.workspaceTransform({head:start,forward:forward.toArray(),right:right.toArray(),room:origin,command});if(spec&&!spec.yaw&&!upgrade.clearWorkspacePath(start,start.map((v,i)=>v-spec.translation[i]),obstacles))spec=null;}
    if(!spec){grabFeedback='VIRTUAL WORKSPACE BOUNDARY / MOVEMENT BLOCKED';paint();return false;}
    try{const [x,y,z]=spec.translation;rBro.xr.setReferenceSpace(ref.getOffsetReferenceSpace(new Ctor({x,y,z},{x:0,y:Math.sin(spec.yaw/2),z:0,w:Math.cos(spec.yaw/2)})));workspaceMoves++;workspaceRelayout=true;return true;}
    catch(error){grabFeedback='WORKSPACE OFFSET UNAVAILABLE';paint();return false;}
  }
  function selectedPanel(){return panelTarget==='ebus'?ebusScreen:panelTarget==='guide'?guidePlane:panelTarget==='menu'?worldMenu:scopeScreen;}
  function stopPanelEdit(){if(panelEdit){panelPinned.add(panelEdit.object);panelEdit=null;panelMoves++;}}
  function startPanelEdit(i,object){if(!object?.visible||panelEdit||!viewerValid)return false;const grip=trackedController(i);if(!grip)return false;grip.updateWorldMatrix(true,false);object.updateWorldMatrix(true,false);panelEdit={i,object,relative:new THREE.Matrix4().copy(grip.matrixWorld).invert().multiply(object.matrixWorld)};return true;}
  function updatePanelEdit(dt){if(!panelEdit)return;const {i,object,relative}=panelEdit;const grip=trackedController(i);if(!grip||!controllerSources[i]||!viewerValid){stopPanelEdit();return;}grip.updateWorldMatrix(true,false);const p=new THREE.Vector3(),q=new THREE.Quaternion(),s=new THREE.Vector3();new THREE.Matrix4().multiplyMatrices(grip.matrixWorld,relative).decompose(p,q,s);const bounded=upgrade.boundedPanelPosition(head.toArray(),p.toArray());if(!bounded)return;const step=upgrade.limitedStep(object.position.toArray(),bounded,dt,1.8);if(step)object.position.set(...step);object.lookAt(head);object.updateWorldMatrix(true,false);}
  function positionPanels(){
    if(!viewerValid)return;
    if(worldMenu&&!panelPinned.has(worldMenu)){
      // Independent world-space panel, placed only on intentional layout
      // changes or recenter. It does not follow the viewer every XR frame.
      worldMenu.position.copy(head).addScaledVector(forward,.91).addScaledVector(right,-.72+panelOffset);
      worldMenu.position.y-=.02;
      worldMenu.scale.setScalar(readingMode?1.24:1);worldMenu.lookAt(head);
      worldMenu.updateWorldMatrix(true,false);
    }
    if(compactDock){compactDock.position.copy(head).addScaledVector(forward,.98).addScaledVector(right,-.46+panelOffset*.65);compactDock.position.y-=.43;compactDock.lookAt(head);compactDock.updateWorldMatrix(true,false);}
    const pose=station.snapshot().pose;
    function place(object,spec){
      if(!object||panelPinned.has(object))return;
      object.position.copy(head).addScaledVector(forward,spec.depth)
        .addScaledVector(right,spec.right);
      object.position.y+=spec.y;
      object.scale.setScalar(spec.scale*(object===guidePlane&&readingMode?1.18:1));
      object.lookAt(head);object.updateWorldMatrix(true,false);
    }
    if(monitorFocus){
      place(scopeScreen,{depth:1.12,right:.62,y:.16,scale:1.23});
      place(guidePlane,{depth:1.27,right:.91,y:-.62,scale:.88});
    }else{
      place(scopeScreen,pose.monitor);
      place(guidePlane,pose.guide);
    }
    if(radialEnabled&&mode==='procedure')place(ebusScreen,{depth:1.54,right:1.47,y:.10,scale:1});
    composeSpatialWorkspace();
  }
  function composeSpatialWorkspace(){
    if(!active||!viewerValid||panelEdit||xrPresentation==='ar'&&!placed)return;
    const R=forward.clone().cross(new THREE.Vector3(0,1,0)).normalize();
    const project=p=>{const d=p.clone().sub(head);return [d.dot(R),d.y,d.dot(forward)];};
    function corners(obj,min,max){obj.updateWorldMatrix(true,false);const pts=[];for(const a of [min.x,max.x])for(const b of [min.y,max.y])for(const z of [min.z,max.z])pts.push(project(new THREE.Vector3(a,b,z).applyMatrix4(obj.matrixWorld)));return pts;}
    let protectedEnvelope=null;
    if(placed&&boundsCache&&mGrp.visible){
      const half=boundsCache.size.clone().multiplyScalar(.5),center=boundsCache.center;
      protectedEnvelope=spatialExperience.envelope(corners(mGrp,center.clone().sub(half),center.clone().add(half)));
    }
    const entries=[
      {id:'monitor',object:scopeScreen,w:1.03,h:.69,lower:.12,visible:mode==='procedure'},
      {id:'ebus',object:ebusScreen,w:.66,h:.70,lower:.09,visible:mode==='procedure'&&radialEnabled},
      {id:'guide',object:guidePlane,w:.90,h:.60,lower:0,visible:mode==='procedure'&&guideVisible},
      {id:'console',object:worldMenu,w:.82,h:.48,lower:0,visible:worldMenu.visible},
      {id:'rail',object:compactDock,w:.76,h:.112,lower:0,visible:!worldMenu.visible}
    ].filter(e=>e.visible);
    const fixed=[],items=[];
    for(const e of entries){
      const obj=e.object,halfX=e.w/2,halfY=e.h/2;
      if(panelPinned.has(obj)){fixed.push(spatialExperience.envelope(corners(obj,new THREE.Vector3(-halfX,-halfY-e.lower,-.05),new THREE.Vector3(halfX,halfY,.05))));continue;}
      const p=project(obj.position),scale=obj.scale.x;
      const preferred={right:p[0],y:p[1],depth:p[2],scale};
      const sign=e.id==='monitor'||e.id==='guide'?1:panelOffset>0?1:-1;
      const alternatives=e.id==='monitor'?[{right:.91,y:.25,depth:1.40,scale},{right:1.06,y:.32,depth:1.57,scale}]:
        e.id==='ebus'?[{right:1.75,y:.15,depth:1.76,scale},{right:1.28,y:.64,depth:1.65,scale},{right:-1.25,y:.23,depth:1.60,scale}]:
        e.id==='guide'?[{right:1.0,y:-.48,depth:1.48,scale},{right:1.26,y:-.12,depth:1.62,scale},{right:.98,y:-.67,depth:1.52,scale}]:
        e.id==='console'?[{right:sign*.98,y:.08,depth:1.24,scale},{right:sign*1.22,y:.13,depth:1.42,scale},{right:sign*.98,y:.48,depth:1.45,scale}]:
        [{right:sign*.65,y:-.57,depth:1.18,scale},{right:sign*.87,y:-.48,depth:1.28,scale},{right:0,y:-.68,depth:1.25,scale}];
      items.push({id:e.id,preferred,alternatives,width:e.w,height:e.h,lowerExtension:e.lower});
    }
    const result=spatialExperience.compose({items,protectedEnvelope,fixed});
    for(const e of entries){const p=result.poses[e.id];if(!p||panelPinned.has(e.object))continue;const o=e.object;o.position.copy(head).addScaledVector(R,p.right).addScaledVector(forward,p.depth);o.position.y+=p.y;o.scale.setScalar(p.scale);o.lookAt(head);o.updateWorldMatrix(true,false);}
    compositionCount++;compositionEvidence={method:result.method,reports:result.reports,protectedEnvelope,pinnedWindows:fixed.length,layoutEvents:compositionCount,physicalOcclusionVerified:false};
  }
  let lastContact=null;
  function surfaceContact(i,source='controller'){
    const controller=source==='hand'?handProxies[i]:trackedController(i);
    if(!controller||!global.PulmoXRSurface)return {hit:false,reason:'tracker-unavailable'};
    controller.updateMatrixWorld(true);
    const point=controller.getWorldPosition(new THREE.Vector3());
    try{
      const result=global.PulmoXRSurface.contact(THREE,surfaceTarget(),point,
        {radius:.155,maxMeshes:256});
      lastContact={hit:result.hit,reason:result.reason,
        distanceM:result.distanceM};
      return result;
    }catch(error){
      lastContact={hit:false,reason:'surface-test-error',distanceM:null};
      console.warn('PulmoSIMU XR surface contact failed:',String(error?.message||error));
      return lastContact;
    }
  }
  function capture(i){
    const grip=effectiveGrip(i);
    if(!grip||!grab[i]||lobeGrabs[i])return;
    grip.updateMatrixWorld(true);mGrp.updateMatrixWorld(true);
    grab[i].relative.copy(grip.matrixWorld).invert().multiply(mGrp.matrixWorld);
  }
  function startGrab(i,source='controller',input='grip'){
    if(spatialCal.snapshot().armed||benchAligned)return false;
    const side=controllerSide[i]||controllerSources[i]?.handedness;
    if(source==='controller'&&input==='grip'&&active&&menuPage==='panel-edit'){const object=selectedPanel();if(startPanelEdit(i,object)){scope.velocity=0;scopeSteering=null;shaftClutch.reset();if(physicalControl?.snapshot().armed)physicalControl.stop('WINDOW_EDIT');grabFeedback='HOLD GRIP / MOVE SELECTED WINDOW';paint();return true;}}
    if(source==='controller'&&mode==='procedure'&&active&&side==='right'){
      if(input!=='grip')return false;
      const g=trackedController(i);if(!g)return false;
      g.updateWorldMatrix(true,false);
      scopeSteering={i,origin:g.getWorldQuaternion(new THREE.Quaternion()),
        deflH:scope.deflH,deflV:scope.deflV};
      grabFeedback='RIGHT GRIP / STEERING SOURCE SCOPE';return true;
    }
    if(!active||!placed||mode!=='anatomy'||grab[i]||section)return false;
    const chosen=source==='controller'?choosePointer(i):null;
    if(chosen?.choice.kind==='panel'){
      grabFeedback='PANEL TARGETED / USE TRIGGER TO SELECT A CONTROL';
      paint();return false;
    }
    const grip=source==='hand'?handProxies[i]:trackedController(i);
    if(!grip)return false;
    // A near-hand contact is tested against original mesh triangles.
    const near=surfaceContact(i,source);
    // If the anatomy is beyond arm's reach, pointing + holding either
    // trigger or grip also acquires the ORIGINAL mesh via triangle raycast.
    const remote=!near.hit
      ?(source==='controller'?chosen.organ:global.PulmoXRRemoteTarget.pick(THREE,hands[i],surfaceTarget(),{maxDistance:2.5})):null;
    if(!near.hit&&!remote?.hit){
      attempts.rejected++;
      grabFeedback=remote?.reason==='no-anatomy-source'
        ?'NO ANATOMY SOURCE / SELECT REFERENCE'
        :'AIM LASER AT LUNG THEN HOLD TRIGGER OR GRIP';
      lastContact={hit:false,reason:remote?.reason||near.reason||'no-contact',
        distanceM:null};
      paint();return false;
    }
    if(inspectMode){
      const lobe=lobes.find(l=>l.mesh===remote?.object||l.mesh.uuid===near.meshUuid);
      if(!lobe)return false;
      const members=inspectionUnit==='lung'?lobes.filter(l=>l.id[0]===lobe.id[0]):[lobe];
      if(lobeGrabs.some(g=>g?.members.some(m=>members.includes(m.lobe))))return false;
      grip.updateMatrixWorld(true);
      lobeGrabs[i]={lobe,members:members.map(l=>{l.mesh.updateMatrixWorld(true);return {lobe:l,relative:new THREE.Matrix4().copy(grip.matrixWorld).invert().multiply(l.mesh.matrixWorld)}})};
      grabSource[i]=source;grab[i]={input,method:'lobe-inspection'};selectedLobe=lobe.id;
      grabFeedback='INSPECT / '+lobe.name.toUpperCase();paint();return true;
    }
    if(analogChangedAt)finishViewChange();
    beginViewChange();
    summonGeneration++;summonTween=null;
    const method=near.hit?'near-surface':'distance-ray';
    attempts[near.hit?'acceptedNear':'acceptedRay']++;
    if(source==='controller')attempts[input==='trigger'?'trigger':'grip']++;
    lastContact={hit:true,reason:method,distanceM:remote?.distanceM??near.distanceM??null};
    grabSource[i]=source;
    grab[i]={relative:new THREE.Matrix4(),input,method,startedAt:performance.now(),
      startPosition:mGrp.position.clone(),startQuaternion:mGrp.quaternion.clone()};
    capture(i);
    if(source==='controller')haptic(i,.31,48);
    grabFeedback='HOLD '+input.toUpperCase()+' / MOVE CONTROLLER / '+method.toUpperCase();
    if(grab[0]&&grab[1]){
      effectiveGrip(0).updateWorldMatrix(true,false);
      effectiveGrip(1).updateWorldMatrix(true,false);
      const a=effectiveGrip(0).getWorldPosition(new THREE.Vector3());
      const b=effectiveGrip(1).getWorldPosition(new THREE.Vector3());
      gesture={a:a.clone(),b:b.clone(),center:a.clone().add(b).multiplyScalar(.5),
        position:mGrp.position.clone(),quat:mGrp.quaternion.clone(),scale:mGrp.scale.x};
    }else gesture=null;
    paint();
    return true;
  }
  function updateGrab(){
    if(!grab[0]&&!grab[1]||mode!=='anatomy')return;
    mGrp.parent?.updateMatrixWorld(true);
    if(inspectMode){
      mGrp.updateMatrixWorld(true);
      for(let i=0;i<2;i++){
        const g=lobeGrabs[i];if(!g)continue;
        const grip=effectiveGrip(i);if(!grip)continue;
        grip.updateMatrixWorld(true);
        for(const member of g.members){
          const target=new THREE.Matrix4().copy(mGrp.matrixWorld).invert().multiply(grip.matrixWorld).multiply(member.relative);
          const p=new THREE.Vector3(),q=new THREE.Quaternion(),s=new THREE.Vector3();target.decompose(p,q,s);
          const offset=p.sub(member.lobe.home);if(offset.length()>2)offset.setLength(2);
          member.lobe.mesh.position.copy(member.lobe.home).add(offset);member.lobe.mesh.quaternion.copy(q);member.lobe.mesh.updateMatrixWorld(true);
        }
      }return;
    }
    const previousPosition=mGrp.position.clone(),previousOrientation=mGrp.quaternion.clone();
    if(grab[0]&&grab[1]&&gesture){
      effectiveGrip(0).updateMatrixWorld(true);effectiveGrip(1).updateMatrixWorld(true);
      const a=effectiveGrip(0).getWorldPosition(new THREE.Vector3()),b=effectiveGrip(1).getWorldPosition(new THREE.Vector3());
      const initial=gesture.b.clone().sub(gesture.a),current=b.clone().sub(a);
      const sizing=limits.scaleFromHands(gesture.scale,initial.length(),
        current.length(),modelExtent());
      if(!sizing.valid)return;
      const rotation=new THREE.Quaternion().setFromUnitVectors(initial.normalize(),current.normalize());
      const scaling=sizing.scale;
      const offset=gesture.position.clone().sub(gesture.center).applyQuaternion(rotation).multiplyScalar(scaling/gesture.scale);
      mGrp.position.copy(a.add(b).multiplyScalar(.5)).add(offset);
      mGrp.quaternion.copy(rotation).multiply(gesture.quat);
      mGrp.scale.setScalar(scaling);
    }else{
      const i=grab[0]?0:1;
      effectiveGrip(i).updateMatrixWorld(true);
      const matrix=new THREE.Matrix4().multiplyMatrices(effectiveGrip(i).matrixWorld,grab[i].relative);
      matrix.decompose(mGrp.position,mGrp.quaternion,mGrp.scale);
    }
    // Keep intended grasp translation without controller-induced rotation.
    if(orientationLocked&&orientationAnchor){
      const proposedCenter=focusWorld();
      mGrp.quaternion.copy(orientationAnchor);
      mGrp.position.add(proposedCenter.sub(focusWorld()));
    }
    // Clamp the visible organ, not its arbitrary scene pivot. This prevents
    // giant scales and allows the grip to move farther within a safe workspace.
    constrainSizeAndFocus();
    if(mGrp.position.distanceToSquared(previousPosition)>.000004||
       mGrp.quaternion.angleTo(previousOrientation)>.004)
      attempts.movedFrames++;
  }
  function updateTrackedHands(frame){
    handSamples={};
    const session=rBro.xr.getSession(),space=rBro.xr.getReferenceSpace();
    const found=[false,false],valid=[false,false];
    if(session&&space&&frame?.getJointPose){
      for(const source of session.inputSources){
        if(!source.hand)continue;
        // The same WebXR input source owns its grip and hand poses.
        const mapped=controllerSources.indexOf(source);
        const i=mapped>=0?mapped:
          source.handedness==='left'?0:source.handedness==='right'?1:-1;
        if(i<0)continue;
        found[i]=true;
        const index=source.hand.get('index-finger-tip'),thumb=source.hand.get('thumb-tip');
        if(!index||!thumb)continue;
        const p=frame.getJointPose(index,space),q=frame.getJointPose(thumb,space);
        if(!p?.transform?.position||!q?.transform?.position)continue;
        const a=p.transform.position,b=q.transform.position;
        const pA=new THREE.Vector3(a.x,a.y,a.z),pB=new THREE.Vector3(b.x,b.y,b.z);
        const proxy=handProxies[i];
        // Use this frame's real target-ray pose, even if the runtime did
        // not emit a Three controller-connected event for hand input.
        const targetPose=source.targetRaySpace&&frame.getPose?.(source.targetRaySpace,space)?.transform;
        if(targetPose?.position&&targetPose?.orientation){
          const h=hands[i];h.position.set(targetPose.position.x,targetPose.position.y,targetPose.position.z);
          h.quaternion.set(targetPose.orientation.x,targetPose.orientation.y,targetPose.orientation.z,targetPose.orientation.w);
          h.matrix.compose(h.position,h.quaternion,h.scale);h.matrixWorldNeedsUpdate=true;h.updateMatrixWorld(true);
        }
        proxy.position.copy(pA).add(pB).multiplyScalar(.5);
        const palmJoint=source.hand.get('middle-finger-phalanx-proximal');const palmPose=palmJoint&&frame.getJointPose(palmJoint,space)?.transform?.position;
        if(handGrasped[i]&&palmPose)proxy.position.set(palmPose.x,palmPose.y,palmPose.z);
        const wrist=source.hand.get('wrist');
        const w=wrist?frame.getJointPose(wrist,space):null;
        if(w?.transform?.orientation){
          const o=w.transform.orientation;
          proxy.quaternion.set(o.x,o.y,o.z,o.w).normalize();
        }else proxy.quaternion.identity();
        proxy.updateMatrixWorld(true);valid[i]=true;
        handSeen[i]=true;
        updateHandVisual(i,source,frame,space);
        if(contactMarkers[i]){
          contactMarkers[i].position.copy(proxy.position);
          contactMarkers[i].visible=mode==='anatomy';
        }
        const pinchDistance=pA.distanceTo(pB);
        const pressed=!handGrasped[i]&&(handPinched[i]?pinchDistance<.055:pinchDistance<=.026);
        if(pressed&&!handPinched[i]){
          const hit=panelIntersection(rayFrom(i));
          if(hit?.action){handMenuBlocked[source.handedness]=true;attempts.menuActions++;activate(hit.action);}
          else if(mode==='anatomy'){
            if(xrPresentation==='ar'&&!placed){confirmOrManualMR();}
            else startGrab(i,'hand');
          }else if(mode==='procedure'&&!handProcedure&&scope.atBranch&&!accessoryDeployed()&&!probeLocked()){
            enterBranch(source.handedness==='right'?Math.min(1,(SEG_MAP[scope.segId]?.children.length||1)-1):0);
          }
        }else if(!pressed&&handPinched[i]&&grabSource[i]==='hand')release(i,'hand');
        handPinched[i]=pressed;if(!pressed)handMenuBlocked[source.handedness]=false;
        // A closed-hand grasp is separate from pinch/menu input and only acquires near anatomy.
        const joint=name=>{const j=source.hand.get(name),p=j&&frame.getJointPose(j,space)?.transform?.position;return p?[p.x,p.y,p.z]:null;};
        const palm=joint('middle-finger-phalanx-proximal'),wristPos=joint('wrist');
        const closed=mode==='anatomy'&&placed&&!pressed&&upgrade.grasp({wrist:wristPos,palm,index:pA.toArray(),middle:joint('middle-finger-tip'),ring:joint('ring-finger-tip'),pinky:joint('pinky-finger-tip'),thumb:pB.toArray(),held:handGrasped[i]});
        if(closed&&!handGrasped[i]&&!grab[i]){if(palm){proxy.position.set(...palm);proxy.updateMatrixWorld(true);}const near=surfaceContact(i,'hand');if(near.hit){startGrab(i,'hand','grasp');handGrasped[i]=grabSource[i]==='hand';}}
        if(!closed&&handGrasped[i]){release(i,'hand','grasp');handGrasped[i]=false;}
      }
    }
    for(let i=0;i<2;i++)if(!found[i]||!valid[i]){
      if(handSeen[i])health.event('handTrackingLost');
      handSeen[i]=false;handPinched[i]=false;handGrasped[i]=false;
      if(contactMarkers[i])contactMarkers[i].visible=false;
      if(grabSource[i]==='hand')release(i,'hand');
    }
  }
  function updateSteering(){
    if(mode!=='procedure'||!patientMotionAllowed()||probeLocked()||!scopeSteering||physicalControl?.snapshot().armed||accessoryDeployed())return;
    const {i,origin,deflH,deflV}=scopeSteering;
    const controller=trackedController(i);
    if(!controller||!controllerSources[i]){
      // Disconnected input must not retain steering or insertion commands.
      scopeSteering=null;scope.velocity=0;
      health.event('controllerTrackingLost');
      return;
    }
    controller.updateWorldMatrix(true,false);
    const orientation=controller.getWorldQuaternion(new THREE.Quaternion());
    // Change in controller orientation relative to the moment of grasp.
    // This is a VR input mapping to the ORIGINAL scope deflection state,
    // not an assertion of mechanical shaft/knob or force-feedback realism.
    const delta=origin.clone().invert().multiply(orientation);
    const euler=new THREE.Euler().setFromQuaternion(delta,'YXZ');
    scope.deflH=instrument.rateLimit(scope.deflH,safe(-1,1,deflH+euler.y/0.65),Math.min(.05,(performance.now()-(scopeSteering.lastUpdate||performance.now()))/1000));
    scope.deflV=instrument.rateLimit(scope.deflV,safe(-1,1,deflV-euler.x/0.65),Math.min(.05,(performance.now()-(scopeSteering.lastUpdate||performance.now()))/1000));
    scopeSteering.lastUpdate=performance.now();
  }
  function rayFrom(i){
    const source=hands[i];if(!source)return null;
    source.updateMatrixWorld(true);
    tempMatrix.identity().extractRotation(source.matrixWorld);
    ray.ray.origin.setFromMatrixPosition(source.matrixWorld);
    ray.ray.direction.set(0,0,-1).applyMatrix4(tempMatrix).normalize();
    ray.near=0;ray.far=4;
    return ray;
  }
  function panelIntersection(r){
    if(!r)return null;
    const candidates=[];
    for(const panel of [
      {object:worldMenu,actions:actionHitbox,role:'menu'},
      {object:guidePlane,actions:guideHitboxes,role:'procedure-guide'},
      {object:compactDock,actions:railActions,role:'compact-command-rail'},
      {object:scopeScreen?.visible?monitorToolbar:null,actions:monitorActions,role:'monitor-controls'},
      {object:scopeScreen?.visible?monitorFrame:null,actions:[],role:'endoscopy-monitor'},
      {object:ebusScreen.visible?ebusToolbar:null,actions:ebusActions,role:'ebus-controls'},
      {object:ebusScreen.visible?ebusFrame:null,actions:[],role:'ebus-monitor'}
    ]){
      if(!panel.object?.visible)continue;
      panel.object.updateWorldMatrix(true,false);
      const surface=panel.object.userData.spatialSurface;
      const physical=surface?surface.hit(r):null;
      const hit=surface?physical?.hit:r.intersectObject(panel.object,false)[0];
      if(!hit?.uv||!Number.isFinite(hit.distance))continue;
      const x=hit.uv.x,y=1-hit.uv.y;
      const item=panel.actions.find(a=>surface?physical?.id===a.id:(x>=a.u&&x<=a.u+a.w&&y>=a.v&&y<=a.v+a.h));
      candidates.push({action:item&&!item.disabled?item.id:null,
        distance:hit.distance,point:hit.point.clone(),x,y,role:panel.role});
    }
    // Research image sections are true interactable 3D surfaces. Their
    // first-hit depth blocks controller rays from grabbing lung triangles
    // behind a visible CT plane. A pane is never an instrument target.
    if(ctSpatial?.snapshot().visible){
      const ctHit=r.intersectObject(ctSpatial.group,true)[0];
      if(ctHit&&Number.isFinite(ctHit.distance))
        candidates.push({action:null,distance:ctHit.distance,
          point:ctHit.point.clone(),role:'ct-source-section'});
    }
    return candidates.sort((a,b)=>a.distance-b.distance)[0]||null;
  }
  function choosePointer(i){
    const raycast=rayFrom(i);
    if(!raycast)return {panel:null,organ:null,choice:{kind:'none'}};
    const panel=panelIntersection(raycast);
    const organ=mode==='anatomy'
      ?global.PulmoXRRemoteTarget.pick(THREE,hands[i],surfaceTarget(),{maxDistance:3.5})
      :null;
    const choice=global.PulmoXRSpatialTarget.choose({panel,anatomy:organ,mode});
    return {panel,organ,choice};
  }
  function findPanelHit(r){return panelIntersection(r)?.action||null}
  function summon(){
    if(benchAligned)return;
    if(patientReturn)return;
    if(!viewerValid||mode!=='anatomy'||grab.some(Boolean))return;
    const bounds=sourceBounds();
    const targetScale=Math.min(mGrp.scale.x,.43/bounds.size.x,.50/bounds.size.y);
    const center=head.clone().addScaledVector(forward,.83).add(new THREE.Vector3(0,-.07,0));
    const dest=center.clone().sub(bounds.center.clone()
      .multiplyScalar(targetScale).applyQuaternion(mGrp.quaternion));
    summonTween={startedAt:performance.now(),durationMs:450,
      from:mGrp.position.clone(),to:dest,fromScale:mGrp.scale.x,toScale:targetScale};
    grabFeedback='SUMMONING SOURCE ANATOMY INTO REACH';paint();
  }
  function updateSummon(){
    if(!summonTween||!active||mode!=='anatomy'||grab.some(Boolean))return;
    const t=safe(0,1,(performance.now()-summonTween.startedAt)/summonTween.durationMs);
    const ease=1-Math.pow(1-t,3);
    mGrp.position.copy(summonTween.from).lerp(summonTween.to,ease);
    mGrp.scale.setScalar(summonTween.fromScale+(summonTween.toScale-summonTween.fromScale)*ease);
    constrainSizeAndFocus();
    if(t>=1)summonTween=null;
  }
  function pickFromScene(r){
    const targets=surfaceTarget();
    const hits=r.intersectObjects(targets,true).filter(h=>h.object.visible);
    if(!hits.length)return false;
    const obj=hits[0].object;
    const pathology=Object.values(ctx.pathoMeshes||{}).find(p=>p.mesh===obj||p.mesh.children.includes(obj));
    if(pathology)showToast('REFERENCE PATHOLOGY · '+pathology.label.split('\n')[0],'info');
    if(mode==='anatomy')summon();
    return true;
  }
  function confirmOrManualMR(){
    if(mrPlacement.snapshot().hitAvailable&&!mrQuality.snapshot().stableCandidate){grabFeedback='HOLD VIEW STEADY ON REAL SURFACE';paint();return false;}
    const result=mrPlacement.confirm();
    if(result.accepted){grabFeedback='SURFACE CONFIRMATION / ALIGNING NEXT FRAME';paint();return true;}
    // A deliberate trigger/pinch places a visibly unregistered model if
    // hit-testing is unsupported. Never equate it with a detected surface.
    const state=mrPlacement.snapshot().state;
    if(state==='unsupported'||state==='scanning'){
      mrPlacement.preview();mGrp.visible=true;frameGroup();
      grabFeedback='MANUAL MR PLACEMENT / UNREGISTERED';paint();return true;
    }
    grabFeedback='MR INITIALIZING / TRY AGAIN';paint();return false;
  }
  function onTrigger(i){
    const sideNow=controllerSide[i]||controllerSources[i]?.handedness;
    if(sideNow==='right'&&(spatialCal.snapshot().armed||performance.now()-calibrationCaptureAt<450))return;
    if(!active||!rayFrom(i)||panelEdit)return;
    const chosen=choosePointer(i);
    const side=controllerSide[i]||controllerSources[i]?.handedness||'unknown';
    if(chosen.choice.kind==='panel'){
      // The same nearest-hit decision is used by hover, grip and select.
      if(chosen.choice.action){
        attempts.menuActions++;
        haptic(i,.23,40);
        activate(chosen.choice.action);
      }else{grabFeedback='POINT AT A PANEL CONTROL';paint()}
      return;
    }
    if(mode==='anatomy'&&xrPresentation==='ar'&&!placed){
      confirmOrManualMR();return;
    }
    if(mode==='anatomy'){
      // The SELECT trigger is an actual hold-to-manipulate action.
      // A short unmoved tap summons the same original mesh within reach.
      startGrab(i,'controller','trigger');
    }else if(mode==='procedure'){
      const intent=deviceLab.triggerIntent({atBranch:scope.atBranch,side,tool:channel.snapshot().tool});
      if(intent==='BRANCH'){if(!accessoryDeployed()&&!probeLocked())enterBranch(side==='right'?Math.min(1,(SEG_MAP[scope.segId]?.children.length||1)-1):0);else{grabFeedback='RETRACT ACCESSORY BEFORE NAVIGATION';paint();}}
      else if(intent==='ACCESSORY'){actuateAccessory();paint();}
      else if(intent==='CLINICAL')fireAction();
    }
  }
  const modes=['nav','biopsy','measure','annotate'];
  function accessoryDeployed(){return channel.snapshot().extensionMm>0;}
  function probeLocked(){return ebusProbe.snapshot().shaftLocked;}
  function probeBasis(extension=ebusProbe.snapshot().extensionMm){const b=global.PulmoXRVideoScope.poseBasis(THREE,scope.dir,scope.roll,scope.deflH,scope.deflV,CFG.deflMax),mm=getMmPerSceneUnit()*(ctx.getReferenceMotionFactor?.()||1);return global.PulmoXREBUS.probePose({origin:scope.pos.toArray(),forward:b.direction.toArray(),right:b.right.toArray(),up:b.up.toArray(),extensionMm:extension,rollRad:ebusProbe.snapshot().rollRad,mmPerUnit:mm});}
  function probeClearance(extension){try{return global.PulmoXREBUS.clearance({pose:probeBasis(extension),extensionMm:extension,field:ctx.getLumenField?.(),mmPerUnit:getMmPerSceneUnit()*(ctx.getReferenceMotionFactor?.()||1)});}catch{return false;}}
  function feedProbe(delta){if(delta>0&&!patientMotionAllowed())return false;if(mode!=='procedure'||getMeshMode()!=='reference'||channel.snapshot().tool!=='none'||physicalControl?.snapshot().armed)return false;const ok=ebusProbe.feed(delta,probeClearance);if(probeLocked()){scope.velocity=0;scopeSteering=null;}radialLast=-Infinity;return ok;}
  function driveProbe(point,closed,valid,dt,q){const local=point?new THREE.Vector3(...point).sub(videoScope.group.position).applyQuaternion(q.clone().invert()).toArray():null;const near=Boolean(local&&Math.hypot(local[0]-.10,local[1]+.012,local[2])<.065);const p=probeGrip.step({valid,closed,near,point:local,axis:[-1,0,0],dt});if(p.held&&p.deltaMm)feedProbe(p.deltaMm);}
  function selectEBUS(next){if(mode!=='procedure'||getMeshMode()!=='reference'||physicalControl?.snapshot().armed||channel.snapshot().tool!=='none'){grabFeedback='EMPTY WORKING CHANNEL AND DISARMED REFERENCE REQUIRED';return false;}if(!ebusProbe.select(next)){grabFeedback='RETRACT PROBE BEFORE MODE CHANGE';return false;}radialEnabled=next!=='off';radialLast=-Infinity;probeGrip.reset();ebusScreen.visible=radialEnabled;if(!radialEnabled){panelPinned.delete(ebusScreen);if(panelTarget==='ebus')panelTarget='monitor';ultrasoundTower.hide();}positionPanels();return true;}
  function checkAccessory(length,open){
    const state=channel.snapshot();
    const frame=ctx.getReferenceMotionFrame?.()||mGrp;
    frame.updateWorldMatrix(true,false);camScope.updateMatrixWorld(true);
    const worldPerMm=frame.getWorldScale(new THREE.Vector3()).x/getMmPerSceneUnit();
    return upgrade.accessoryClearance({lengthMm:length,diameterMm:state.odMm,worldPerMm,sourceMmPerUnit:getMmPerSceneUnit(),open,field:ctx.getLumenField?.(),toSource:p=>frame.worldToLocal(new THREE.Vector3(...p).applyMatrix4(camScope.matrixWorld)).toArray()});
  }
  function actuateAccessory(){const state=channel.snapshot();if(state.tool==='forceps'&&!state.jawsOpen&&state.extensionMm>=3){accessoryEvidence=checkAccessory(state.extensionMm,true);if(!accessoryEvidence.allowed){grabFeedback='ACCESSORY CLEARANCE / '+accessoryEvidence.state;return;}}channel.action();}
  function activate(action){
    grabFeedback='MENU SELECTED / '+String(action||'').toUpperCase();
    if(active){for(const surface of [mainSpatialSurface,railSurface,guideSpatialSurface,monitorControlSurface,ebusControlSurface])surface.press(action,performance.now());}
    if(action==='console-collapse'){worldMenu.visible=false;compactDock.visible=true;hoverPanels.fill(null);stopPanelEdit();positionPanels();paint();return;}
    if(action==='console-reading'){readingMode=!readingMode;panelPinned.delete(worldMenu);panelPinned.delete(guidePlane);positionPanels();contextActions++;paint();return;}
    if(action==='dock-channel'||action==='monitor-station'){if(mode!=='procedure')return;menuPage=action==='dock-channel'?'channel':'station';actionHitbox=panelLayouts[menuPage];worldMenu.visible=true;compactDock.visible=false;positionPanels();contextActions++;paint();return;}
    if(action==='dock-guide'||action==='monitor-guide'){if(mode!=='procedure')return;guideVisible=!guideVisible;guidePlane.visible=guideVisible;positionPanels();contextActions++;paint();return;}
    if(action==='monitor-focus'){if(mode!=='procedure')return;monitorFocus=!monitorFocus;panelPinned.delete(scopeScreen);panelPinned.delete(guidePlane);positionPanels();contextActions++;paint();return;}
    if(action==='dock-open'){worldMenu.visible=true;compactDock.visible=false;positionPanels();paint();return;}
    if(action==='dock-mode'){setMode(mode==='anatomy'?'procedure':'anatomy');paint();return;}
    if(action==='dock-recenter'){if(xrPresentation==='ar'){worldMenu.visible=true;menuPage='mr';actionHitbox=panelLayouts.mr;positionPanels();}else frameGroup();paint();return;}
    if(action==='dock-exit'){void rBro.xr.getSession()?.end();return;}
    if(action==='more'||action==='back'||action==='precision'||action==='back-expert'){stopPanelEdit();
      menuPage=action==='more'||action==='back-expert'?'advanced':
        action==='precision'?(mode==='procedure'?'station':'precision'):'primary';
      actionHitbox=panelLayouts[menuPage];hoverPanels.fill(null);
      paint();return;
    }
    if(action==='mr-confirm'&&xrPresentation==='ar'){
      if(!mrQuality.snapshot().stableCandidate){grabFeedback='HOLD VIEW STEADY ON REAL SURFACE';paint();return;}
      const result=mrPlacement.confirm();
      grabFeedback=result.accepted?'SURFACE SELECTED / CONFIRMING':
        'NO REAL SURFACE HIT / SELECT UNREGISTERED PREVIEW';
      paint();return;
    }
    if(action==='mr-reposition'&&xrPresentation==='ar'){
      invalidateBench();clearSpatialCalibration();
      release(0);release(1);scope.velocity=0;
      placed=false;mGrp.visible=false;mrAnchorFault=false;mrQuality.reset();mrPlacement.reposition();
      if(mrReticle)mrReticle.visible=false;
      grabFeedback='MR SCAN / AIM HEADSET AT A REAL SURFACE';
      paint();return;
    }
    if(action==='mr-preview'&&xrPresentation==='ar'){
      invalidateBench();
      mrAnchorFault=false;mrQuality.reset();mrPlacement.preview();mGrp.visible=true;
      frameGroup();grabFeedback='UNREGISTERED FLOATING PREVIEW / NOT ROOM ANCHORED';
      paint();return;
    }
    if(action==='mr-status'&&xrPresentation==='ar'){
      grabFeedback='MR / '+mrPlacement.snapshot().state.toUpperCase()+
        ' / '+mrPlacement.snapshot().anchorStatus.toUpperCase()+' / PLANES '+environmentEvidence.planes;
      paint();return;
    }
    if(action==='guide'&&mode==='anatomy'){menuPage='section';actionHitbox=panelLayouts.section;paint();return;}
    if(action==='inspect'){menuPage='inspection';actionHitbox=panelLayouts.inspection;paint();return;}
    if(action==='inspect-toggle'){if(inspectMode&&inspectionUnit==='lung'){inspectionUnit='lobe';spreadLobes('lobe');}else{inspectionUnit='lobe';setInspection(!inspectMode);}paint();return;}
    if(action==='inspect-spread'){spreadLobes('lobe');paint();return;}
    if(action==='inspect-lungs'){spreadLobes('lung');paint();return;}
    if(action==='inspect-restore'){setInspection(false);paint();return;}
    if(action==='inspect-isolate'){filterLobes();paint();return;}
    if(action==='section-open'){menuPage='section';actionHitbox=panelLayouts.section;paint();return;}
    if(action==='section-back'){clearSection();menuPage='inspection';actionHitbox=panelLayouts.inspection;paint();return;}
    if(action==='section-toggle'){toggleSection();paint();return;}
    if(action==='section-axis'){sectionAxis=['x','y','z'][(['x','y','z'].indexOf(sectionAxis)+1)%3];updateSection();paint();return;}
    if(action==='section-prev'||action==='section-next'){sectionFraction=safe(.05,.95,sectionFraction+(action==='section-next'?.1:-.1));updateSection();paint();return;}
    if(action==='panel-shift'){panelPinned.delete(worldMenu);panelOffset=panelOffset===0?1.44:0;positionPanels();paint();return;}
    if(action==='panel-edit-open'){panelTarget=mode==='procedure'?'monitor':'menu';menuPage='panel-edit';actionHitbox=panelLayouts['panel-edit'];paint();return;}
    if(action==='panel-select'){const targets=mode==='procedure'?(radialEnabled?['monitor','ebus','guide','menu']:['monitor','guide','menu']):['menu'];panelTarget=targets[(targets.indexOf(panelTarget)+1)%targets.length];paint();return;}
    if(action==='panel-size'){const object=selectedPanel();if(object?.visible){object.scale.setScalar(object.scale.x>=1.25?.75:Math.min(1.3,object.scale.x+.15));panelPinned.add(object);}paint();return;}
    if(action==='panel-reset'){stopPanelEdit();panelPinned.clear();panelOffset=0;positionPanels();paint();return;}
    if(action==='panel-compose'){if(panelEdit)return;positionPanels();grabFeedback='WORKSPACE ARRANGED / PINNED WINDOWS RETAINED';paint();return;}
    if(action==='regional-toggle'){regionalDisplay=!regionalDisplay;regionalLast=-Infinity;paint();return;}
    if(action==='workspace-open'){menuPage='workspace';actionHitbox=panelLayouts.workspace;paint();return;}
    if(action==='workspace-options'){menuPage='workspace-options';actionHitbox=panelLayouts['workspace-options'];paint();return;}
    if(action.startsWith('workspace-')){moveWorkspace(action.slice(10));paint();return;}
    if(action==='imaging-open'){menuPage='imaging';actionHitbox=panelLayouts.imaging;paint();return;}
    if(action==='imaging-back'){menuPage='channel';actionHitbox=panelLayouts.channel;paint();return;}
    if(action==='radial-toggle'){selectEBUS(ebusProbe.snapshot().mode==='off'?'radial':'off');paint();return;}
    if(action==='ebus-mode'){selectEBUS(ebusProbe.snapshot().mode==='convex'?'radial':'convex');paint();return;}
    if(action==='ebus-open'){menuPage='ebus';actionHitbox=panelLayouts.ebus;paint();return;}
    if(action==='ebus-feed'||action==='ebus-retract'){feedProbe(action==='ebus-feed'?2:-2);grabFeedback=ebusProbe.snapshot().status;paint();return;}
    if(action==='ebus-roll'){ebusProbe.rotate(Math.PI/12);radialLast=-Infinity;paint();return;}
    if(action==='radial-coupling'){radialCoupling=radialCoupling==='air'?'ideal-fluid':'air';radialLast=-Infinity;paint();return;}
    if(action==='radial-range'){radialRange=radialRange===20?10:radialRange===10?30:20;radialLast=-Infinity;paint();return;}
    if(action==='motion-toggle'){motionEnabled=!motionEnabled;if(!motionEnabled)ctx.setReferenceMotion?.(1);paint();return;}
    if(action==='channel-open'){menuPage='channel';actionHitbox=panelLayouts.channel;paint();return;}
    if(action==='channel-back'){menuPage='station';actionHitbox=panelLayouts.station;paint();return;}
    if(action==='optics-toggle'){if(depthOpticsAvailable){opticsEnabled=!opticsEnabled;lastFeed=0;}else grabFeedback='DEPTH OPTICS UNAVAILABLE / RAW SOURCE';paint();return;}
    if(action==='channel-cycle'){if(ebusProbe.snapshot().mode!=='off'){grabFeedback='STOW EBUS BEFORE CHANGING CHANNEL TOOL';paint();return;}if(mode==='procedure'&&getMeshMode()==='reference'){channel.cycle();accessoryEvidence={state:'NOT_CHECKED'};}paint();return;}
    if(action==='channel-extend'||action==='channel-retract'){if(ebusProbe.snapshot().mode!=='off'){grabFeedback='EBUS ACTIVE / CHANNEL RESERVED';paint();return;}if(mode==='procedure'&&getMeshMode()==='reference'){if(action==='channel-retract')channel.extend(-2);else{const state=channel.snapshot();accessoryEvidence=checkAccessory(Math.min(12,state.extensionMm+2),state.jawsOpen);if(accessoryEvidence.allowed){channel.extend(2);scope.velocity=0;scopeSteering=null;if(physicalControl?.snapshot().armed)physicalControl.stop('ACCESSORY_DEPLOYED');}else grabFeedback='ACCESSORY CLEARANCE / '+accessoryEvidence.state;}}paint();return;}
    if(action==='channel-action'){if(ebusProbe.snapshot().mode!=='off')return;if(mode==='procedure'&&getMeshMode()==='reference')actuateAccessory();paint();return;}
    if(action==='device-visible'){deviceVisible=!deviceVisible;paint();return;}
    if(action==='airway-map'){routeVisible=!routeVisible;updateRouteOverlay();paint();return;}
    if(action==='optical-light'){opticalIndex=(opticalIndex+1)%opticalModes.length;const p=opticalModes[opticalIndex];ctx.setEndoscopeLighting?.(p.gain,p.glare);grabFeedback='ENDOSCOPE LIGHT / '+p.name;lastFeed=0;paint();return;}
    if(action==='view'&&xrPresentation==='ar'&&mode==='anatomy'){
      menuPage='mr';actionHitbox=panelLayouts.mr;hoverPanels.fill(null);
      paint();return;
    }
    if(action==='view'&&mode==='procedure'){
      menuPage='station';actionHitbox=panelLayouts.station;hoverPanels.fill(null);
      paint();return;
    }
    if(action==='view'){
      const modeNext=stage.snapshot().mode==='studio'?'focus':'studio';
      stage.setMode(modeNext);stage.root.getObjectByName('xr-architectural-datums').visible=false;laboratory.root.visible=modeNext==='studio';if(patientReturn)plinth.visible=false;chamber.visible=modeNext==='focus';floor.material.opacity=modeNext==='focus'?.25:0;
      grabFeedback=modeNext==='focus'?'FOCUS WORKSPACE / MINIMAL CONTEXT':
        'SPATIAL STUDIO / ARCHITECTURAL CONTEXT';
      paint();return;
    }
    if(action==='tool-proxy'){
      const current=toolProxy.snapshot().enabled;
      toolProxy.set(!current);
      grabFeedback=!current?
        '6DOF CONTROLLER PROXY / NOT A CALIBRATED INSTRUMENT':
        'CONTROLLER PROXY HIDDEN';
      paint();return;
    }
    if(action==='trace-clear'){
      replay.clear();
      traceButton.textContent='LOAD INSTRUMENT TRACE';
      grabFeedback='LOCAL MEASUREMENT TRACE CLEARED';
      paint();return;
    }
    if(action==='body-toggle'){bodyEnabled=!bodyEnabled;paint();return;}
    if(action==='hand-procedure'){handProcedure=!handProcedure;resetEmbodiedGrip();scope.velocity=0;grabFeedback=handProcedure?'BARE-HAND MODE / OPEN HANDS THEN PINCH HANDLE':'CONTROLLER MODE';paint();return;}
    if(action==='spatial-open'){menuPage='spatial-cal';actionHitbox=panelLayouts['spatial-cal'];paint();return;}
    if(action==='spatial-pivot'||action==='spatial-jig'){if(mode!=='anatomy'||!viewerValid){grabFeedback='CALIBRATE IN ANATOMY MODE';paint();return;}try{invalidateBench();release(0);release(1);spatialCal.begin(action==='spatial-pivot'?'pivot':'jig');showPassiveProxy=true;grabFeedback='ARM CAPTURE / RIGHT TRIGGER AT LANDMARK';}catch(e){grabFeedback=e.message;}paint();return;}
    if(action==='spatial-capture'){spatialCal.arm();paint();return;}
    if(action==='spatial-align'){alignBench();paint();return;}
    if(action==='spatial-proxy'){showPassiveProxy=!showPassiveProxy;paint();return;}
    if(action==='spatial-planes'){planeMaskEnabled=xrPresentation==='ar'&&!planeMaskEnabled;if(!planeMaskEnabled)planeMask.clear();paint();return;}
    if(action==='spatial-clear'){invalidateBench();clearSpatialCalibration();grabFeedback='CALIBRATION CLEARED / UNREGISTERED';paint();return;}
    if(action==='spatial-viewpoint'){moveWorkspace('viewpoint');paint();return;}
    if(action==='spatial-back'){menuPage=xrPresentation==='ar'?'mr':'advanced';actionHitbox=panelLayouts[menuPage];paint();return;}
    if(action==='patient-open'){menuPage='patient';actionHitbox=panelLayouts.patient;paint();return;}
    if(action==='patient-toggle'){if(patientReturn){if(probeLocked()||accessoryDeployed()){grabFeedback='RETRACT ACCESSORY BEFORE LEAVING STATION';paint();return;}leavePatientStation();grabFeedback='REFERENCE EXPLORATION RESTORED';}else enterPatientStation();paint();return;}
    if(action==='patient-view'){if(patientReturn){const modes=['external','transparent','internal'];patientView=modes[(modes.indexOf(patientView)+1)%3];patientStation.setView(patientView);}paint();return;}
    if(action==='patient-back'){menuPage='station';actionHitbox=panelLayouts.station;paint();return;}
    if(action==='research-open'){
      menuPage='research';actionHitbox=panelLayouts.research;hoverPanels.fill(null);grabFeedback='READ-ONLY LOCAL SNAPSHOT / REFERENCE SCOPE UNCHANGED';paint();return;
    }
    if(action==='research-clear'){
      researchLoadEpoch++;researchSnapshot=null;researchCanvas=null;grabFeedback='RESEARCH SNAPSHOT CLEARED';paint();return;
    }
    if(action==='session-review-open'){menuPage='session-review';actionHitbox=panelLayouts['session-review'];paint();return;}
    if(action==='record-start'){if(mode!=='procedure'||xrPresentation!=='vr'||getMeshMode()!=='reference'||!viewerValid){grabFeedback='REFERENCE VR PROCEDURE REQUIRED';paint();return;}if(localRecording.snapshot().sampleCount&&!localRecording.snapshot().running){grabFeedback='EXPORT OR CLEAR PREVIOUS RECORDING FIRST';paint();return;}if(!localRecording.snapshot().running)localRecording.start(lastFrame||performance.now());grabFeedback='LOCAL MOTION RECORDING ACTIVE';paint();return;}
    if(action==='record-stop'){stopRecording();const p=localRecording.packet();if(p)loadRecordedReview(p);paint();return;}
    if(action==='record-clear'){reviewEpoch++;stopRecording('CLEARED');localRecording.clear();loadRecordedReview(null);paint();return;}
    if(action==='record-share'){if(localShare)stopLocalShare();else beginLocalShare();paint();return;}
    if(action==='review-prev'||action==='review-next'){stepReview(action==='review-next'?10:-10);paint();return;}
    if(action==='review-play'){if(reviewPacket){reviewPlaying=!reviewPlaying;reviewStart=lastFrame||performance.now();reviewBase=reviewPacket.samples[reviewIndex].t_ms;if(reviewIndex===reviewPacket.samples.length-1){reviewIndex=0;reviewBase=reviewPacket.samples[0].t_ms;reviewEngine.draw(reviewCanvas,reviewPacket.samples,0);}}paint();return;}
    if(action==='review-back'){menuPage=mode==='procedure'?'station':'research';actionHitbox=panelLayouts[menuPage];paint();return;}
    if(action==='haptic-open'){menuPage='haptic-lab';actionHitbox=panelLayouts['haptic-lab'];paint();return;}
    if(action==='haptic-cue'){cueEngine.set(!cueEngine.snapshot().enabled);paint();return;}
    if(action==='haptic-arm'){if(mode!=='procedure'||xrPresentation!=='vr'||!physicalControl?.snapshot().armed||!sensorObserver.snapshot(performance.now()).motion){grabFeedback='FRESH CALIBRATED LOCAL INSTRUMENT REQUIRED';paint();return;}hapticDryRun.arm();grabFeedback='LOCAL DRY RUN / RELEASE LEFT GRIP FIRST';paint();return;}
    if(action==='haptic-stop'){hapticDryRun.stop();paint();return;}
    if(action==='haptic-back'){menuPage='trace';actionHitbox=panelLayouts.trace;paint();return;}
    if(action==='trace-open'){
      replay.pause(performance.now());
      menuPage='trace';actionHitbox=panelLayouts.trace;hoverPanels.fill(null);
      grabFeedback=replay.snapshot().loaded?'EXTERNAL TRACE / READ-ONLY INSPECTION':
        'IMPORT A MEASURED TRACE BEFORE ENTERING XR';
      paint();return;
    }
    if(action==='trace-back'){
      replay.pause(performance.now());
      menuPage='station';actionHitbox=panelLayouts.station;hoverPanels.fill(null);
      paint();return;
    }
    if(action==='trace-play'||action==='trace-pause'||
       action==='trace-prev'||action==='trace-next'){
      if(!replay.snapshot().loaded){
        grabFeedback='NO INDEPENDENT MEASURED TRACE LOADED';
        paint();return;
      }
      if(action==='trace-play')replay.play(performance.now());
      if(action==='trace-pause')replay.pause(performance.now());
      if(action==='trace-prev')replay.step(-1);
      if(action==='trace-next')replay.step(1);
      grabFeedback='READ-ONLY TRACE / ORIGINAL BRONCHOSCOPE UNCHANGED';
      paint();return;
    }
    if(action==='display-path'){
      const evidence=nativeLayersOptIn?(nativeQuad?.snapshot()||displayPath.snapshot()):{actualDisplayPath:'SAME_CONTEXT_OFFSCREEN_WEBGL_RENDER_TARGET',reason:'PHYSICAL_ACCEPTANCE_COMPATIBILITY_MODE'};
      grabFeedback='DISPLAY / '+evidence.actualDisplayPath+' / '+evidence.reason;
      showToast('XR display: '+evidence.reason+' · safe source-image fallback','info');
      paint();return;
    }
    if(action==='station-layout'||action==='station-video'||
       action==='station-optics'||action==='station-control'){
      const key=action.split('-')[1];
      station.next(key==='video'?'video':key);
      const settings=station.snapshot();
      if(key==='layout')positionPanels();
      if(key==='video'){
        scopeTarget.setSize(settings.capture.width,settings.capture.height);
        // Explicit video changes also re-arm a failed renderer and reset the
        // previous setting's CPU-cost samples.
        monitorStatus='not-started';monitorRetryAt=0;lastFeed=0;
        monitorBudget=instrument.videoCostGovernor();
      }
      if(key==='optics'){
        camScope.fov=settings.fovDegrees;camScope.updateProjectionMatrix();
      }
      grabFeedback='PROCEDURE STATION / '+key.toUpperCase()+' UPDATED';
      paint();return;
    }
    if(action==='undo'||action==='redo'){
      if(!stepViewHistory(action))grabFeedback='NO '+action.toUpperCase()+' VIEW AVAILABLE';
      paint();return;
    }
    if(action==='pivot'){
      pivotMode=pivotMode==='center'?'origin':'center';
      grabFeedback='VIEW ROTATION PIVOT / '+pivotMode.toUpperCase();paint();return;
    }
    if(action==='axis'){
      const axisList=['yaw','pitch','roll','locked'];
      axisMode=axisList[(axisList.indexOf(axisMode)+1)%axisList.length];
      orientationLocked=axisMode==='locked';
      orientationAnchor=orientationLocked?mGrp.quaternion.clone():null;
      grabFeedback=orientationLocked?'ORIENTATION LOCKED / POSITION AND SCALE STILL ACTIVE':
        'VIEW ROTATION AXIS / '+axisMode.toUpperCase();
      paint();return;
    }
    if(action==='ct-axis'){
      ctAxis=['axial','coronal','sagittal'][
        (['axial','coronal','sagittal'].indexOf(ctAxis)+1)%3];
      paint();return;
    }
    if(action==='ct-prev'||action==='ct-next'){
      if(ctSpatial)ctSpatial.step(ctAxis,action==='ct-next'?1:-1);
      paint();return;
    }
    if(action==='ct-sections'){
      if(ctSpatial)ctSpatial.setVisible(!ctSpatial.snapshot().visible);
      paint();return;
    }
    if(action==='anatomy')setMode('anatomy');
    if(action==='procedure')setMode('procedure');
    if(action==='reset')frameGroup();
    if(action==='scale'){
      const study=getCTStudy?.();
      if(study&&getMeshMode()==='patient'){
        try{
          if(!ctSpatial)ctSpatial=global.PulmoXRCTSpatial.create(THREE,mGrp,study);
          ctSpatial.setVisible(true);ctAxis='axial';menuPage='ct';
          actionHitbox=panelLayouts.ct;
          grabFeedback='VERIFIED CT VOXELS / MPR INSPECTION, NOT SEGMENTED TISSUE';
        }catch(error){showToast('Spatial CT unavailable: '+String(error.message||error),'warn')}
      }else if(!setSourceScale()){
        showToast('Source scale requires eligible original co-framed metadata','warn');
      }
    }
    if(action==='tool')setClinMode(modes[(modes.indexOf(getClinMode())+1)%modes.length]);
    if(action==='guide'){
      guideVisible=!guideVisible;
      if(guidePlane)guidePlane.visible=mode==='procedure'&&guideVisible;positionPanels();
    }
    if(action==='guide-hide'){guideVisible=false;if(guidePlane)guidePlane.visible=false;positionPanels()}
    if(mode==='procedure'&&!accessoryDeployed()&&!probeLocked()&&!physicalControl?.snapshot().armed&&action==='scope-forward'){advance(.045);scope.velocity=0;}
    if(mode==='procedure'&&!accessoryDeployed()&&!probeLocked()&&!physicalControl?.snapshot().armed&&action==='scope-back'){retract(.045);scope.velocity=0;}
    if(action==='guide-next')guideStep=Math.min(4,guideStep+1);
    if(action==='guide-prev')guideStep=Math.max(0,guideStep-1);
    if(action==='guide-tool'){
      setClinMode(modes[(modes.indexOf(getClinMode())+1)%modes.length]);
      guideActions++;
    }
    if(action==='guide-branch0'||action==='guide-branch1'||action==='guide-branch2'){
      const index=Number(action.slice(-1));
      const current=global.PulmoXRProcedureGuide.describe({
        segmentId:scope.segId,segment:SEG_MAP[scope.segId],segments:SEG_MAP,
        position:scope.t,atBranch:scope.atBranch,clinicalMode:getClinMode(),
        clearance:clearanceState(),monitorStatus,guideStep});
      if(current.branches[index]?.enabled&&!accessoryDeployed()&&!probeLocked()){
        enterBranch(current.branches[index].index);
        guideActions++;
      }
    }
    if(action==='exit')rBro.xr.getSession()?.end().catch(()=>{});
    paint();
  }
  const mapButtons=[[],[]];
  function toggleSpatialPanel(i){
    if(!active||!viewerValid)return false;
    if(worldMenu.visible){
      worldMenu.visible=false;hoverPanels.fill(null);
      grabFeedback='SPATIAL CONTROLS HIDDEN / LEFT Y TO SHOW';
    }else{
      worldMenu.visible=true;
      grabFeedback='SPATIAL CONTROLS READY / AIM AND TRIGGER';
    }
    compactDock.visible=active&&viewerValid&&!worldMenu.visible;positionPanels();
    attempts.panelToggles++;
    haptic(i,.20,36);paint();return true;
  }
  function applyAnalogAnatomy(side,x,y,dt){
    if(benchAligned||spatialCal.snapshot().armed)return;
    if(mode!=='anatomy'||!placed||!boundsCache||grab.some(Boolean))return;
    const center=pivotMode==='center'?focusWorld():mGrp.position.clone();
    let modified=false;
    if(side==='right'){
      // Right stick: yaw around the visible anatomy centre and move along
      // the viewer's depth axis. It does not move the camera or the user.
      if(Math.abs(x)>.02&&!orientationLocked){
        if(!analogChangedAt)beginViewChange();
        const axes={yaw:new THREE.Vector3(0,1,0),
          pitch:new THREE.Vector3(1,0,0),roll:new THREE.Vector3(0,0,1)};
        mGrp.quaternion.premultiply(
          new THREE.Quaternion().setFromAxisAngle(axes[axisMode],-x*dt*1.35));
        modified=true;
      }
      if(pivotMode==='center')mGrp.position.copy(center).sub(boundsCache.center.clone()
        .multiplyScalar(mGrp.scale.x).applyQuaternion(mGrp.quaternion));
      if(Math.abs(y)>.02){
        if(!analogChangedAt)beginViewChange();
        mGrp.position.addScaledVector(forward,-y*dt*.76);
        modified=true;
      }
    }else if(side==='left'&&Math.abs(y)>.02){
      if(!analogChangedAt)beginViewChange();
      const next=limits.clampScale(mGrp.scale.x*Math.exp(-y*dt*.92),modelExtent());
      mGrp.scale.setScalar(next);
      if(pivotMode==='center')mGrp.position.copy(center).sub(boundsCache.center.clone()
        .multiplyScalar(next).applyQuaternion(mGrp.quaternion));
      modified=true;
    }
    if(modified){
      constrainSizeAndFocus();attempts.thumbstickFrames++;
      analogChangedAt=performance.now();
    }
  }
  function poll(dt){
    const session=rBro.xr.getSession();if(!session)return;
    let leftControllerAvailable=false;
    for(const src of session.inputSources){
      if(!src.gamepad)continue;
      // Gamepad commands depend on handedness, not a connected-event slot.
      const mapped=controllerSources.indexOf(src);
      const i=mapped>=0?mapped:src.handedness==='left'?0:src.handedness==='right'?1:-1;
      if(i<0)continue;
      // Input-source array slots are connection order, NOT left/right.
      const side=src.handedness||controllerSide[i];
      if(side==='left')leftControllerAvailable=true;
      const buttons=src.gamepad.buttons||[];
      const edge=idx=>{const pressed=!!buttons[idx]?.pressed,prior=!!mapButtons[i][idx];mapButtons[i][idx]=pressed;return pressed&&!prior};
      // Quest Touch xr-standard: trigger[0], grip[1]. Polling backs up
      // WebXR select/squeeze events, but cannot bypass triangle targeting.
      if(buttons[0]){
        if(edge(0)&&!grab[i]){if(side==='right'&&spatialCal.snapshot().armed){calibrationCaptureAt=performance.now();if(calibrationPose){spatialCal.capture(calibrationPose);grabFeedback=spatialCal.snapshot().state.replaceAll('_',' ');}else grabFeedback='RIGHT CONTROLLER NOT TRACKED / RETRY';paint();}else onTrigger(i);}
        if(!buttons[0].pressed&&grab[i]?.input==='trigger')release(i,'controller','trigger');
      }
      if(spatialCal.snapshot().armed)continue;
      if(buttons[1]){
        const pressed=!!buttons[1].pressed;
        if(pressed&&!squeezeHeld[i]&&!grab[i]&&!scopeSteering)
          startGrab(i,'controller','grip');
        if(!pressed&&squeezeHeld[i])release(i,'controller','grip');
        squeezeHeld[i]=pressed;
      }
      const axes=Array.from(src.gamepad.axes||[]);
      const n=axes.length>=4?2:0;
      const ax=Number.isFinite(axes[n])?axes[n]:0,ay=Number.isFinite(axes[n+1])?axes[n+1]:0;
      const fallback=n===2&&Math.abs(ax)<.13&&Math.abs(ay)<.13;
      const x=instrument.inputAxis(fallback?axes[0]:ax),y=instrument.inputAxis(fallback?axes[1]:ay);
      if(side==='right'&&edge(4))setMode(mode==='anatomy'?'procedure':'anatomy');
      if(side==='left'&&edge(4))frameGroup();
      // Quest left-controller Y: restore/hide the spatial menu. This avoids
      // a permanent HUD and works from either anatomy or bronchoscopy.
      if(side==='left'&&edge(5))toggleSpatialPanel(i);
      if(mode==='anatomy'){
        if(!panelEdit)applyAnalogAnatomy(side,x,y,dt);
        continue;
      }
      if(mode!=='procedure')continue;
      if(!patientMotionAllowed()||accessoryDeployed()||probeLocked()||ebusProbe.snapshot().mode==='radial'||panelEdit){scope.velocity=0;continue;}
      if(physicalControl?.snapshot().armed)continue;
      if(side==='left'){
        // Insertion and shaft roll are independent. Neutral stick stops
        // insertion immediately; no unintended scope drift after release.
        scope.velocity=instrument.velocity(y,CFG.scopeSpd,station.snapshot().insertionMultiplier);
        scope.roll=instrument.rollStep(scope.roll,x,dt);
      }else if(side==='right'&&!scopeSteering){
        // Tip bending is incremental and holds when the stick returns to
        // centre. The original scope state and lumen gate remain in charge.
        scope.deflH=instrument.tipStep(scope.deflH,x,dt);
        scope.deflV=instrument.tipStep(scope.deflV,-y,dt);
      }
      if(side==='right'&&edge(5))activate('tool');
    }
    // A vanished controller must not leave the original scope advancing.
    if(mode==='procedure'&&!leftControllerAvailable)scope.velocity=0;
  }
  // World-space architectural VR shell, independent of the anatomical source.
  // Fixed when the session starts: viewer head motion creates genuine parallax
  // and presence. No fake lung geometry, HUD skyboxes or computer-room overlay.
  const room=new THREE.Group();room.visible=false;scene.add(room);
  // Quiet, static vertical luminance gradient for architectural room depth.
  // Vertex colours are one mesh and one draw call; no blur, bloom or sky HDR.
  const chamberGeometry=new THREE.SphereGeometry(8,32,16);
  const heights=chamberGeometry.attributes.position,tones=[];
  const floorTone=new THREE.Color(0x070e14),horizonTone=new THREE.Color(0x182630);
  const ceilingTone=new THREE.Color(0x090f16);
  for(let i=0;i<heights.count;i++){
    const height=heights.getY(i);
    const tone=height<0
      ?floorTone.clone().lerp(horizonTone,Math.min(1,Math.max(0,(height+4)/4)))
      :horizonTone.clone().lerp(ceilingTone,Math.min(1,height/6));
    tones.push(tone.r,tone.g,tone.b);
  }
  chamberGeometry.setAttribute('color',new THREE.Float32BufferAttribute(tones,3));
  const chamber=new THREE.Mesh(chamberGeometry,
    new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.BackSide,
      depthWrite:false,fog:false}));
  chamber.position.y=1.22;
  chamber.renderOrder=-100;room.add(chamber);
  const floor=new THREE.Mesh(new THREE.CircleGeometry(3.8,64),
    new THREE.MeshStandardMaterial({color:0x14242e,roughness:1,metalness:0,
      transparent:true,opacity:1}));
  floor.rotation.x=-Math.PI/2;room.add(floor);
  const grid=new THREE.GridHelper(7.2,20,0x4e6671,0x263843);
  grid.material.transparent=true;grid.material.opacity=.24;
  grid.position.y=.003;room.add(grid);
  const ringMaterial=new THREE.MeshBasicMaterial({
    color:0x51727f,transparent:true,opacity:.19,depthWrite:false});
  const spatialRings=[];
  for(const radius of [1.3,2.3,3.4]){
    const ring=new THREE.Mesh(new THREE.RingGeometry(radius,radius+.006,64),
      ringMaterial);
    ring.rotation.x=-Math.PI/2;ring.position.y=.012;room.add(ring);
    spatialRings.push(ring);
  }
  const patientStation=global.PulmoXRPatient.createVisual(THREE,scene);
  let patientReturn=null,patientZone={allowed:false,state:'INACTIVE'},patientView='external',patientZoneLast=null;
  function patientMotionAllowed(){return !patientReturn||Boolean(viewerValid&&patientStation.zone(head.toArray()).allowed);}
  function advance(amount){if(!patientMotionAllowed()){scope.velocity=0;return;}const prior=scope.t,id=scope.segId;const r=sourceAdvance(amount);if(amount>0){lastAdvanceAt=performance.now();lastAdvanceBlocked=scope.segId===id&&Math.abs(scope.t-prior)<1e-7&&!scope.atBranch;}return r;}
  function retract(amount){lastAdvanceAt=-Infinity;lastAdvanceBlocked=false;if(!patientMotionAllowed()){scope.velocity=0;return;}return sourceRetract(amount);}
  function enterBranch(index){if(!patientMotionAllowed())return;return sourceEnterBranch(index);}
  function leavePatientStation(){
    if(!patientReturn)return;const old=patientReturn;patientReturn=null;
    mGrp.position.copy(old.position);mGrp.quaternion.copy(old.quaternion);mGrp.scale.copy(old.scale);mGrp.updateMatrixWorld(true);scaleMode=old.scaleMode;
    plinth.visible=old.plinth;patientStation.hide();patientZone={allowed:false,state:'INACTIVE'};patientZoneLast=null;resetEmbodiedGrip();scope.velocity=0;updateScope();
  }
  function enterPatientStation(){
    if(!active||!viewerValid||xrPresentation!=='vr'||mode!=='procedure'||getMeshMode()!=='reference'||accessoryDeployed()||probeLocked()||physicalControl?.snapshot().armed){grabFeedback='REFERENCE VR PROCEDURE / RETRACT AND DISARM FIRST';return false;}
    const root=pathCurves[deviceLab.ancestry(SEG_MAP,scope.segId)[0]],scale=getMmPerSceneUnit()/1000;
    if(!root||!Number.isFinite(scale)||scale<=0||scale>.5){grabFeedback='REFERENCE GEOMETRY UNAVAILABLE';return false;}
    const spec=global.PulmoXRPatient.placement({head:head.toArray(),forward:forward.toArray(),floor:room.position.y});
    if(!global.PulmoXRPatient.operatorZone(spec,head.toArray()).allowed){grabFeedback='STAND OR SIT AT REACHABLE HEAD-END HEIGHT';return false;}
    stopPanelEdit();summonGeneration++;summonTween=null;release(0);release(1);finishViewChange();ctx.setReferenceMotion?.(1);
    patientReturn={position:mGrp.position.clone(),quaternion:mGrp.quaternion.clone(),scale:mGrp.scale.clone(),scaleMode,plinth:plinth.visible};
    patientStation.place(spec);patientView='external';plinth.visible=false;scope.velocity=0;scopeSteering=null;resetEmbodiedGrip();shaftClutch.reset();clutchAxis=null;clutchHeld=false;
    mGrp.quaternion.copy(patientStation.group.quaternion);mGrp.scale.setScalar(scale);mGrp.position.copy(new THREE.Vector3(...spec.neck)).sub(root.getPoint(0).multiplyScalar(scale).applyQuaternion(mGrp.quaternion));mGrp.updateMatrixWorld(true);scaleMode='reference-patient';
    patientZone=patientStation.zone(head.toArray());patientZoneLast=null;updateScope();positionPanels();grabFeedback='REFERENCE PATIENT / AIRWAY ENTRY ALIGNED';return true;
  }
  function updatePatientStation(){
    if(!patientReturn)return;
    if(getMeshMode()!=='reference'||mode!=='procedure'||xrPresentation!=='vr'){leavePatientStation();return;}
    patientStation.group.visible=viewerValid;
    patientZone=viewerValid?patientStation.zone(head.toArray()):{allowed:false,state:'NO_TRACKED_OPERATOR'};
    if(!patientZone.allowed){scope.velocity=0;scopeSteering=null;resetEmbodiedGrip();shaftClutch.reset();clutchAxis=null;clutchHeld=false;if(physicalControl?.snapshot().armed)physicalControl.stop('OPERATOR_OUTSIDE_REFERENCE_ZONE');}
    if(patientZoneLast!==patientZone.state){patientZoneLast=patientZone.state;grabFeedback=patientZone.allowed?'REFERENCE PATIENT / OPERATOR ZONE':'APPROACH HEAD OF BED / MOTION BLOCKED';paint();}
  }
  const plinth=new THREE.Mesh(new THREE.CylinderGeometry(.72,.78,.055,64),
    new THREE.MeshStandardMaterial({color:0x253742,metalness:.16,roughness:.86,
      transparent:true,opacity:.45}));
  plinth.position.set(0,.03,0);room.add(plinth);
  const stage=global.PulmoXRStagePresence.create(THREE,room,{floor,grid,plinth,rings:spatialRings});
  const laboratory=spatialGeometry.createLaboratory(THREE,room);
  chamber.visible=false;grid.material.opacity=0;ringMaterial.opacity=0;floor.material.transparent=true;floor.material.opacity=0;
  stage.root.getObjectByName('xr-architectural-datums').visible=false;
  const plinthSkirt=new THREE.Mesh(new THREE.CylinderGeometry(.755,.775,.038,64),new THREE.MeshStandardMaterial({color:0x404a50,metalness:.32,roughness:.56}));plinthSkirt.position.y=-.024;plinthSkirt.name='machined-plinth-skirt';plinth.add(plinthSkirt);
  plinth.material.color.setHex(0x727c80);plinth.material.opacity=1;plinth.material.transparent=false;
  // Surface target from a real per-frame hit test. Not room reconstruction.
  mrReticle=new THREE.Mesh(
    new THREE.RingGeometry(.075,.092,48).rotateX(-Math.PI/2),
    new THREE.MeshBasicMaterial({color:0x8fd6d7,side:THREE.DoubleSide,
      transparent:true,opacity:.94,depthWrite:false}));
  mrReticle.name='xr17b-actual-hit-test-reticle';
  mrReticle.renderOrder=32;mrReticle.visible=false;scene.add(mrReticle);
  // Plinth only indicates working region, never anatomical scale or a patient metric.
  const panelCanvas=document.createElement('canvas');panelCanvas.width=1024;panelCanvas.height=600;
  const panelCtx=panelCanvas.getContext('2d');
  const panelTex=new THREE.CanvasTexture(panelCanvas);
  panelTex.encoding=THREE.sRGBEncoding;
  worldMenu=new THREE.Mesh(new THREE.PlaneGeometry(.82,.48),
    new THREE.MeshBasicMaterial({map:panelTex,transparent:true,side:THREE.DoubleSide,depthWrite:false,depthTest:false}));
  worldMenu.renderOrder=40;worldMenu.visible=false;scene.add(worldMenu);
  const mainSpatialSurface=spatialGeometry.create(THREE,worldMenu,{width:.82,height:.48,radius:1.65,deckStart:.535,texture:panelTex});
  const dockCanvas=document.createElement('canvas');dockCanvas.width=1024;dockCanvas.height=180;const dockCtx=dockCanvas.getContext('2d'),dockTex=new THREE.CanvasTexture(dockCanvas);dockTex.encoding=THREE.sRGBEncoding;
  const compactDock=new THREE.Mesh(new THREE.PlaneGeometry(.76,.112),new THREE.MeshBasicMaterial({map:dockTex}));compactDock.name='xr-compact-command-rail';compactDock.visible=false;scene.add(compactDock);
  let railActions=spatialExperience.layoutRail('anatomy');
  const railSurface=spatialGeometry.create(THREE,compactDock,{width:.76,height:.112,radius:1.6,deckStart:0,texture:dockTex,depth:.019});
  railSurface.setActions(railActions);
  // Two focused pages, six generous targets each. The expert controls do
  // not crowd the core anatomy view, and no original command is removed.
  const panelLayouts={
    primary:[
      {id:'anatomy',label:'ANATOMY',u:.05,v:.54,w:.43,h:.12},
      {id:'procedure',label:'BRONCHOSCOPY',u:.52,v:.54,w:.43,h:.12},
      {id:'reset',label:'RECENTER',u:.05,v:.69,w:.43,h:.12},
      {id:'view',label:'FOCUS VIEW',u:.52,v:.69,w:.43,h:.12},
      {id:'more',label:'EXPERT TOOLS',u:.05,v:.84,w:.43,h:.12},
      {id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}
    ],
    advanced:[
      {id:'inspect',label:'LOBE INSPECTION',u:.05,v:.54,w:.43,h:.095},
      {id:'scale',label:'SOURCE SCALE',u:.52,v:.54,w:.43,h:.095},
      {id:'guide',label:'PROCEDURE GUIDE',u:.05,v:.65,w:.43,h:.095},
      {id:'precision',label:'VIEW CONTROLS',u:.52,v:.65,w:.43,h:.095},
      {id:'tool',label:'INSTRUMENT',u:.05,v:.76,w:.43,h:.095},
      {id:'airway-map',label:'AIRWAY MAP',u:.52,v:.76,w:.43,h:.095},
      {id:'back',label:'BACK',u:.05,v:.87,w:.43,h:.095},
      {id:'workspace-open',label:'VIRTUAL WORKSPACE',u:.52,v:.87,w:.43,h:.095}
    ],
    workspace:[
      {id:'workspace-forward',label:'STEP FORWARD',u:.05,v:.54,w:.43,h:.095},
      {id:'workspace-back',label:'STEP BACK',u:.52,v:.54,w:.43,h:.095},
      {id:'workspace-left',label:'STEP LEFT',u:.05,v:.65,w:.43,h:.095},
      {id:'workspace-right',label:'STEP RIGHT',u:.52,v:.65,w:.43,h:.095},
      {id:'workspace-turn-left',label:'TURN LEFT 30°',u:.05,v:.76,w:.43,h:.095},
      {id:'workspace-turn-right',label:'TURN RIGHT 30°',u:.52,v:.76,w:.43,h:.095},
      {id:'back-expert',label:'BACK',u:.05,v:.87,w:.43,h:.095},
      {id:'workspace-options',label:'WORKSPACE OPTIONS',u:.52,v:.87,w:.43,h:.095}
    ],
    'workspace-options':[
      {id:'spatial-open',label:'SPATIAL CALIBRATION',u:.05,v:.54,w:.43,h:.095},{id:'panel-shift',label:'SWITCH MENU SIDE',u:.52,v:.54,w:.43,h:.095},
      {id:'panel-edit-open',label:'WINDOW POSITION',u:.05,v:.65,w:.43,h:.095},{id:'research-open',label:'RESEARCH EVIDENCE',u:.52,v:.65,w:.43,h:.095},{id:'body-toggle',label:'BODY',u:.05,v:.76,w:.43,h:.095},{id:'hand-procedure',label:'HAND PROCEDURE',u:.52,v:.76,w:.43,h:.095},
      {id:'workspace-open',label:'BACK',u:.05,v:.87,w:.43,h:.095},{id:'spatial-viewpoint',label:'JUMP TO VIEWPOINT',u:.52,v:.87,w:.43,h:.095}
    ],
    research:[{id:'research-clear',label:'CLEAR SNAPSHOT',u:.05,v:.69,w:.43,h:.12},{id:'session-review-open',label:'MOTION REVIEW',u:.52,v:.69,w:.43,h:.12},{id:'workspace-options',label:'BACK',u:.05,v:.87,w:.43,h:.095},{id:'exit',label:'EXIT XR',u:.52,v:.87,w:.43,h:.095}],
    'panel-edit':[
      {id:'panel-select',label:'SELECT WINDOW',u:.05,v:.54,w:.43,h:.095},{id:'panel-size',label:'RESIZE WINDOW',u:.52,v:.54,w:.43,h:.095},
      {id:'panel-reset',label:'RESET WINDOWS',u:.05,v:.65,w:.43,h:.095},{id:'panel-compose',label:'ARRANGE WORKSPACE',u:.52,v:.65,w:.43,h:.095},{id:'back-expert',label:'BACK',u:.05,v:.87,w:.43,h:.095},{id:'exit',label:'EXIT XR',u:.52,v:.87,w:.43,h:.095}
    ],
    inspection:[
      {id:'inspect-toggle',label:'INDIVIDUAL LOBES',u:.05,v:.54,w:.43,h:.12},
      {id:'inspect-spread',label:'SEPARATE LOBES',u:.52,v:.54,w:.43,h:.12},
      {id:'inspect-lungs',label:'SEPARATE LUNGS',u:.05,v:.69,w:.43,h:.12},
      {id:'inspect-restore',label:'REASSEMBLE',u:.52,v:.69,w:.43,h:.12},
      {id:'back-expert',label:'BACK',u:.05,v:.84,w:.43,h:.12},
      {id:'inspect-isolate',label:'ISOLATE SELECTED',u:.52,v:.84,w:.43,h:.12}
    ],
    section:[
      {id:'section-toggle',label:'SECTION ON / OFF',u:.05,v:.54,w:.43,h:.095},{id:'section-axis',label:'SECTION AXIS',u:.52,v:.54,w:.43,h:.095},
      {id:'section-prev',label:'SLICE BACK',u:.05,v:.65,w:.43,h:.095},{id:'section-next',label:'SLICE FORWARD',u:.52,v:.65,w:.43,h:.095},
      {id:'section-back',label:'BACK',u:.05,v:.87,w:.43,h:.095},{id:'exit',label:'EXIT XR',u:.52,v:.87,w:.43,h:.095}
    ],
    ct:[
      {id:'ct-axis',label:'SELECT PLANE',u:.05,v:.54,w:.43,h:.12},
      {id:'ct-sections',label:'HIDE / SHOW MPR',u:.52,v:.54,w:.43,h:.12},
      {id:'ct-prev',label:'SLICE - 1',u:.05,v:.69,w:.43,h:.12},
      {id:'ct-next',label:'SLICE + 1',u:.52,v:.69,w:.43,h:.12},
      {id:'back-expert',label:'BACK',u:.05,v:.84,w:.43,h:.12},
      {id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}
    ],
    station:[
      {id:'station-layout',label:'LAYOUT',u:.05,v:.54,w:.43,h:.095},
      {id:'station-video',label:'VIDEO DETAIL',u:.52,v:.54,w:.43,h:.095},
      {id:'station-optics',label:'ENDOSCOPE VIEW',u:.05,v:.65,w:.43,h:.095},
      {id:'station-control',label:'FINE CONTROL',u:.52,v:.65,w:.43,h:.095},
      {id:'patient-open',label:'PATIENT STATION',u:.05,v:.76,w:.43,h:.095},
      {id:'optical-light',label:'LIGHTING',u:.52,v:.76,w:.43,h:.095},
      {id:'channel-open',label:'WORKING CHANNEL',u:.05,v:.87,w:.43,h:.095},
      {id:'back-expert',label:'BACK',u:.52,v:.87,w:.43,h:.095}
    ],
    patient:[{id:'patient-toggle',label:'PATIENT STATION',u:.05,v:.54,w:.43,h:.12},{id:'patient-view',label:'PATIENT VIEW',u:.52,v:.54,w:.43,h:.12},{id:'device-visible',label:'VIRTUAL SCOPE',u:.05,v:.69,w:.43,h:.12},{id:'hand-procedure',label:'HAND PROCEDURE',u:.52,v:.69,w:.43,h:.12},{id:'patient-back',label:'BACK TO STATION',u:.05,v:.84,w:.43,h:.12},{id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}],
    channel:[
      {id:'channel-cycle',label:'SELECT ACCESSORY',u:.05,v:.54,w:.43,h:.095},
      {id:'channel-action',label:'ACTUATE',u:.52,v:.54,w:.43,h:.095},
      {id:'channel-extend',label:'EXTEND',u:.05,v:.65,w:.43,h:.095},
      {id:'channel-retract',label:'RETRACT',u:.52,v:.65,w:.43,h:.095},
      {id:'optics-toggle',label:'OPTICAL RESPONSE',u:.05,v:.76,w:.43,h:.095},
      {id:'trace-open',label:'INSTRUMENT TRACE',u:.52,v:.76,w:.43,h:.095},
      {id:'channel-back',label:'BACK',u:.05,v:.87,w:.43,h:.095},
      {id:'imaging-open',label:'IMAGING LAB',u:.52,v:.87,w:.43,h:.095}
    ],
    imaging:[
      {id:'radial-toggle',label:'RADIAL VIEW',u:.05,v:.54,w:.43,h:.095},
      {id:'radial-coupling',label:'COUPLING',u:.52,v:.54,w:.43,h:.095},
      {id:'radial-range',label:'RANGE',u:.05,v:.65,w:.43,h:.095},
      {id:'motion-toggle',label:'RESPIRATORY MOTION',u:.52,v:.65,w:.43,h:.095},
      {id:'regional-toggle',label:'REGIONAL MOTION MAP',u:.05,v:.76,w:.43,h:.095},{id:'ebus-open',label:'EBUS INSTRUMENT',u:.52,v:.76,w:.43,h:.095},
      {id:'imaging-back',label:'BACK',u:.05,v:.87,w:.43,h:.095},
      {id:'exit',label:'EXIT XR',u:.52,v:.87,w:.43,h:.095}
    ],
    ebus:[{id:'radial-toggle',label:'EBUS ON/OFF',u:.05,v:.54,w:.43,h:.095},{id:'ebus-mode',label:'TYPE',u:.52,v:.54,w:.43,h:.095},{id:'ebus-feed',label:'FEED PROBE',u:.05,v:.65,w:.43,h:.095},{id:'ebus-retract',label:'RETRACT PROBE',u:.52,v:.65,w:.43,h:.095},{id:'ebus-roll',label:'ROTATE PROBE',u:.05,v:.76,w:.43,h:.095},{id:'radial-coupling',label:'COUPLING',u:.52,v:.76,w:.43,h:.095},{id:'imaging-open',label:'BACK',u:.05,v:.87,w:.43,h:.095},{id:'exit',label:'EXIT XR',u:.52,v:.87,w:.43,h:.095}],
    mr:[
      {id:'mr-confirm',label:'PLACE ON SURFACE',u:.05,v:.54,w:.43,h:.12},
      {id:'mr-reposition',label:'REPOSITION',u:.52,v:.54,w:.43,h:.12},
      {id:'mr-preview',label:'FLOATING PREVIEW',u:.05,v:.69,w:.43,h:.12},
      {id:'spatial-open',label:'SPATIAL CALIBRATION',u:.52,v:.69,w:.43,h:.12},
      {id:'back',label:'BACK',u:.05,v:.84,w:.43,h:.12},
      {id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}
    ],
    'spatial-cal':[{id:'spatial-pivot',label:'POINTER PIVOT',u:.05,v:.54,w:.43,h:.095},{id:'spatial-jig',label:'JIG LANDMARKS',u:.52,v:.54,w:.43,h:.095},{id:'spatial-capture',label:'ARM CAPTURE',u:.05,v:.65,w:.43,h:.095},{id:'spatial-align',label:'ALIGN REFERENCE',u:.52,v:.65,w:.43,h:.095},{id:'spatial-proxy',label:'TRACKED POINTER',u:.05,v:.76,w:.43,h:.095},{id:'spatial-planes',label:'PLANE MASK',u:.52,v:.76,w:.43,h:.095},{id:'spatial-clear',label:'CLEAR / UNLOCK',u:.05,v:.87,w:.43,h:.095},{id:'spatial-back',label:'BACK',u:.52,v:.87,w:.43,h:.095}],
    trace:[
      {id:'trace-play',label:'PLAY TRACE',u:.05,v:.54,w:.43,h:.095},
      {id:'trace-pause',label:'PAUSE',u:.52,v:.54,w:.43,h:.095},
      {id:'trace-prev',label:'PREV SAMPLE',u:.05,v:.65,w:.43,h:.095},
      {id:'trace-next',label:'NEXT SAMPLE',u:.52,v:.65,w:.43,h:.095},
      {id:'tool-proxy',label:'TRACKED TOOL PROXY',u:.05,v:.76,w:.43,h:.095},
      {id:'trace-clear',label:'CLEAR TRACE',u:.52,v:.76,w:.43,h:.095},
      {id:'trace-back',label:'BACK TO STATION',u:.05,v:.87,w:.43,h:.095},
      {id:'haptic-open',label:'HAPTIC LAB',u:.52,v:.87,w:.43,h:.095}
    ],
    'haptic-lab':[{id:'haptic-cue',label:'CONTACT CUES',u:.05,v:.54,w:.43,h:.12},{id:'haptic-arm',label:'ARM LOCAL DRY RUN',u:.52,v:.54,w:.43,h:.12},{id:'haptic-stop',label:'STOP DRY RUN',u:.05,v:.69,w:.43,h:.12},{id:'haptic-back',label:'BACK TO TRACE',u:.52,v:.69,w:.43,h:.12},{id:'session-review-open',label:'MOTION REVIEW',u:.05,v:.84,w:.43,h:.12},{id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}],
    'session-review':[{id:'record-start',label:'START RECORDING',u:.05,v:.54,w:.43,h:.095},{id:'record-stop',label:'STOP / REVIEW',u:.52,v:.54,w:.43,h:.095},{id:'review-prev',label:'PREVIOUS SECOND',u:.05,v:.65,w:.43,h:.095},{id:'review-next',label:'NEXT SECOND',u:.52,v:.65,w:.43,h:.095},{id:'review-play',label:'PLAY / PAUSE',u:.05,v:.76,w:.43,h:.095},{id:'record-share',label:'LOCAL OBSERVER',u:.52,v:.76,w:.43,h:.095},{id:'review-back',label:'BACK',u:.05,v:.87,w:.43,h:.095},{id:'record-clear',label:'CLEAR RECORDING',u:.52,v:.87,w:.43,h:.095}],
    precision:[
      {id:'undo',label:'UNDO',u:.05,v:.54,w:.43,h:.12},
      {id:'redo',label:'REDO',u:.52,v:.54,w:.43,h:.12},
      {id:'pivot',label:'PIVOT CENTER',u:.05,v:.69,w:.43,h:.12},
      {id:'axis',label:'AXIS YAW',u:.52,v:.69,w:.43,h:.12},
      {id:'back-expert',label:'BACK',u:.05,v:.84,w:.43,h:.12},
      {id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}
    ]
  };
  for(const actions of Object.values(panelLayouts))actions.push({id:'console-reading',label:'Read',u:.68,v:.018,w:.14,h:.066},{id:'console-collapse',label:'Minimize',u:.83,v:.018,w:.14,h:.066});
  actionHitbox=panelLayouts.primary;
  function paint(){
    const c=panelCtx,w=1024,h=600;
    const audit=getPatientAudit?.();
    const role=getMeshMode();
    const referenceFocus=role==='reference'?(getReferenceFocus?.()||null):null;
    const source=role==='patient'&&getCTStudy?.()
      ?'VERIFIED NIFTI SOURCE / RAS-COFRAMED · ANATOMY UNVALIDATED'
      :role==='patient'&&audit?.status==='coframed'
      ?(audit.source==='synthetic'?'SYNTHETIC · COFRAMED ONLY':'CT-DERIVED · COFRAMED / UNVERIFIED')
      :'PROCEDURAL REFERENCE · UNCALIBRATED';
    const clearance=mode==='procedure'?clearanceState():null;
    spatialUI.frame(c,w,h);
    const pageTitle=menuPage==='session-review'?'Reference motion review':menuPage==='haptic-lab'?'Instrument contact laboratory':menuPage==='spatial-cal'?'Spatial bench calibration':menuPage==='patient'?'Reference patient station':menuPage==='ebus'?'EBUS instrument research':menuPage==='research'?'Source research evidence':menuPage==='mr'?'Mixed reality placement':
      menuPage==='trace'?'Physical instrument trace':
      menuPage==='station'?'Bronchoscopy workstation':
      menuPage==='channel'?'Working channel':
      menuPage==='imaging'?'Imaging laboratory':
      menuPage==='workspace'||menuPage==='workspace-options'?'Virtual workspace':
      menuPage==='section'?'Reference lobe section':
      menuPage==='panel-edit'?'Independent window placement':
      menuPage==='inspection'?'Anatomical disassembly':
      menuPage==='precision'?'Precision anatomy controls':
      menuPage==='advanced'?'Research instruments':
      menuPage==='ct'?'Source CT / spatial sections':
      mode==='procedure'?'Bronchoscopy':'Anatomical exploration';
    spatialUI.header(c,w,'PulmoSIMU  /  '+(xrPresentation==='ar'?'Mixed reality':'Spatial laboratory'),pageTitle);
    spatialUI.text(c,(menuPage==='panel-edit'?'HOLD CONTROLLER GRIP TO MOVE SELECTED WINDOW':menuPage==='trace'?
      (replay.snapshot().loaded?
        'READ-ONLY RECORD · '+replay.snapshot().reviewIndex+' / '+(replay.snapshot().total-1):
        'NO BENCH TRACE · LOAD FROM DESKTOP FIRST')
      :menuPage==='channel'?channel.snapshot().label+' / '+channel.snapshot().extensionMm+' MM'
      :mode==='procedure'
      ?('SEG '+scope.segId+' · '+(SEG_MAP[scope.segId]?.name||'—'))
      :menuPage==='ct'&&ctSpatial
        ?'CT '+ctAxis.toUpperCase()+' / slice '+ctSpatial.snapshot().sections[ctAxis]
        :inspectMode?'INSPECTION · '+(lobes.find(l=>l.id===selectedLobe)?.name||'GRAB A LOBE'):'Original anatomical scene · '+scaleMode+' scale'+(referenceFocus?' · '+referenceFocus:'' )),42,159,w-84,{size:28,min:23,weight:500});
    spatialUI.text(c,spatialUI.label(menuPage==='trace'
       ?'PHYSICAL MEASUREMENT REVIEW · NEVER DRIVES THE ORIGINAL SCOPE'
       :mode==='procedure'
       ?(menuPage==='channel'?'AT JUNCTION: TRIGGERS SELECT BRANCH · USE ACTUATE FOR TOOL':'L GRIP: FEED SHAFT · L STICK: INSERT/ROLL · R GRIP: STEER')
       :xrPresentation==='ar'&&!placed
         ?'MR: SCAN SURFACE · TRIGGER TO PLACE · MENU: OPTIONS'
         :'GRAB TO MOVE · R STICK: ROTATE / DEPTH · LEFT Y: MENU'),42,200,w-84,{size:25,min:22,color:spatialUI.C.muted});
    spatialUI.text(c,spatialUI.label(menuPage==='trace'
      ?('TRACE QA '+replay.snapshot().engineeringStatus+' / '+
        (replay.snapshot().observation?.tracking?'TRACKING OK':'TRACKING NOT VERIFIED'))
      :xrPresentation==='ar'?
      'MR '+mrPlacement.snapshot().state.toUpperCase()+' / '+
      mrPlacement.snapshot().anchorStatus.toUpperCase()+' · UNVERIFIED ROOM DEPTH'
      :source),42,235,w-84,{size:23,min:20,color:spatialUI.C.fine});
    const clinicalLabel=mode==='procedure'
      ?(clearance
        ?'EST. LUMEN '+clearance.diameterMm.toFixed(1)+' mm  /  TOOL '+clearance.scopeDiameterMm.toFixed(1)+' mm  ·  '+(clearance.canAdvance?'PASS':'BLOCK')
        :'NO VALID LUMEN-TOOL CLEARANCE MODEL')
      :(role==='reference'&&physicsSnapshot
       ?'R3 REFERENCE (ASSUMED)  ·  '+(physicsSnapshot.pressurePa/98.0665).toFixed(1)+
          ' cmH2O  ·  '+(physicsSnapshot.flowM3PerS*1000).toFixed(2)+' L/s'
       :role==='reference'
        ?'R3 REDUCED-ORDER RESEARCH · LOADING / UNAVAILABLE'
        :'CT SOURCE SCALE IS METADATA-DERIVED, NOT CLINICALLY VERIFIED');
    const traceObs=replay.snapshot().observation;
    spatialUI.text(c,spatialUI.label(menuPage==='trace'&&traceObs
      ?'MEAS INS '+traceObs.measured.insertionMm.toFixed(1)+' mm · ERR '+
        traceObs.residual.insertionMm.toFixed(1)+' mm · TIP H '+traceObs.measured.tipHDeg.toFixed(1)+'°'
      :menuPage==='trace'
        ?'NO IMPORTED EXTERNAL REFERENCE RECORD · SOURCE SCOPE UNCHANGED'
        :clinicalLabel),42,272,w-84,{size:24,min:21,color:spatialUI.C.muted});
    spatialUI.text(c,spatialUI.label(active?grabFeedback:'Reference engine ready'),42,305,w-84,{size:23,min:20,color:/BLOCK|FAIL|UNAVAILABLE|INTERFERENCE/.test(grabFeedback)?spatialUI.C.warn:spatialUI.C.accent});
    if(menuPage==='session-review'){const s=localRecording.snapshot();c.save();c.fillStyle='#101b23';c.fillRect(35,142,w-70,175);spatialUI.text(c,s.reason.replaceAll('_',' ')+' · '+s.sampleCount+' samples',42,166,490,{size:23,min:17});spatialUI.text(c,localShareCode?'LOCAL OBSERVER '+localShareCode:'LOCAL RECORDING / NO AUTO UPLOAD',42,204,490,{size:19,min:15});spatialUI.text(c,reviewPacket?'Replay '+(reviewPacket.samples[reviewIndex].t_ms/1000).toFixed(1)+' s · '+(reviewIndex+1)+'/'+reviewPacket.samples.length:'Stop recording or import from session tools',42,245,490,{size:19,min:15});spatialUI.text(c,'READ-ONLY / REFERENCE MOTION',42,287,490,{size:18,min:14,color:spatialUI.C.warn});if(reviewPacket&&reviewCanvas)c.drawImage(reviewCanvas,535,149,440,164);c.restore();}
    if(menuPage==='haptic-lab'){const live=sensorObserver.snapshot(performance.now()),dry=hapticDryRun.snapshot();c.save();c.fillStyle='#101b23';c.fillRect(35,142,w-70,175);spatialUI.text(c,'GEOMETRY · '+hapticEvidence.state.replaceAll('_',' '),42,165,w-84,{size:24,min:18});spatialUI.text(c,'Margin '+(hapticEvidence.marginMm===null?'—':hapticEvidence.marginMm.toFixed(2)+' mm')+' · Index '+(hapticEvidence.risk===null?'—':hapticEvidence.risk.toFixed(2)),42,205,w-84,{size:22,min:17});spatialUI.text(c,live.state.replaceAll('_',' ')+' / Load '+(live.reportedAxialForceN===null?'—':live.reportedAxialForceN.toFixed(2)+' N (device reported)'),42,245,w-84,{size:19,min:15,color:spatialUI.C.muted});spatialUI.text(c,dry.reason.replaceAll('_',' ')+' · NO PHYSICAL OUTPUT',42,288,w-84,{size:19,min:15,color:spatialUI.C.warn});c.restore();}
    if(menuPage==='spatial-cal'){const s=spatialCal.snapshot(),oc=planeMask.snapshot();c.save();c.fillStyle='#101b23';c.fillRect(35,142,w-70,175);const instruction=s.kind==='pivot'?'FIX POINTER TIP / CAPTURE 5 DIFFERENT ORIENTATIONS':s.kind==='jig'?['TOUCH ORIGIN','TOUCH +X 200 mm','TOUCH +Z 150 mm','TOUCH OPPOSITE CORNER'][s.captured]:'START POINTER PIVOT OR JIG LANDMARKS';spatialUI.text(c,instruction,42,165,w-84,{size:23,min:17});spatialUI.text(c,s.state.replaceAll('_',' ')+' · '+s.captured+' samples',42,205,w-84,{size:21,min:17});spatialUI.text(c,'Pointer check '+(s.pointerHoldoutMm===null?'—':s.pointerHoldoutMm.toFixed(1)+' mm')+' / Jig check '+(s.jigHoldoutMm===null?'—':s.jigHoldoutMm.toFixed(1)+' mm'),42,244,w-84,{size:20,min:17,color:spatialUI.C.muted});spatialUI.text(c,(benchAligned?'MANUAL JIG ALIGNED':'NOT REGISTERED')+' · '+oc.state.replaceAll('_',' '),42,287,w-84,{size:19,min:15,color:spatialUI.C.warn});c.restore();}
    if(menuPage==='patient'){c.save();c.fillStyle='#101b23';c.fillRect(35,142,w-70,175);spatialUI.text(c,patientReturn?'SUPINE MANIKIN · '+patientView.toUpperCase():'ACTIVATE TO PLACE A REFERENCE PATIENT',42,165,w-84,{size:25,min:19});spatialUI.text(c,patientReturn?patientZone.state.replaceAll('_',' '):'Existing source anatomy remains unchanged',42,205,w-84,{size:22,min:18});spatialUI.text(c,'HEAD-END OPERATOR ZONE · MOUTH TO TRACHEA VISUAL GUIDE',42,246,w-84,{size:19,min:15,color:spatialUI.C.muted});spatialUI.text(c,'REFERENCE CONTEXT ONLY · NO PATIENT REGISTRATION',42,288,w-84,{size:19,min:15,color:spatialUI.C.warn});c.restore();}
    if(menuPage==='ebus'){const p=ebusProbe.snapshot();c.save();c.fillStyle='#101b23';c.fillRect(35,142,w-70,175);spatialUI.text(c,'REFERENCE / '+p.mode.toUpperCase()+' · '+p.extensionMm.toFixed(1)+' mm extension',42,165,w-84,{size:25,min:19});spatialUI.text(c,p.status.replaceAll('_',' ')+' · '+(p.shaftLocked?'SCOPE LOCKED':'SCOPE NOT LOCKED'),42,205,w-84,{size:22,min:18});spatialUI.text(c,'LEFT GRIP OR PINCH PROBE AT CHANNEL PORT · FEED/RETRACT',42,245,w-84,{size:19,min:15,color:spatialUI.C.muted});spatialUI.text(c,'IDEAL INTERFACE ONLY · NO TISSUE ULTRASOUND',42,288,w-84,{size:19,min:15,color:spatialUI.C.warn});c.restore();}
    if(menuPage==='research'){
      c.save();c.fillStyle='#101b23';c.fillRect(35,142,w-70,175);
      const q=researchSnapshot,b=replay.snapshot();
      const width=q?.frame?610:w-84;
      spatialUI.text(c,q?'UNREVIEWED SOURCE SNAPSHOT · '+q.route_sha256.slice(0,10):'LOAD XR REVIEW SNAPSHOT IN DESKTOP SESSION TOOLS',42,163,width,{size:23,min:18,color:spatialUI.C.accent});
      spatialUI.text(c,q?('Route '+q.route.length_mm.toFixed(1)+' mm · tool '+q.route.diameter_mm.toFixed(1)+' mm · '+q.route.node_count+' nodes'):'No source evidence loaded',42,198,width,{size:23,min:18});
      spatialUI.text(c,q?.mechanics?('L sampled overlap '+q.mechanics.overlap_mm.toFixed(2)+' mm · phase '+q.mechanics.phase):'No current mechanics evidence',42,232,width,{size:22,min:18,color:spatialUI.C.muted});
      spatialUI.text(c,'Independent bench: '+(b.loaded?b.engineeringStatus:'NOT LOADED'),42,266,width,{size:21,min:17,color:spatialUI.C.muted});
      spatialUI.text(c,'NO PATIENT REGISTRATION OR INSTRUMENT COUPLING',42,301,width,{size:19,min:15,color:spatialUI.C.warn});
      if(researchCanvas){c.drawImage(researchCanvas,750,149,220,165);}
      c.restore();
    }
    spatialUI.line(c,42,318,w-84);
    for(const a of actionHitbox){
      const x=a.u*w,y=a.v*h,width=a.w*w,height=a.h*h;
      const selected=a.id===mode||(a.id==='scale'&&scaleMode==='source-derived');
      const hovered=hoverPanels.includes(a.id);
      const available=(a.id!=='scale'||Boolean((audit?.sourceScaleEligible||getCTStudy?.())&&role==='patient')) &&
        (a.id!=='undo'||viewHistory.status().undo>0) &&
        (a.id!=='redo'||viewHistory.status().redo>0) &&
        (!a.id.startsWith('trace-')||a.id==='trace-open'||a.id==='trace-back'||replay.snapshot().loaded) &&
        (a.id!=='procedure'||role==='reference');
      a.disabled=!available;
      const settings=station.snapshot();
      const label=a.id==='console-reading'?(readingMode?'Normal':'Enlarge'):a.id==='console-collapse'?'Minimize':a.id==='trace-play'&&(replay.snapshot().playing)?'PLAYING TRACE':
        a.id==='record-share'?(localShare?'STOP LOCAL OBSERVER':'LOCAL OBSERVER'):a.id==='review-play'?(reviewPlaying?'PAUSE REVIEW':'PLAY REVIEW'):a.id==='haptic-cue'?(cueEngine.snapshot().enabled?'CONTACT CUES ON':'CONTACT CUES OFF'):a.id==='spatial-proxy'?(showPassiveProxy?'POINTER ON':'POINTER OFF'):a.id==='spatial-planes'?(planeMaskEnabled?'PLANE MASK ON':'PLANE MASK OFF'):a.id==='patient-toggle'?(patientReturn?'LEAVE PATIENT':'ENTER PATIENT STATION'):a.id==='patient-view'?'VIEW '+patientView.toUpperCase():a.id==='body-toggle'?(bodyEnabled?'BODY ON':'BODY OFF'):a.id==='hand-procedure'?(handProcedure?'HAND PROCEDURE ON':'HAND PROCEDURE OFF'):
        a.id==='ebus-mode'?'TYPE '+ebusProbe.snapshot().mode.toUpperCase():a.id==='radial-toggle'?(radialEnabled?'EBUS ON':'EBUS OFF'):
        a.id==='radial-coupling'?(radialCoupling==='air'?'AIR':'IDEAL FLUID'):
        a.id==='radial-range'?'RANGE '+radialRange+' MM':
        a.id==='motion-toggle'?(motionEnabled?'MOTION ON':'MOTION OFF'):
        a.id==='optics-toggle'?(!depthOpticsAvailable?'RAW SOURCE ONLY':opticsEnabled?'OPTICS ON':'OPTICS OFF'):
        a.id==='channel-action'?(channel.snapshot().tool==='suction'?(channel.snapshot().suction?'STOP INDICATOR':'SUCTION INDICATOR'):channel.snapshot().jawsOpen?'CLOSE JAWS':'ACTUATE'):
        a.id==='device-visible'?(deviceVisible?'HIDE VIRTUAL SCOPE':'SHOW VIRTUAL SCOPE'):
        a.id==='airway-map'?(routeVisible?'HIDE AIRWAY MAP':'SHOW AIRWAY MAP'):
        a.id==='optical-light'?'LIGHT '+opticalModes[opticalIndex].name:
        a.id==='guide'&&mode==='anatomy'?'LOBE SECTION':
        a.id==='panel-select'?'WINDOW '+panelTarget.toUpperCase():
        a.id==='regional-toggle'?(regionalDisplay?'REGIONAL MAP ON':'REGIONAL MAP OFF'):
        a.id==='section-toggle'?(section?'SECTION OFF':'SECTION ON'):
        a.id==='section-axis'?'AXIS '+sectionAxis.toUpperCase():
        a.id==='inspect-isolate'?(inspectionFilter==='isolate'?'SHOW ALL LOBES':'ISOLATE SELECTED'):
        a.id==='inspect-toggle'?(inspectMode?'WHOLE ORGAN':'INDIVIDUAL LOBES'):
        a.id==='display-path'?'DISPLAY PATH':
        a.id==='station-layout'?'LAYOUT '+settings.pose.label:
        a.id==='station-video'?'VIDEO '+settings.capture.label:
        a.id==='station-optics'?'VIEW '+settings.fovDegrees+' DEG':
        a.id==='station-control'?(settings.control==='fine'?'FINE INSERTION':'STANDARD INSERTION'):
        a.id==='scale'&&role==='patient'&&getCTStudy?.()?'CT SECTIONS':
        a.id==='axis'&&orientationLocked?'ORIENTATION LOCKED':
        a.id==='ct-axis'?'PLANE '+ctAxis.toUpperCase():
        a.id==='ct-sections'&&ctSpatial?(ctSpatial.snapshot().visible?'HIDE MPR':'SHOW MPR'):
        a.id==='view'
        ?(mode==='procedure'?'PROCEDURE STATION':
          xrPresentation==='ar'?'MR PLACEMENT':
          stage.snapshot().mode==='studio'?'FOCUS VIEW':'STUDIO VIEW'):
        a.id==='pivot'?(pivotMode==='center'?'PIVOT CENTER':'PIVOT ORIGIN'):
        a.id==='axis'?'AXIS '+axisMode.toUpperCase():a.label;
      spatialUI.control(c,{x,y,w:width,h:height,title:label,hovered,selected,available,action:a.id,physical:true});
    }
    mainSpatialSurface.setActions(actionHitbox);mainSpatialSurface.update(hoverPanels);
    railActions=spatialExperience.layoutRail(mode);
    dockCtx.clearRect(0,0,1024,180);for(const a of railActions){const title=a.id==='dock-mode'?(mode==='anatomy'?'Bronchoscopy':'Anatomy'):a.id==='dock-guide'?(guideVisible?'Hide guide':'Guide'):a.label;spatialUI.control(dockCtx,{x:a.u*1024,y:a.v*180,w:a.w*1024,h:a.h*180,title,hovered:hoverPanels.includes(a.id),available:!(a.id==='dock-mode'&&mode==='anatomy'&&role!=='reference'),action:a.id,physical:true});a.disabled=a.id==='dock-mode'&&mode==='anatomy'&&role!=='reference';}railSurface.setActions(railActions);railSurface.update(hoverPanels);dockTex.needsUpdate=true;
    panelTex.needsUpdate=true;
    paintMonitorControls();paintEBUSControls();paintGuide();
  }
  // Live bronchoscopy is rendered from the ORIGINAL camScope directly
  // into a render target owned by the SAME VR WebGL context. No canvas
  // readback, second-context upload, video recording or synthetic camera.
  const scopeTarget=new THREE.WebGLRenderTarget(
    station.snapshot().capture.width,station.snapshot().capture.height,{
    minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,
    format:THREE.RGBAFormat,depthBuffer:true,stencilBuffer:false
  });
  scopeTexture=scopeTarget.texture;scopeTexture.encoding=THREE.sRGBEncoding;
  const depthOpticsAvailable=rBro.capabilities.isWebGL2;
  if(depthOpticsAvailable){scopeTarget.depthTexture=new THREE.DepthTexture(scopeTarget.width,scopeTarget.height,THREE.UnsignedIntType);}
  const opticalResponse=depthOpticsAvailable?deviceLab.opticalMaterial(THREE,scopeTarget):null;
  let monitorStatus='not-started',monitorError=null,monitorFrames=0;
  let monitorFailures=0,monitorRecoveries=0,monitorRetryAt=0;
  let monitorBudget=instrument.videoCostGovernor();
  const monitorRoot=new THREE.Group();
  const endoscopyTower=global.PulmoXRProcedureTower.create(THREE,scene),ultrasoundTower=global.PulmoXRProcedureTower.create(THREE,scene,{secondary:true});
  monitorRoot.name='xr16-independent-endoscopy-display';
  const monitorFrame=new THREE.Mesh(new THREE.PlaneGeometry(1.00,.665),
    new THREE.MeshBasicMaterial({color:0x171b1e,side:THREE.DoubleSide}));
  monitorFrame.position.z=-.012;monitorRoot.add(monitorFrame);
  const monitorHousing=spatialGeometry.monitorHousing(THREE,monitorRoot);
  const monitorControlsCanvas=document.createElement('canvas');monitorControlsCanvas.width=1024;monitorControlsCanvas.height=128;const monitorControlsCtx=monitorControlsCanvas.getContext('2d'),monitorControlsTex=new THREE.CanvasTexture(monitorControlsCanvas);monitorControlsTex.encoding=THREE.sRGBEncoding;
  const monitorToolbar=new THREE.Mesh(new THREE.PlaneGeometry(.70,.078),new THREE.MeshBasicMaterial({map:monitorControlsTex}));monitorToolbar.name='endoscope-independent-control-shelf';monitorToolbar.position.set(0,-.395,.005);monitorRoot.add(monitorToolbar);
  const monitorActions=[{id:'monitor-focus',label:'Focus image',u:.02,v:.10,w:.34,h:.80},{id:'monitor-guide',label:'Guide',u:.38,v:.10,w:.28,h:.80},{id:'monitor-station',label:'Settings',u:.68,v:.10,w:.30,h:.80}];
  const monitorControlSurface=spatialGeometry.create(THREE,monitorToolbar,{width:.70,height:.078,radius:2.5,deckStart:0,depth:.017,texture:monitorControlsTex});
  function paintMonitorControls(){monitorControlsCtx.clearRect(0,0,1024,128);for(const a of monitorActions){const title=a.id==='monitor-focus'?(monitorFocus?'Restore layout':'Focus image'):a.id==='monitor-guide'?(guideVisible?'Hide guide':'Show guide'):a.label;spatialUI.control(monitorControlsCtx,{x:a.u*1024,y:a.v*128,w:a.w*1024,h:a.h*128,title,hovered:hoverPanels.includes(a.id),available:true,action:a.id,physical:true});}monitorControlSurface.setActions(monitorActions);monitorControlSurface.update(hoverPanels);monitorControlsTex.needsUpdate=true;}
  // Keep anatomical source pixels clear: identification and instrument data
  // are physically above/below the image, never drawn across the lumen.
  const rawMonitorMaterial=new THREE.MeshBasicMaterial({map:scopeTexture,side:THREE.DoubleSide,depthTest:false,depthWrite:false,toneMapped:false});
  const monitorImage=new THREE.Mesh(new THREE.PlaneGeometry(.94,.529),
    opticalResponse?.material||rawMonitorMaterial);
  monitorImage.position.y=-.012;
  rawMonitorMaterial.depthTest=true;rawMonitorMaterial.depthWrite=true;if(opticalResponse){opticalResponse.material.depthTest=true;opticalResponse.material.depthWrite=true;}
  monitorImage.visible=false;monitorImage.renderOrder=30;monitorRoot.add(monitorImage);
  const radialCanvas=document.createElement('canvas');radialCanvas.width=512;radialCanvas.height=512;
  const radialTexture=new THREE.CanvasTexture(radialCanvas);radialTexture.encoding=THREE.sRGBEncoding;
  const ebusScreen=new THREE.Group();ebusScreen.name='ui09-independent-ebus-display';ebusScreen.visible=false;scene.add(ebusScreen);
  const ebusFrame=new THREE.Mesh(new THREE.PlaneGeometry(.64,.69),new THREE.MeshBasicMaterial({color:0x171b1e,side:THREE.DoubleSide}));ebusFrame.position.z=-.012;ebusScreen.add(ebusFrame);
  const ebusHousing=spatialGeometry.monitorHousing(THREE,ebusScreen,{width:.64,height:.69,depth:.052});
  const radialPanel=new THREE.Mesh(new THREE.PlaneGeometry(.57,.57),new THREE.MeshBasicMaterial({map:radialTexture,side:THREE.DoubleSide,depthTest:true,depthWrite:true,toneMapped:false}));
  radialPanel.position.set(0,-.006,.003);radialPanel.renderOrder=31;radialPanel.visible=false;ebusScreen.add(radialPanel);
  const ebusStatusCanvas=document.createElement('canvas');ebusStatusCanvas.width=1024;ebusStatusCanvas.height=76;
  const ebusStatusCtx=ebusStatusCanvas.getContext('2d'),ebusStatusTexture=new THREE.CanvasTexture(ebusStatusCanvas);ebusStatusTexture.encoding=THREE.sRGBEncoding;
  const ebusStatusMesh=new THREE.Mesh(new THREE.PlaneGeometry(.60,.044),new THREE.MeshBasicMaterial({map:ebusStatusTexture,toneMapped:false}));ebusStatusMesh.position.set(0,.313,.005);ebusScreen.add(ebusStatusMesh);
  const ebusControlCanvas=document.createElement('canvas');ebusControlCanvas.width=1024;ebusControlCanvas.height=150;
  const ebusControlCtx=ebusControlCanvas.getContext('2d'),ebusControlTexture=new THREE.CanvasTexture(ebusControlCanvas);ebusControlTexture.encoding=THREE.sRGBEncoding;
  const ebusToolbar=new THREE.Mesh(new THREE.PlaneGeometry(.61,.078),new THREE.MeshBasicMaterial({map:ebusControlTexture}));ebusToolbar.position.set(0,-.393,.005);ebusScreen.add(ebusToolbar);
  const ebusActions=[{id:'ebus-feed',label:'Feed',u:.018,v:.10,w:.232,h:.80},{id:'ebus-retract',label:'Retract',u:.264,v:.10,w:.232,h:.80},{id:'ebus-roll',label:'Rotate',u:.510,v:.10,w:.232,h:.80},{id:'radial-range',label:'Range',u:.756,v:.10,w:.226,h:.80}];
  const ebusControlSurface=spatialGeometry.create(THREE,ebusToolbar,{width:.61,height:.078,radius:2.5,deckStart:0,depth:.017,texture:ebusControlTexture});
  function paintEBUSControls(){const p=ebusProbe.snapshot(),c=ebusControlCtx;c.clearRect(0,0,1024,150);for(const a of ebusActions){a.disabled=p.mode==='off'||a.id!=='radial-range'&&p.mode!=='radial'||a.id==='ebus-feed'&&p.extensionMm>=12||a.id==='ebus-retract'&&p.extensionMm<=0;const title=a.id==='radial-range'?radialRange+' mm':a.label;spatialUI.control(c,{x:a.u*1024,y:a.v*150,w:a.w*1024,h:a.h*150,title,hovered:hoverPanels.includes(a.id),available:!a.disabled,action:a.id,physical:true});}ebusControlSurface.setActions(ebusActions);ebusControlSurface.update(hoverPanels);ebusControlTexture.needsUpdate=true;}
  function paintEBUSStatus(){const p=ebusProbe.snapshot(),c=ebusStatusCtx;c.fillStyle='#171e23';c.fillRect(0,0,1024,76);spatialUI.text(c,p.mode.toUpperCase()+' EBUS / REFERENCE',18,48,670,{size:32,min:24,weight:550});spatialUI.text(c,p.mode==='radial'?p.extensionMm.toFixed(1)+' mm':radialRange+' mm',1005,48,300,{size:28,min:22,align:'right',color:spatialUI.C.accent});ebusStatusTexture.needsUpdate=true;ebusHousing.setStatus(radialEvidence?.state==='SCAN_UNAVAILABLE'?'unavailable':'live');}
  function updateProcedureTowers(){const enabled=active&&viewerValid&&mode==='procedure'&&xrPresentation==='vr'&&stage.snapshot().mode==='studio';const args={enabled,floor:room.position.y,room:room.position.toArray(),head:head.toArray(),patient:patientStation.snapshot().placement};endoscopyTower.update({...args,screen:scopeScreen,halfHeight:.344,lowerExtension:.10});ultrasoundTower.update({...args,screen:ebusScreen,halfHeight:.35,lowerExtension:.09});}
  function updateEBUSAssembly(){
    if(mode!=='procedure'||getMeshMode()!=='reference'||!viewerValid){ebusVisual.hide();return;}
    const frame=ctx.getReferenceMotionFrame?.()||mGrp;frame.updateWorldMatrix(true,false);const rootCurve=pathCurves[deviceLab.ancestry(SEG_MAP,scope.segId)[0]],entry=rootCurve?.getPoint(0).applyMatrix4(frame.matrixWorld),entryDirection=rootCurve?.getTangent(0).transformDirection(frame.matrixWorld),p=ebusProbe.snapshot(),b=probeBasis(p.mode==='radial'?p.extensionMm:0),tipWorld=scope.pos.clone().applyMatrix4(frame.matrixWorld),probeWorld=new THREE.Vector3(...b.origin).applyMatrix4(frame.matrixWorld),worldPerMm=frame.getWorldScale(new THREE.Vector3()).x/(getMmPerSceneUnit()*(ctx.getReferenceMotionFactor?.()||1));
    ebusVisual.update({handle:videoScope.group,entry,entryDirection,tipWorld,probeWorld,probeForward:new THREE.Vector3(...b.forward).transformDirection(frame.matrixWorld),probeUp:new THREE.Vector3(...b.up).transformDirection(frame.matrixWorld),worldPerMm,diameterMm:getScopeDiameterMm(),mode:p.mode,extensionMm:p.extensionMm,showConnection:videoScope.group.visible,mouth:patientReturn?patientStation.mouthWorld():null,showEntryGuide:patientView!=='external'});
  }
  function updateRadial(now){
    const show=mode==='procedure'&&getMeshMode()==='reference'&&radialEnabled;
    ebusScreen.visible=show;radialPanel.visible=show;monitorImage.scale.setScalar(1);monitorImage.position.x=0;
    if(!show)return;
    if(now-radialLast<500)return;radialLast=now;
    try{const field=ctx.getLumenField?.();if(!field)throw Error('NO_SOURCE_FIELD');
     const p=ebusProbe.snapshot(),basis=probeBasis(p.mode==='radial'?p.extensionMm:0),args={...basis,field,mmPerUnit:getMmPerSceneUnit()*(ctx.getReferenceMotionFactor?.()||1),rangeMm:radialRange,coupling:radialCoupling};
     if(p.mode==='radial'&&(p.extensionMm<=0||!probeClearance(p.extensionMm)))radialEvidence={state:p.extensionMm<=0?'PROBE_STOWED':'CLEARANCE_BLOCKED',rays:[],rangeMm:radialRange,coupling:radialCoupling};
     else radialEvidence=p.mode==='convex'?global.PulmoXREBUS.sector(args):global.PulmoXRRadialImaging.scan(args);
    }catch(error){radialEvidence={state:'SCAN_UNAVAILABLE',reason:String(error?.message||error).slice(0,80),coupling:radialCoupling};}
    if(ebusProbe.snapshot().mode==='convex')global.PulmoXREBUS.paint(radialCanvas,radialEvidence);else global.PulmoXRRadialImaging.paint(radialCanvas,radialEvidence);radialTexture.needsUpdate=true;paintEBUSStatus();paintEBUSControls();
  }
  function monitorStrip(y){
    const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=74;
    const ctx=canvas.getContext('2d');
    const texture=new THREE.CanvasTexture(canvas);texture.encoding=THREE.sRGBEncoding;
    const mesh=new THREE.Mesh(new THREE.PlaneGeometry(.94,.056),
      new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,
        depthTest:false,depthWrite:false}));
    mesh.material.depthTest=true;mesh.material.depthWrite=true;mesh.position.set(0,y,.004);mesh.renderOrder=31;
    monitorRoot.add(mesh);
    return {canvas,ctx,texture};
  }
  const topStrip=monitorStrip(.294),bottomStrip=monitorStrip(-.303);
  function updateRegionalOverlay(now){
    if(!regionalOverlay&&!regionalDisplay)return;
    if(!regionalOverlay){regionalOverlay=new THREE.Group();regionalOverlay.name='AG03 regional reference ventilation display';for(const l of lobes){const marker=new THREE.Mesh(new THREE.SphereGeometry(.035,12,8),new THREE.MeshBasicMaterial({color:0x77aabb,transparent:true,opacity:.85,depthWrite:false}));marker.userData.lobe=l.id;regionalOverlay.add(marker);}mGrp.add(regionalOverlay);}
    regionalOverlay.visible=regionalDisplay&&mode==='procedure'&&getMeshMode()==='reference';if(!regionalOverlay.visible||now-regionalLast<400)return;regionalLast=now;const values=regionalEvidence.snapshot().regions;const baseline=.0025/5;for(const marker of regionalOverlay.children){const l=lobes.find(v=>v.id===marker.userData.lobe);if(!l)continue;l.mesh.geometry.computeBoundingBox();const center=l.mesh.geometry.boundingBox.getCenter(new THREE.Vector3()).applyMatrix4(l.mesh.matrixWorld);marker.position.copy(mGrp.worldToLocal(center));const value=values[l.id]?.netM3;const scale=upgrade.regionalDisplayScale(value,baseline);marker.visible=scale!==null;if(scale!==null){marker.scale.setScalar(1+8*(scale-1));marker.material.color.setHex(scale>=1?0x6da9bd:0xc4a076);}}}
  function paintMonitorChrome(){
    monitorHousing.setStatus(monitorStatus);
    const settings=station.snapshot();
    const seg=SEG_MAP[scope.segId];
    function draw(strip,left,rightText,small=false){
      const c=strip.ctx,w=strip.canvas.width,h=strip.canvas.height;
      c.clearRect(0,0,w,h);c.fillStyle=spatialUI.C.surface;c.fillRect(0,0,w,h);
      spatialUI.text(c,spatialUI.label(left),18,48,w*.60,{size:small?25:30,min:22,weight:550});
      spatialUI.text(c,spatialUI.label(rightText),w-18,48,w*.36,{size:24,min:20,align:'right',color:spatialUI.C.accent});
      strip.texture.needsUpdate=true;
    }
    draw(topStrip,'Endoscopy  /  '+(seg?.name||'Reference airway'),'GEN '+(seg?.gen??'—')+' · '+monitorStatus.toUpperCase());
    draw(bottomStrip,
      'SEG '+scope.segId+' · '+(seg?.name||'—')+' · DEPTH '+Math.round(scope.depth)+' MM',
      settings.capture.label+' · '+settings.fovDegrees+'°',true);
  }
  scopeScreen=monitorRoot;scopeScreen.visible=false;scene.add(scopeScreen);
  // A dedicated, world-space, interactive procedure window. Its content
  // originates only from SEG_MAP, the real scope state, and the selected
  // original instrument. No overlay on the primary endoscope pixels.
  const guideCanvas=document.createElement('canvas');
  guideCanvas.width=1024;guideCanvas.height=680;
  guideCtx=guideCanvas.getContext('2d');
  guideTex=new THREE.CanvasTexture(guideCanvas);guideTex.encoding=THREE.sRGBEncoding;
  guidePlane=new THREE.Mesh(new THREE.PlaneGeometry(.90,.60),
    new THREE.MeshBasicMaterial({map:guideTex,transparent:true,
      side:THREE.DoubleSide,depthTest:false,depthWrite:false}));
  guidePlane.renderOrder=41;guidePlane.visible=false;
  guidePlane.name='xr14-original-procedure-guide';scene.add(guidePlane);
  const guideSpatialSurface=spatialGeometry.create(THREE,guidePlane,{width:.90,height:.60,radius:2.25,deckStart:.595,footerStart:.965,texture:guideTex,depth:.023});
  const guideStages=['ORIENT','NAVIGATE','CHOOSE','INSTRUMENT','REVIEW'];
  function paintGuide(){
    if(!guideCtx||!guideTex)return;
    const route=global.PulmoXRProcedureGuide.route({segments:SEG_MAP,segmentId:scope.segId,position:scope.t,lengthMm:id=>(pathCurves[id]?.getLength()||0)*getMmPerSceneUnit()});
    const state=global.PulmoXRProcedureGuide.describe({
      segmentId:scope.segId,segment:SEG_MAP[scope.segId],segments:SEG_MAP,
      position:scope.t,atBranch:scope.atBranch,clinicalMode:getClinMode(),
      clearance:clearanceState(),monitorStatus,guideStep
    });
    guideLastState=state;
    const c=guideCtx,w=guideCanvas.width,h=guideCanvas.height;
    spatialUI.frame(c,w,h);
    spatialUI.text(c,'PulmoSIMU  /  Procedure guide',42,46,w*.62,{size:24,weight:550,color:spatialUI.C.muted});
    spatialUI.text(c,'Step '+(guideStep+1)+'/5  ·  '+spatialUI.label(guideStages[guideStep]),w-42,46,w*.32,{size:24,align:'right',color:spatialUI.C.accent});
    spatialUI.text(c,spatialUI.label(state.heading),42,110,w-84,{size:44,min:34,weight:530});
    spatialUI.text(c,spatialUI.label(state.location),42,157,w-84,{size:32,weight:500});
    c.fillStyle='#9cb7c2';c.font='26px system-ui';
    c.fillText('GEN '+(state.generation??'—')+
      '    SEGMENT '+scope.segId+'    PATH '+state.progressPercent+'%',42,213);
    c.fillStyle=spatialUI.C.edge;c.fillRect(42,236,w-84,4);
    c.fillStyle=spatialUI.C.accent;c.fillRect(42,236,(w-84)*state.progressPercent/100,4);
    c.fillStyle='#dce8eb';c.font='600 26px system-ui';
    c.fillText(state.tool+'   /   '+state.monitor,42,289);
    c.fillStyle='#aebfc7';c.font='24px system-ui';
    spatialUI.text(c,route.labels.join(' › '),42,337,w-84,{size:25,min:20,color:spatialUI.C.muted});
    c.fillStyle=state.clearance.includes('BLOCKING')?'#e8b99d':'#93adba';
    c.font='23px system-ui';
    const observed=procedureMetrics.snapshot();
    c.fillText('ROUTE '+Math.round(route.distanceMm)+' MM · VISITED '+observed.visitedSegments+'/'+observed.totalSegments+' SEGMENTS',42,378);
    const branches=Array.from({length:Math.max(2,state.branches.length)},(_,i)=>({id:'guide-branch'+i,label:state.branches[i]?.label||'NO SOURCE BRANCH',disabled:!state.branches[i]?.enabled,u:.04+i*.92/Math.max(2,state.branches.length),v:.60,w:.88/Math.max(2,state.branches.length),h:.14}));
    guideHitboxes=[
      ...branches,
      {id:'scope-back',label:'RETRACT',disabled:accessoryDeployed()||probeLocked()||Boolean(physicalControl?.snapshot().armed),u:.04,v:.76,w:.43,h:.10},
      {id:'scope-forward',label:'ADVANCE',disabled:accessoryDeployed()||probeLocked()||Boolean(physicalControl?.snapshot().armed),u:.52,v:.76,w:.43,h:.10},
      {id:'guide-prev',label:'PREVIOUS',disabled:guideStep===0,u:.04,v:.88,w:.20,h:.08},
      {id:'guide-next',label:'NEXT STEP',disabled:guideStep===4,u:.27,v:.88,w:.20,h:.08},
      {id:'guide-tool',label:'TOOL MODE',disabled:false,u:.50,v:.88,w:.21,h:.08},
      {id:'guide-hide',label:'MINIMIZE',disabled:false,u:.74,v:.88,w:.22,h:.08}
    ];
    for(const a of guideHitboxes){
      const x=a.u*w,y=a.v*h,width=a.w*w,height=a.h*h;
      const hovered=hoverPanels.includes(a.id);
      spatialUI.control(c,{x,y,w:width,h:height,title:a.label,hovered,available:!a.disabled,action:a.id,physical:true});
    }
    spatialUI.text(c,spatialUI.label(state.provenance),42,672,w-84,{size:18,min:17,color:spatialUI.C.fine});
    // Real current instrument commands, deliberately separate from the
    // endoscopic image. Angles are reference actuator inputs, not measured
    // patient mechanics.
    c.fillStyle='#c7d7dc';c.font='23px system-ui';
    c.fillText('TIP H '+Math.round(scope.deflH*CFG.deflMax)+'°  /  V '+
      Math.round(scope.deflV*CFG.deflMax)+'°   ·   SHAFT '+
      Math.round(scope.roll*180/Math.PI)+'°',42,401);
    guideSpatialSurface.setActions(guideHitboxes);guideSpatialSurface.update(hoverPanels);
    guideTex.needsUpdate=true;
    paintMonitorChrome();
  }
  function captureFeed(now){
    if(mode!=='procedure'||now-lastFeed<station.snapshot().capture.intervalMs)return;
    if(monitorStatus==='unavailable'&&now<monitorRetryAt)return;
    lastFeed=now;
    const previousTarget=rBro.getRenderTarget();
    const previousXR=rBro.xr.enabled;
    const previousAspect=camScope.aspect,previousFov=camScope.fov;
    const menuWasVisible=worldMenu.visible,roomWasVisible=room.visible;
    const guideWasVisible=guidePlane?.visible||false;
    const monitorWasVisible=scopeScreen.visible;
    const originalBg=scene.background;
    const hidden=[...(ctx.getEndoscopeHiddenObjects?.()||[]),routeOverlay,regionalOverlay,toolProxyOverlay,videoScope.group,insertedShaft.mesh,ebusVisual.group,patientStation.group,spatialVisual.group,planeMask.group,ebusScreen,endoscopyTower.group,ultrasoundTower.group,compactDock,embodiment.group,...contactMarkers,...laserTips,...laserLines].filter(Boolean).map(o=>[o,o.visible]);
    try{
      for(const [o] of hidden)o.visible=false;
      // The endoscopic feed needs an opaque background even over passthrough.
      if(xrPresentation==='ar')scene.background=new THREE.Color(0x04080f);
      scopeScreen.visible=false;worldMenu.visible=false;room.visible=false;
      if(guidePlane)guidePlane.visible=false;
      // Three.js replaces auxiliary cameras while XR rendering is enabled.
      // Disable only for the offscreen pass, then restore in finally.
      rBro.xr.enabled=false;
      camScope.aspect=scopeTarget.width/scopeTarget.height;
      camScope.fov=global.PulmoXRVideoScope.verticalFov(station.snapshot().fovDegrees,camScope.aspect);
      camScope.updateProjectionMatrix();
      rBro.setRenderTarget(scopeTarget);
      rBro.clear(true,true,true);
      accessory.update(camScope,channel.snapshot(),(ctx.getReferenceMotionFrame?.()||mGrp).getWorldScale(new THREE.Vector3()).x/getMmPerSceneUnit());
      if(ctx.renderEndoscope)ctx.renderEndoscope();else rBro.render(scene,camScope);
      opticalResponse?.update({near:camScope.near,far:camScope.far,scale:mGrp.getWorldScale(new THREE.Vector3()).x,mmPerUnit:getMmPerSceneUnit(),enabled:opticsEnabled});
      monitorStatus='live';monitorError=null;monitorFrames++;
      monitorImage.visible=true;
    }catch(error){
      monitorStatus='unavailable';
      monitorError=String(error?.message||error).slice(0,140);
      monitorImage.visible=false;
      console.warn('PulmoSIMU endoscopic render target unavailable',monitorError);
    }finally{
      try{rBro.setRenderTarget(previousTarget)}catch(error){
        monitorStatus='unavailable';
        monitorError=String(error?.message||error).slice(0,140);
      }
      rBro.xr.enabled=previousXR;
      // r147 owns the XR render target for both base and projection layers.
      // Restore the exact frame target before the stereo pass.
      if(previousXR&&rBro.xr.isPresenting&&previousTarget===null){
        monitorStatus='unavailable';
        monitorError='XR frame target unavailable after scope pass';
        monitorImage.visible=false;
      }
      camScope.aspect=previousAspect;camScope.fov=previousFov;camScope.updateProjectionMatrix();
      scene.background=originalBg;
      worldMenu.visible=menuWasVisible;
      if(guidePlane)guidePlane.visible=guideWasVisible;
      room.visible=roomWasVisible;
      accessory.group.visible=false;
      scopeScreen.visible=monitorWasVisible;
      for(const [o,visible] of hidden)o.visible=visible;
      // One failed offscreen pass must not disable the monitor for the entire
      // session. Retry is bounded by backoff and always source-rendered.
      if(monitorStatus==='unavailable'){
        monitorFailures++;
        monitorRetryAt=now+Math.min(8000,500*Math.pow(2,Math.min(4,monitorFailures-1)));
        monitorImage.visible=false;
      }else if(monitorStatus==='live'&&monitorFailures){
        monitorRecoveries++;monitorFailures=0;monitorRetryAt=0;
      }
    }
  }
  for(let i=0;i<2;i++){
    scene.add(handProxies[i]);
    const marker=new THREE.Mesh(new THREE.SphereGeometry(.009,8,6),
      new THREE.MeshBasicMaterial({color:0xc0dce0,transparent:true,opacity:.76,depthTest:false}));
    marker.visible=false;marker.renderOrder=36;scene.add(marker);contactMarkers.push(marker);
  }
  function sampleVisualGeometry(now){if(now-visualLast<500)return;visualLast=now;for(const [name,obj,height,canvas,font] of [['console',worldMenu,.48,600,21],['guide',guidePlane,.60,680,21],['rail',compactDock,.112,180,32],['monitorControls',monitorToolbar,.078,128,32]]){if(!obj.visible||(name==='monitorControls'&&!scopeScreen.visible))continue;const p=obj.getWorldPosition(new THREE.Vector3()),s=obj.getWorldScale(new THREE.Vector3()),q=obj.getWorldQuaternion(new THREE.Quaternion()),toHead=head.clone().sub(p);const d=toHead.length(),normal=new THREE.Vector3(0,0,1).applyQuaternion(q);const observation=spatialExperience.readability({heightM:height,canvasHeight:canvas,fontPx:font,scale:s.y,distanceM:d,incidence:Math.min(1,Math.abs(normal.dot(toHead.normalize())))});if(observation)visualEvidence[name]=observation;}}
  function updatePointerAim(now){
    if(!active||now-lastAimTime<45)return;
    lastAimTime=now;
    let hoverChanged=false;
    for(let i=0;i<2;i++){
      const laser=laserLines[i],tip=laserTips[i];
      if(!laser||!tip)continue;
      const available=Boolean(controllerSources[i]||handSeen[i])&&(placed||xrPresentation==='ar');
      laser.visible=Boolean(available);
      tip.visible=false;aimTargets[i]=null;
      if(!available){
        if(hoverPanels[i]!==null){hoverPanels[i]=null;hoverChanged=true}
        continue;
      }
      // XR-12: the *same* nearest-hit router handles hover, click and grip.
      const {panel,organ,choice}=choosePointer(i);
      const nextHover=choice.kind==='panel'?choice.action:null;
      if(hoverPanels[i]!==nextHover){
        if(nextHover)haptic(i,.09,16);
        hoverPanels[i]=nextHover;hoverChanged=true;
      }
      if(choice.kind==='panel')
        aimTargets[i]={kind:'panel',hit:true,action:choice.action,distanceM:choice.distanceM};
      else if(choice.kind==='anatomy')
        aimTargets[i]={kind:'anatomy',hit:true,reason:organ.reason,distanceM:choice.distanceM};
      else if(organ)
        aimTargets[i]={kind:'none',hit:false,reason:organ.reason,distanceM:null};
      // Pointer never extends through the foremost actual surface.
      const distance=choice.kind==='none'?1.10:choice.distanceM;
      laser.scale.z=Math.max(.05,Math.min(3.5,distance));
      laser.material.color.setHex(choice.kind==='panel'?0xd8e6ed:
        choice.kind==='anatomy'?0xb4c9d2:0x84939c);
      laser.material.opacity=choice.kind==='none'?.25:.95;
      const point=choice.kind==='panel'?panel?.point:
        choice.kind==='anatomy'?organ?.point:null;
      if(point){
        tip.position.copy(point);tip.visible=true;
        tip.material.color.setHex(choice.kind==='panel'?0xd8e6ed:0xb4c9d2);
      }
    }
    if(hoverChanged)paint();
  }
  for(let i=0;i<2;i++){
    const controller=rBro.xr.getController(i),grip=rBro.xr.getControllerGrip(i);
    hands[i]=controller;grips[i]=grip;scene.add(controller,grip);
    const line=new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3(0,0,-1)]),
      new THREE.LineBasicMaterial({color:i?0xb6d1d9:0x789ba9,transparent:true,opacity:.45}));
    line.scale.z=3.5;line.visible=false;controller.add(line);
    laserLines[i]=line;
    const tip=new THREE.Mesh(new THREE.SphereGeometry(.012,10,8),
      new THREE.MeshBasicMaterial({color:0x7be3dd,depthTest:false,depthWrite:false}));
    tip.renderOrder=38;tip.visible=false;scene.add(tip);
    laserTips[i]=tip;
    controller.addEventListener('connected',event=>{
      controllerSide[i]=event.data?.handedness||null;
      controllerSources[i]=event.data||null;
      mapButtons[i].length=0;squeezeHeld[i]=false;
      health.event('controllerConnected');
    });
    controller.addEventListener('selectstart',event=>{
      // Hand selectstart is the physical pinch handled by XRJointPose.
      // Mark the corresponding gamepad edge handled to prevent a duplicate
      // procedure action when both WebXR events and buttons report the press.
      if(!event.data?.hand){
        mapButtons[i][0]=true;
        onTrigger(i);
      }
    });
    controller.addEventListener('selectend',event=>{
      if(!event.data?.hand)release(i,'controller','trigger');
    });
    controller.addEventListener('squeezestart',()=>{
      squeezeHeld[i]=true;
      startGrab(i,'controller','grip');
    });
    controller.addEventListener('squeezeend',()=>{
      squeezeHeld[i]=false;
      release(i,'controller','grip');
    });
    controller.addEventListener('disconnected',()=>{
      health.event('controllerDisconnected');
      release(i,'controller');controllerSide[i]=null;controllerSources[i]=null;
      toolProxyOverlay.visible=false;
      mapButtons[i].length=0;squeezeHeld[i]=false;
      if(laserLines[i])laserLines[i].visible=false;
      if(laserTips[i])laserTips[i].visible=false;
      scope.velocity=0;
    });
  }
  // Event buttons and actual headset session lifecycle
  button=document.createElement('button');
  button.type='button';button.id='pulmo-procedural-xr';
  button.textContent='ENTER PROCEDURAL XR';
  Object.assign(button.style,{position:'fixed',bottom:'18px',right:'18px',zIndex:10020,
    padding:'13px 19px',border:'1px solid #9ebfc8',background:'#0e2029',
    color:'#d9edf1',font:'600 12px system-ui',cursor:'pointer',borderRadius:'4px'});
  global.document.body.appendChild(button);
  // User-initiated local export after XR exit. Aggregates only, no patient
  // imaging, 6DoF tracking poses, trajectories, geometry or uploads.
  const reportButton=document.createElement('button');
  reportButton.type='button';reportButton.id='pulmo-xr-session-report';
  reportButton.textContent='EXPORT XR SESSION REPORT';
  Object.assign(reportButton.style,{position:'fixed',bottom:'74px',right:'18px',
    zIndex:10020,padding:'9px 13px',border:'1px solid #516b76',
    background:'#13212a',color:'#c6d8de',font:'500 11px system-ui',
    cursor:'pointer',borderRadius:'3px',display:'none'});
  global.document.body.appendChild(reportButton);
  const traceSummary=()=>{
    const {observation,...evidence}=replay.snapshot();
    return evidence; // No raw external trace rows in XR session reports.
  };
  const scienceSnapshot=()=>global.PulmoXRScience?.classify({
    modelRole:getMeshMode(),ctStudy:getCTStudy?.()||null,
    referenceSolverReady:Boolean(global.PulmoXRReferencePhysics?.step),
    geometryValidated:false,airwayReviewed:false,motionStudy:null
  })||null;
  const localSessionReport=()=>({
    release:'AG03 / UI06 · Integrated PulmoSIMU Research XR (UNVALIDATED)',
    spatialErgonomics:{observations:visualEvidence,contextActions,composition:compositionEvidence,physicalReadabilityVerified:false},
    session:health.snapshot(),
    interactionAttempts:{...attempts},
    quantitativeProcedure:procedureMetrics.snapshot(),
    localMotionReview:{...localRecording.snapshot(),rawTrajectoryIncluded:false,localObserverActive:!!localShare,remoteCollaboration:false},
    integratedUpgrade:{workspaceMoves,mrQuality:mrQuality.snapshot(),mrAnchorFault,coupling:couplingEvidence.snapshot(),regional:regionalEvidenceFinal||regionalEvidence.snapshot(),regionalDisplay:'REGIONAL_FLOW_DERIVED_MARKERS_ONLY',accessory:accessoryEvidence,contactProxy,environment:environmentEvidence,panelMoves,handGrasped:handGrasped.slice()},
    spatialRegistration:registrationEvidence,
    passiveContactInterface:contactEvidence,
    hapticLab:hapticFinal||{contact:hapticEvidence,cues:cueEngine.snapshot(),sensor:sensorObserver.snapshot(performance.now()),dryRun:hapticDryRun.snapshot()},
    respiratoryMotion:motionReport||motionEvidence,
    ebusInstrument:ebusProbe.snapshot(),
    radialImaging:radialEvidence?{...radialEvidence,rays:undefined}:null,
    proceduralGuidance:{visible:guideVisible,manualDemoStep:guideStep,
      actions:guideActions,lastSourceState:guideLastState},
    procedureStation:station.snapshot(),
    XR17CDisplayPath:displayPath.snapshot(),
    lumenSurface:ctx.getLumenEvidence?.()||null,
    nativeQuadEvidence:endedXRGeometry?.nativeQuadEvidence||nativeQuad?.snapshot()||null,
    calibratedSimulatorInput:physicalControl?.snapshot()||null,
    XR17CInstrumentTrace:traceSummary(),
    XR17CTrackedControllerProxy:toolProxy.snapshot(),
    embodiment:{...embodiment.snapshot(),handProcedure,handHeld,bodyEnabled,clinicalNavigation:false},
    virtualVideoScope:{visible:videoScope.group.visible,clutchHeld,referenceDimensions:videoScope.reference,calibrated:false},
    deviceLab:{depthOpticsAvailable,opticsEnabled,opticsApplied:Boolean(depthOpticsAvailable&&opticsEnabled),shaft:shaftEvidence,channel:channel.snapshot()},
    instrumentInputs:{tipH:scope.deflH,tipV:scope.deflV,shaftRollRad:scope.roll},
    videoCapture:{width:scopeTarget.width,height:scopeTarget.height,
      requestedIntervalMs:station.snapshot().capture.intervalMs,
      framesRendered:monitorFrames},
    sessionType:xrPresentation==='ar'?'mixed-reality-passthrough-preview':'immersive-vr',
    xrCapabilityEvidence:endedXRGeometry?.xrCapabilityEvidence||xrCapabilities.snapshot(),
    mrPlacementEvidence:endedXRGeometry?.mrPlacementEvidence||mrPlacement.snapshot(),
    lastInteractionResult:grabFeedback,
    xrGeometryAtExit:endedXRGeometry,
    maximumModelSpanM:limits.CONFIG.maximumSpanM,
    scienceIntegrity:scienceSnapshot(),
    spatialCT:endedXRGeometry?.spatialCT||ctSpatial?.snapshot()||null,
    E4Acceptance:global.PulmoXRAcceptance?.assess({
      session:health.snapshot(),xrGeometryAtExit:endedXRGeometry,
      endoscopeMonitor:{status:monitorStatus,framesRendered:monitorFrames},
      scienceIntegrity:scienceSnapshot()
    })||null,
    scaleClamps,workspaceClamps,
    endoscopeMonitor:{status:monitorStatus,framesRendered:monitorFrames,
      retryFailures:monitorFailures,recoveries:monitorRecoveries},
    videoBudget:monitorBudget.snapshot(),
    clinicalAndPhysicalReview:'EXTERNAL_EVIDENCE_REQUIRED_LOCAL_VALIDATION_LAB',
    integration:'Unified original-engine XR, verified local CT MPR and reference-only R3; CT-specific bronchoscopy and physical/clinical certification remain blocked'
  });
  reportButton.addEventListener('click',()=>{
    const blob=new Blob([JSON.stringify(localSessionReport(),null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download='pulmosimu-e1-e4-session-report.json';
    document.body.appendChild(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(url),2000);
  });
  global.pulmoXRSessionReport=localSessionReport;
  // Import research trace on the original desktop page, BEFORE entering XR.
  // All measured records remain local; a headset file chooser is never
  // launched while the user is inside an immersive procedure.
  const traceFile=document.createElement('input');
  traceFile.type='file';traceFile.accept='.json,application/json';
  traceFile.id='pulmo-xr-trace-file';traceFile.style.display='none';
  document.body.appendChild(traceFile);
  const traceButton=document.createElement('button');
  traceButton.type='button';traceButton.id='pulmo-xr-trace-import';
  traceButton.textContent='LOAD INSTRUMENT TRACE';
  Object.assign(traceButton.style,{position:'fixed',bottom:'126px',right:'18px',
    zIndex:10019,padding:'8px 12px',border:'1px solid #49626d',
    background:'#14222b',color:'#b6d0d7',font:'500 11px system-ui',
    cursor:'pointer',borderRadius:'3px',display:'none'});
  document.body.appendChild(traceButton);
  traceButton.addEventListener('click',()=>{if(!rBro.xr.isPresenting)traceFile.click()});
  traceFile.addEventListener('change',async()=>{
    const file=traceFile.files?.[0];
    if(!file)return;
    try{
      if(file.size>8_000_000)throw Error('TRACE_FILE_TOO_LARGE');
      const measured=JSON.parse(await file.text());
      const result=replay.load(measured);
      if(!result.ok)throw Error(result.reason);
      traceButton.textContent='TRACE LOADED / '+result.total+' SAMPLES';
      showToast('Independent instrument measurements ready for read-only XR review','info');
    }catch(error){
      showToast('Trace rejected: '+String(error?.message||error).slice(0,100),'warn');
    }finally{traceFile.value=''}
  });
  const researchSetup=document.createElement('details');researchSetup.id='pulmo-xr-research-import';
  researchSetup.innerHTML='<summary>Source research snapshot</summary><p>Import a locally exported XR review snapshot. Read-only; no scope or patient registration.</p><input type="file" accept=".json" aria-label="Research snapshot"><p role="status">No snapshot loaded.</p>';
  researchSetup.querySelector('input').addEventListener('change',async ev=>{
    const file=ev.target.files?.[0];ev.target.value='';if(!file)return;const epoch=++researchLoadEpoch;researchSnapshot=null;researchCanvas=null;
    const status=researchSetup.querySelector('[role="status"]');status.textContent='Reading local snapshot';
    try{if(file.size>1024*1024)throw Error('Snapshot exceeds 1 MiB');const doc=JSON.parse(await file.text());if(epoch!==researchLoadEpoch)return;if(rBro.xr.isPresenting)throw Error('Exit XR before import');researchSnapshot=global.PulmoXRResearchSnapshot.sanitize(doc);if(researchSnapshot.frame){researchCanvas=document.createElement('canvas');researchCanvas.width=160;researchCanvas.height=120;researchCanvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(researchSnapshot.frame.rgba),160,120),0,0);}status.textContent='Loaded / review pending · Expert tools → Virtual workspace → Workspace options → Research evidence';paint();}
    catch(e){if(epoch===researchLoadEpoch){researchSnapshot=null;researchCanvas=null;status.textContent='Rejected: '+e.message;}}
  });
  // Local instrument setup in the same workstation, before entering XR.
  const instrumentSetup=document.createElement('details');
  instrumentSetup.id='pulmo-calibrated-instrument';
  Object.assign(instrumentSetup.style,{position:'fixed',bottom:'166px',right:'18px',
   zIndex:10019,width:'290px',background:'#14222b',color:'#d7e4e8',padding:'8px 12px',
   border:'1px solid #49626d',font:'12px/1.5 system-ui',display:'none'});
  instrumentSetup.innerHTML='<summary>Physical instrument · local setup</summary>'+
   '<p>Research simulator only. Import a measured PULMOSIMU-CALIBRATION/1 file, connect a read-only serial device, then arm. Hold left grip in VR bronchoscopy to move the virtual scope. Release to stop.</p>'+
   '<input type="file" accept=".json,application/json" aria-label="Measured calibration file">'+
   '<p><button type="button" data-op="connect">Connect</button> <button type="button" data-op="arm">Arm simulator</button> <button type="button" data-op="stop">Stop</button></p>'+
   '<p role="status">No device connected.</p>';
  document.body.appendChild(instrumentSetup);
  const inputStatus=instrumentSetup.querySelector('[role="status"]');
  let calibrationLoaded=false,serialConnecting=false;
  const inputStatusRefresh=()=>{
    const a=physicalControl?.snapshot(),s=serialInput?.status(),live=sensorObserver.snapshot(performance.now());
    inputStatus.textContent=(calibrationLoaded?'Calibration loaded':'Calibration required')+' · '+
      (s?.state||'DISCONNECTED')+' · '+(a?.reason||'UNAVAILABLE')+' · '+live.state+' · '+(live.motion?live.motion.insertionMm.toFixed(1)+' mm / '+live.motion.shaftRollDeg.toFixed(1)+'°':'No fresh motion')+' · '+(live.reportedAxialForceN===null?'No fresh load':live.reportedAxialForceN.toFixed(2)+' N device-reported');
  };
  if(physicalCalibration&&physicalControl&&global.PulmoXRReadOnlySerial){
   serialInput=global.PulmoXRReadOnlySerial.create({navigatorObject:navigator,
    onLine:(line,now)=>{const observed=sensorObserver.ingest(line,now);if(observed.kind==='motion'&&physicalControl.snapshot().armed){if(!observed.accepted){physicalControl.stop(observed.reason);hapticDryRun.stop('MOTION_FRAME_REJECTED');return;}const result=physicalCalibration.ingest(line,now);if(!result.accepted){physicalControl.stop(result.reason);hapticDryRun.stop(result.reason);}}},
    onState:s=>{if(s.state==='DISCONNECTED'){if(physicalControl.snapshot().armed)physicalControl.stop('SERIAL_DISCONNECTED');hapticDryRun.stop('SERIAL_DISCONNECTED');sensorObserver.reset();}inputStatusRefresh();}
   });
  }
  instrumentSetup.querySelector('input').addEventListener('change',async e=>{
    try{physicalControl?.stop('CALIBRATION_REPLACED');hapticDryRun.stop('CALIBRATION_REPLACED');calibrationLoaded=false;
     const file=e.target.files?.[0];if(!file||file.size>100000)throw Error('INVALID_CALIBRATION_FILE');
     physicalCalibration.configure(JSON.parse(await file.text()));calibrationLoaded=true;
     inputStatusRefresh();
    }catch(err){inputStatus.textContent='Calibration rejected: '+String(err.message).slice(0,100)}
    finally{e.target.value=''}
  });
  instrumentSetup.querySelector('[data-op="connect"]').addEventListener('click',async()=>{
    if(serialConnecting)return;
    serialConnecting=true;
    try{if(!serialInput)throw Error('LOCAL_SERIAL_UNAVAILABLE');sensorObserver.reset();await serialInput.connect()}
    catch(err){inputStatus.textContent=String(err.message).slice(0,120)}
    finally{serialConnecting=false}
  });
  instrumentSetup.querySelector('[data-op="arm"]').addEventListener('click',()=>{
    try{if(!calibrationLoaded||serialInput?.status().state!=='ACQUIRING')throw Error('CALIBRATION_AND_CONNECTED_DEVICE_REQUIRED');
      if(!confirm('Arm the virtual reference simulator only? Physical measurement accuracy has not been independently verified.'))return;
      physicalControl.arm();inputStatusRefresh();
    }catch(err){inputStatus.textContent=String(err.message).slice(0,120)}
  });
  instrumentSetup.querySelector('[data-op="stop"]').addEventListener('click',()=>{
    physicalControl?.stop('OPERATOR_STOP');hapticDryRun.stop('OPERATOR_STOP');void serialInput?.disconnect();inputStatusRefresh();
  });
  addEventListener('pagehide',()=>{stopRecording('PAGE_EXIT');hapticDryRun.stop('PAGE_EXIT');physicalControl?.stop('PAGE_EXIT');void serialInput?.disconnect()});
  const sensorStatusTimer=setInterval(()=>{if(instrumentSetup.open&&!rBro.xr.isPresenting)inputStatusRefresh();},250);
  addEventListener('pagehide',()=>clearInterval(sensorStatusTimer),{once:true});
  const syncTraceButton=()=>{
    const splash=document.getElementById('splash');
    traceButton.style.display=!rBro.xr.isPresenting&&splash?.style.display==='none'
      ?'block':'none';
    instrumentSetup.style.display=traceButton.style.display;
  };
  const splash=document.getElementById('splash');
  if(splash&&typeof MutationObserver!=='undefined'){
    new MutationObserver(syncTraceButton).observe(splash,
      {attributes:true,attributeFilter:['style']});
  }
  syncTraceButton();
  // AR is an explicit, capability-gated alternative session for the SAME
  // original engine. Quest 2 displays grayscale camera passthrough. Browser
  // alpha compositing reveals the room; no fabricated room scan or occlusion.
  arButton=document.createElement('button');
  arButton.type='button';arButton.id='pulmo-passthrough-ar';
  arButton.textContent='Enter mixed reality';
  Object.assign(arButton.style,{position:'fixed',bottom:'18px',right:'241px',
    zIndex:10020,padding:'13px 16px',border:'1px solid #9ab8bd',
    background:'#15252d',color:'#d4e5ea',font:'600 12px system-ui',
    cursor:'pointer',borderRadius:'4px',display:'none'});
  document.body.appendChild(arButton);
  async function enterXR(presentation){
    if(rBro.xr.isPresenting){await rBro.xr.getSession()?.end();return}
    try{
      const splash=document.getElementById('splash'),app=document.getElementById('app');
      if(splash)splash.style.display='none';if(app)app.style.display='flex';
      onResize();
      // invoke request synchronously from this click: do NOT await probe()
      // before XR requestSession, or transient browser activation may expire.
      const session=await xrCapabilities.request(presentation);
      // local-floor is not guaranteed for passthrough. A local space is
      // sufficient for hit-test placement and avoids a rejected XR setup.
      const floorGranted=Array.from(session.enabledFeatures||[]).includes('local-floor');
      rBro.xr.setReferenceSpaceType(presentation==='ar'||!floorGranted?'local':'local-floor');
      xrPresentation=presentation;
      xrCapabilities.bind(session,presentation);
      try{await rBro.xr.setSession(session)}
      catch(error){
        xrCapabilities.end('three-session-rejected');
        try{await session.end()}catch{}
        throw error;
      }
    }catch(error){
      xrCapabilities.failure(error);
      xrPresentation='vr';
      runtimeStatus.style.display='block';
      runtimeStatus.textContent='XR ENTRY FAILED / '+String(error?.message||error).slice(0,160);
      showToast((presentation==='ar'?'MR passthrough':'Quest XR')+
        ' unavailable: '+String(error?.message||error),'warn');
      console.warn('PulmoSIMU WebXR entry',error);
    }
  }
  button.addEventListener('click',()=>enterXR('vr'));
  arButton.addEventListener('click',()=>enterXR('ar'));
  const runtimeStatus=document.createElement('div');
  runtimeStatus.id='pulmo-xr-runtime-status';
  runtimeStatus.setAttribute('role','status');
  Object.assign(runtimeStatus.style,{position:'fixed',bottom:'114px',right:'18px',
    zIndex:10019,maxWidth:'350px',textAlign:'right',font:'11px/1.6 system-ui',
    color:'#d0e1e6',background:'#14232bcf',padding:'7px 10px',
    border:'1px solid #49636f',borderRadius:'3px',display:'none'});
  document.body.appendChild(runtimeStatus);
  // Consolidate the original controls into one desktop dock; all original IDs and handlers survive.
  const launchDock=document.createElement('aside');launchDock.id='pulmo-launch-dock';launchDock.setAttribute('aria-label','Spatial session controls');
  const dockControls=document.createElement('div');dockControls.className='ps-dock-controls';
  const dockLabel=document.createElement('span');dockLabel.className='ps-dock-label';dockLabel.textContent='Spatial session';
  const motionSetup=document.createElement('details');motionSetup.id='pulmo-motion-review';motionSetup.innerHTML='<summary>Reference motion review</summary><p>Opt-in reference motion only. Export includes head-relative hand/controller trajectories; review before sharing. No room coordinates, patient data or automatic upload.</p><p><button type="button" data-motion="export">Export recording</button> <button type="button" data-motion="clear">Clear recording</button></p><input type="file" accept=".json,application/json" aria-label="Reference motion recording"><p><a href="./experimental/motion-review.html" target="_blank" rel="noopener">Open independent motion reviewer</a></p><p role="status">No recording.</p>';const motionStatus=motionSetup.querySelector('[role="status"]');
  motionSetup.querySelector('[data-motion="export"]').addEventListener('click',()=>{const packet=localRecording.packet();if(!packet){motionStatus.textContent='No recorded samples';return;}const url=URL.createObjectURL(new Blob([JSON.stringify(packet)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='pulmosimu-reference-motion.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);motionStatus.textContent='Local export requested · '+packet.samples.length+' samples';});
  motionSetup.querySelector('[data-motion="clear"]').addEventListener('click',()=>{reviewEpoch++;stopRecording('CLEARED');localRecording.clear();loadRecordedReview(null);motionStatus.textContent='Cleared';});
  motionSetup.querySelector('input').addEventListener('change',async e=>{const file=e.target.files?.[0];e.target.value='';if(!file)return;const epoch=++reviewEpoch;try{if(file.size>5*1024*1024||rBro.xr.isPresenting)throw Error('Import local JSON up to 5 MiB outside XR');const packet=reviewEngine.sanitize(JSON.parse(await file.text()));if(epoch!==reviewEpoch||rBro.xr.isPresenting)return;loadRecordedReview(packet);motionStatus.textContent='Read-only review loaded · '+packet.samples.length+' samples';}catch(err){if(epoch===reviewEpoch)motionStatus.textContent='Rejected: '+err.message;}});
  const dockTools=document.createElement('details');dockTools.className='ps-dock-tools';const toolSummary=document.createElement('summary');toolSummary.textContent='Session tools';const toolsPanel=document.createElement('div');toolsPanel.className='ps-tools-panel';toolsPanel.append(traceButton,instrumentSetup,researchSetup,motionSetup,reportButton);dockTools.append(toolSummary,toolsPanel);
  dockControls.append(dockLabel,arButton,button,dockTools);launchDock.append(runtimeStatus,dockControls);document.body.appendChild(launchDock);
  const syncDock=()=>{launchDock.hidden=rBro.xr.isPresenting||document.getElementById('splash')?.style.display!=='none';};
  if(splash&&typeof MutationObserver!=='undefined')new MutationObserver(syncDock).observe(splash,{attributes:true,attributeFilter:['style']});syncDock();
  function refreshXRAvailability(info){
    const caps=info||xrCapabilities.snapshot(),q=caps.sessions;
    button.disabled=q.vr==='unsupported'||q.vr==='unavailable'||q.vr==='error';
    if(button.disabled)button.textContent='VR unavailable';
    else button.textContent='Enter VR';
    // MR is passthrough only; room anchors and physical occlusion are
    // NOT supported merely because immersive-ar is available.
    arButton.style.display=q.mr==='supported'?'block':'none';
    const wantsMR=new URLSearchParams(location.search).get('entry')==='mr';
    runtimeStatus.style.display=wantsMR?'block':'none';
    if(wantsMR)runtimeStatus.textContent=q.mr==='supported'
      ?'MR passthrough available · room anchors and physical occlusion not verified'
      :q.mr==='checking'
        ?'Checking mixed-reality support · desktop remains available'
        :'MR passthrough unavailable in this browser · use desktop or supported headset';
  }
  refreshXRAvailability();
  void xrCapabilities.probe().then(refreshXRAvailability).catch(error=>{
    xrCapabilities.failure(error);refreshXRAvailability();
  });
  rBro.xr.addEventListener('sessionstart',()=>{
    // Preserve ordinary opaque 2D/VR rendering and fully restore it at exit.
    backgroundBeforeXR=scene.background;fogBeforeXR=scene.fog;
    clearBeforeXR=rBro.getClearColor(new THREE.Color()).clone();
    clearAlphaBeforeXR=rBro.getClearAlpha?.()??1;
    if(xrPresentation==='ar'){
      scene.background=null;scene.fog=null;
      mrHidden=(ctx.getMRHiddenObjects?.()||[]).map(o=>[o,o.visible]);
      for(const [o] of mrHidden)o.visible=false;
      rBro.setClearColor(0x000000,0);
    }
    launchDock.hidden=true;dockTools.open=false;
    health.begin(rBro.xr.getSession()?.frameRate??null);
    displayPath.inspect({session:rBro.xr.getSession(),renderer:rBro,
      gl:rBro.getContext?.()||null});
    nativeQuad?.bind(rBro);
    replay.pause(performance.now());
    traceButton.style.display='none';instrumentSetup.style.display='none';
    qualityStatus=configureComfortRendering();
    reportButton.style.display='none';
    rBro.xr.setReferenceSpace?.(null);clearSpatialCalibration();planeMaskEnabled=false;spatialFinal=null;workspaceMoves=0;workspaceRelayout=false;panelOffset=0;panelMoves=0;panelTarget='monitor';panelPinned.clear();panelEdit=null;environmentEvidence=upgrade.observeEnvironment(null);environmentLast=-Infinity;regionalDisplay=false;regionalLast=-Infinity;regionalEvidenceFinal=null;contactProxy={state:'UNAVAILABLE'};sectionAxis='x';sectionFraction=.5;mrAnchorFault=false;accessoryLast=-Infinity;regionalEvidence.reset();mrQuality.reset();couplingEvidence.reset();handGrasped.fill(false);accessoryEvidence={state:'UNAVAILABLE'};
    active=true;placed=false;viewerValid=false;lastFrame=0;lastFeed=0;
    lastPanelUpdate=0;scaleMode='comfort';endedXRGeometry=null;
    guideVisible=true;guideStep=0;guideActions=0;
    station.reset();channel.reset();insertedShaft.hide();opticsEnabled=true;procedureMetrics.reset(SEGS);
    registrationObserver.reset();registrationEvidence=registrationObserver.snapshot();contactEvidence=global.PulmoXRContactInterface.evaluate();resetHapticContext();hapticFinal=null;
    motionEnabled=true;motionEvidence=respiratoryMotion.reset();motionReport=null;ctx.setReferenceMotion?.(1);
    ebusProbe.reset();probeGrip.reset();ebusVisual.hide();radialEnabled=false;radialCoupling='air';radialLast=-Infinity;radialRange=20;radialEvidence=null;bodyEnabled=true;handProcedure=false;resetEmbodiedGrip();
    toolProxy.reset();toolProxyOverlay.visible=false;
    physicalBench?.reset();
    scopeTarget.setSize(station.snapshot().capture.width,station.snapshot().capture.height);
    camScope.fov=station.snapshot().fovDegrees;camScope.updateProjectionMatrix();
    viewHistory.clear();analogChangedAt=0;pivotMode='center';axisMode='yaw';
    orientationLocked=false;orientationAnchor=null;ctAxis='axial';
    if(ctSpatial){ctSpatial.dispose();ctSpatial=null;}
    menuPage='primary';actionHitbox=panelLayouts.primary;
    readingMode=false;monitorFocus=false;visualEvidence={};visualLast=-Infinity;contextActions=0;compositionEvidence=null;compositionCount=0;
    stage.setMode('studio');stage.root.getObjectByName('xr-architectural-datums').visible=false;laboratory.root.visible=true;chamber.visible=false;floor.material.opacity=0;
    physicsSnapshot=null;physicsError=null;
    global.PulmoXRReferencePhysics?.reset?.();
    handPinched.fill(false);handSeen.fill(false);grabSource.fill(null);
    squeezeHeld.fill(false);aimTargets.fill(null);hoverPanels.fill(null);
    laserLines.forEach(line=>{if(line)line.visible=false});
    laserTips.forEach(tip=>{if(tip)tip.visible=false});
    summonTween=null;lastAimTime=0;
    Object.keys(attempts).forEach(k=>attempts[k]=0);
    grabFeedback='AIM AT LUNG / HOLD TRIGGER OR GRIP';
    scaleClamps=0;workspaceClamps=0;limitsApplied=0;
    contactMarkers.forEach(m=>m.visible=false);
    monitorStatus='not-started';monitorError=null;monitorFrames=0;
    monitorFailures=0;monitorRecoveries=0;monitorRetryAt=0;
    monitorBudget=instrument.videoCostGovernor();
    monitorImage.visible=false;
    originalModelVisibility=mGrp.visible;
    saved={position:mGrp.position.clone(),quaternion:mGrp.quaternion.clone(),
      scale:mGrp.scale.clone(),extIdle:getExtIdle()};
    setExtIdle(false);vrCam.position.set(0,0,0);vrCam.quaternion.identity();
    worldMenu.visible=xrPresentation==='ar';compactDock.visible=false;scopeScreen.visible=false;room.visible=xrPresentation!=='ar';
    mrMenuPositioned=false;
    if(xrPresentation==='ar'){
      mGrp.visible=false; // Never auto-place unregistered geometry over passthrough.
      void mrPlacement.start(rBro.xr.getSession()).then(()=>{
        const state=mrPlacement.snapshot().state;
        grabFeedback=state==='unsupported'?'MR HIT TEST UNAVAILABLE / TRIGGER OR PINCH TO PLACE MANUALLY':'LOOK AT SURFACE / TRIGGER OR PINCH TO PLACE';
        paint();
      });
      grabFeedback='MR SCAN / FIND A REAL SURFACE OR CHOOSE PREVIEW';
    }else mrPlacement.reset();
    mrReticle.visible=false;
    if(guidePlane)guidePlane.visible=false;
    room.position.copy(head).setY(-.02);
    button.textContent='EXIT PROCEDURAL XR';
    arButton.textContent='EXIT MR PREVIEW';
    setMode('anatomy');
    if(xrPresentation==='ar'){menuPage='mr';actionHitbox=panelLayouts.mr;grabFeedback='MR / LOOK AT SURFACE OR USE FLOATING PREVIEW';}
    paint();
    rBro.setAnimationLoop((time,frame)=>{
      if(!active||!rBro.xr.isPresenting)return;
      health.startFrame(time);
      if(!headPose(frame)){
        localRecording.suspend();
        // Tracking loss: never continue the last object or instrument command.
        registrationEvidence=registrationObserver.step({tracked:false});contactEvidence=global.PulmoXRContactInterface.evaluate();hapticDryRun.stop('TRACKING_LOST');hapticEvidence=hapticLab.estimate();lastShaftAt=-Infinity;lastShaftSample=null;
        ctx.setReferenceMotion?.(1);procedureMetrics.suspend();health.event('viewerPoseLost');health.endFrame({poseAvailable:false});
        nativeQuad?.update({visible:false});
        if(physicalControl?.snapshot().armed)physicalControl.stop('TRACKING_LOST');
        viewerValid=false;ebusScreen.visible=false;endoscopyTower.hide();ultrasoundTower.hide();if(benchAligned){invalidateBench();mGrp.visible=false;placed=false;}clearSpatialCalibration();patientStation.group.visible=false;compactDock.visible=false;scope.velocity=0;stopPanelEdit();regionalEvidence.suspend();release(0);release(1);
        ebusVisual.hide();probeGrip.reset();toolProxyOverlay.visible=false;videoScope.group.visible=false;shaftClutch.reset();clutchAxis=null;clutchHeld=false;
        if(mrReticle)mrReticle.visible=false;
        handSeen.fill(false);handPinched.fill(false);handGrasped.fill(false);mrQuality.observe(null);embodiment.reset();resetEmbodiedGrip();handSamples={};handMenuBlocked={left:false,right:false};
        contactMarkers.forEach(marker=>marker.visible=false);
        laserLines.forEach(line=>{if(line)line.visible=false});
        laserTips.forEach(tip=>{if(tip)tip.visible=false});
        return;
      }
      if(workspaceRelayout){workspaceRelayout=false;positionPanels();}
      if(!placed&&xrPresentation!=='ar'){
        room.position.copy(head).setY(head.y-1.18);
        const center=head.clone().addScaledVector(forward,1.5);
        plinth.position.x=center.x-head.x;plinth.position.z=center.z-head.z;
        frameGroup();
      }
      if(xrPresentation==='ar'){
        if(time-environmentLast>500){environmentLast=time;environmentEvidence=upgrade.observeEnvironment(frame);}
        const hit=mrPlacement.scan(frame,rBro.xr.getReferenceSpace());
        if(!placed)mrQuality.observe(hit?.point?[hit.point.x,hit.point.y,hit.point.z]:null);
        mrReticle.visible=Boolean(hit?.kind==='candidate'&&!placed);
        if(hit?.kind==='candidate')
          mrReticle.position.set(hit.point.x,hit.point.y+.006,hit.point.z);
        if(hit?.kind==='placed'){
          mrReticle.visible=false;
          placeOnMRSurface(hit.point);
        }else if(hit?.kind==='anchor-delta'&&placed&&!mrAnchorFault){
          const delta=hit.delta;
          if([delta.x,delta.y,delta.z].every(Number.isFinite)&&
             Math.hypot(delta.x,delta.y,delta.z)<.15){
            mGrp.position.add(new THREE.Vector3(delta.x,delta.y,delta.z));
            mGrp.updateWorldMatrix(true,false);
          }else{mrAnchorFault=true;release(0);release(1);scope.velocity=0;grabFeedback='MR ANCHOR DISCONTINUITY / REPOSITION REQUIRED';paint();}
        }
        if(!placed&&viewerValid&&!mrMenuPositioned){
          positionPanels();
          worldMenu.position.copy(head).addScaledVector(forward,.86).add(new THREE.Vector3(0,-.12,0));
          worldMenu.lookAt(head);worldMenu.updateWorldMatrix(true,false);
          mrMenuPositioned=true;
        }
      }
      const dt=lastFrame?Math.min(.05,Math.max(0,(time-lastFrame)/1000)):.016;lastFrame=time;
      replay.tick(time); // Non-actuating, local measured trace playback.
      observeCalibration(frame);updatePatientStation();poll(dt);
      if(analogChangedAt&&!grab.some(Boolean)&&performance.now()-analogChangedAt>380)
        finishViewChange();
      updateTrackedHands(frame);updateEmbodiedBody(frame);updateSummon();updateGrab();updatePanelEdit(dt);updateSteering();
      compactDock.visible=active&&viewerValid&&!worldMenu.visible;
      updateToolProxy();updateVideoScope(frame,dt);updatePointerAim(time);
      for(const surface of [mainSpatialSurface,railSurface,guideSpatialSurface,monitorControlSurface])surface.update(hoverPanels,time);sampleVisualGeometry(time);
      // The identical SI-unit R3 solver used in the research laboratory;
      // explicitly reference-only, never CT-fitted or mesh-deforming.
      if(getMeshMode()==='reference'&&global.PulmoXRReferencePhysics?.step){
        try{physicsSnapshot=global.PulmoXRReferencePhysics.step(dt);physicsError=null}
        catch(error){physicsSnapshot=null;physicsError=String(error?.message||error).slice(0,140)}
      }else physicsSnapshot=null;
      // Only a calibrated, live, local device can request simulator motion.
      // Left grip is a dead-man; no replay, no hardware write, no patient input.
      if(physicalControl?.snapshot().armed&&(!patientMotionAllowed()||accessoryDeployed()||ebusProbe.snapshot().mode!=='off'))physicalControl.stop('ACCESSORY_DEPLOYED');
      if(physicalControl?.snapshot().armed&&panelEdit)physicalControl.stop('WINDOW_EDIT');
      if(physicalControl?.snapshot().armed){
        scope.velocity=0;scopeSteering=null;
        const deadman=trackedLeftGrip(frame);
        const length=pathCurves[scope.segId]?.getLength()*getMmPerSceneUnit();
        const command=physicalControl.step({now:performance.now(),deadman,
          reference:getMeshMode()==='reference',procedure:mode==='procedure',
          visible:viewerValid&&xrPresentation==='vr',
          connected:serialInput?.status().state==='ACQUIRING',
          segmentLengthMm:length,deflectionMax:CFG.deflMax});
        if(command){
          scope.roll+=command.rollRad;
          scope.deflH=safe(-1,1,scope.deflH+command.tipH);
          scope.deflV=safe(-1,1,scope.deflV+command.tipV);
          if(command.insertionT>0)advance(command.insertionT);
          else if(command.insertionT<0)retract(-command.insertionT);
        }
      }
      // Advance the ORIGINAL medical procedure engine on XR time, not
      // tab requestAnimationFrame, which can be throttled in a headset.
      if(mode==='procedure'&&!accessoryDeployed()&&!probeLocked()&&Math.abs(scope.velocity)>.0001){
        const step=Math.min(1.5,Math.max(.1,dt*60));
        const distance=scope.velocity*step;
        if(distance>0)advance(distance);else retract(-distance);
        scope.velocity*=Math.pow(scope.velocityDecay,step);
      }
      motionEvidence=respiratoryMotion.step(physicsSnapshot,motionEnabled&&mode==='procedure'&&getMeshMode()==='reference'&&viewerValid,dt);
      if(mode==='procedure'&&getMeshMode()==='reference'&&viewerValid)couplingEvidence.step(physicsSnapshot,motionEvidence,dt);
      ctx.setReferenceMotion?.(motionEvidence.scale,[routeOverlay,insertedShaft.mesh]);
      updateScope();updateRouteOverlay();updateEBUSAssembly();updateRadial(time);updateProcedureTowers();updateSection();
      if(mode==='procedure'&&accessoryDeployed()&&time-accessoryLast>100){accessoryLast=time;const state=channel.snapshot();accessoryEvidence=checkAccessory(state.extensionMm,state.jawsOpen);if(!accessoryEvidence.allowed){scope.velocity=0;grabFeedback='REFERENCE ACCESSORY INTERFERENCE / RETRACT';}}
      if(mode==='procedure'&&getMeshMode()==='reference')regionalEvidence.step(physicsSnapshot);else regionalEvidence.suspend();
      updateRegionalOverlay(time);
      if(mode==='procedure'&&getMeshMode()==='reference')shaftEvidence=insertedShaft.update({timeMs:time,segments:SEG_MAP,curves:pathCurves,id:scope.segId,t:scope.t,diameterMm:getScopeDiameterMm(),mmPerUnit:getMmPerSceneUnit(),field:ctx.getLumenField?.()});
      else insertedShaft.hide();
      registrationEvidence=registrationObserver.step({tracked:viewerValid&&mode==='procedure'&&getMeshMode()==='reference',timeMs:time,positionMm:scope.pos.toArray().map(v=>v*getMmPerSceneUnit())});
      contactEvidence=global.PulmoXRContactInterface.evaluate({shaft:shaftEvidence,reference:getMeshMode()==='reference',tracking:viewerValid,procedure:mode==='procedure'});
      contactProxy=mode==='procedure'&&getMeshMode()==='reference'?upgrade.contactEstimate(shaftEvidence.minMarginMm,getScopeDiameterMm()):upgrade.contactEstimate(null,null);
      updateHapticLab(frame,time);observeRecording(frame,time);tickReview(time);
      procedureMetrics.observe({timeMs:time,procedure:mode==='procedure',reference:getMeshMode()==='reference',tracking:viewerValid,segmentId:scope.segId,t:scope.t,positionMm:scope.pos.toArray().map(v=>v*getMmPerSceneUnit()),shaftState:shaftEvidence.state});
      if(time-lastPanelUpdate>440){lastPanelUpdate=time;paint()}
      let monitorCpuMs=0;
      if(mode==='procedure'){
        const monitorStarted=performance.now(),beforeStatus=monitorStatus,
          beforeFrames=monitorFrames;
        captureFeed(time);monitorCpuMs=performance.now()-monitorStarted;
        if(beforeStatus!=='unavailable'&&monitorStatus==='unavailable')
          health.event('monitorUnavailable');
        // Only consider actual offscreen source renders, not skipped frames.
        // This is the measured CPU submission cost, never GPU frame time.
        if(monitorStatus==='live'&&monitorFrames>beforeFrames){
          const recommended=monitorBudget.note(monitorCpuMs,station.snapshot().video);
          if(recommended&&station.snapshot().video!==recommended){
            station.set('video',recommended); // deterministic quality step
            const video=station.snapshot().capture;
            scopeTarget.setSize(video.width,video.height);
            lastFeed=0;
            grabFeedback='VIDEO QUALITY ADAPTED / HIGH MEASURED CPU COST';
            paint();
          }
        }
      }
      // Native compositor gets only the original source render target.
      // The mesh remains visible until a real XR subimage copy succeeds.
      // Native copies raw source pixels; retain the processed mesh when optical response is selected.
      const useProcessedOptics=Boolean(opticalResponse&&opticsEnabled);
      monitorImage.material=useProcessedOptics?opticalResponse.material:rawMonitorMaterial;
      const nativeVisible=mode==='procedure'&&monitorStatus==='live'&&xrPresentation==='vr'&&deviceLab.allowNativeDisplay({opticsActive:useProcessedOptics,radialEnabled,nativeOptIn:nativeLayersOptIn});
      const p=monitorImage.getWorldPosition(new THREE.Vector3());
      const q=monitorImage.getWorldQuaternion(new THREE.Quaternion());
      const nativeActive=nativeQuad?.update({renderer:rBro,frame,target:scopeTarget,
        space:rBro.xr.getReferenceSpace(),position:{x:p.x,y:p.y,z:p.z},
        orientation:{x:q.x,y:q.y,z:q.z,w:q.w},visible:nativeVisible});
      monitorImage.visible=mode==='procedure'&&monitorStatus==='live'&&!nativeActive;
      // Stereoscopic source render. CPU call time is NOT a GPU timing.
      const renderStarted=performance.now();
      try{rBro.render(scene,vrCam)}
      finally{health.endFrame({renderMs:performance.now()-renderStarted,
        monitorMs:monitorCpuMs})}
    });
  });
  rBro.xr.addEventListener('sessionend',()=>{
    if(localRecording.snapshot().running){stopRecording('SESSION_ENDED');const packet=localRecording.packet();if(packet)loadRecordedReview(packet);}else stopLocalShare();reviewPlaying=false;if(localRecording.snapshot().sampleCount)motionStatus.textContent='Recorded '+localRecording.snapshot().sampleCount+' samples · ready for export';
    hapticDryRun.stop('SESSION_ENDED');hapticFinal={contact:hapticEvidence,cues:cueEngine.snapshot(),sensor:sensorObserver.snapshot(performance.now()),dryRun:hapticDryRun.snapshot()};resetHapticContext();
    spatialFinal={...spatialCal.snapshot(),benchAligned,occlusion:planeMask.snapshot()};clearSpatialCalibration();planeMaskEnabled=false;
    // Snapshot before restoring the original desktop scene scale.
    endedXRGeometry={
      modelSpanM:boundsCache?+(modelExtent()*mGrp.scale.x).toFixed(3):null,
      scaleMode,scaleClamps,workspaceClamps,
      finalGrabStatus:grabFeedback,
      stagePresentation:stage.snapshot(),architecture:laboratory.snapshot(),
      xrFoveation:qualityStatus,
      sessionType:xrPresentation==='ar'?'immersive-ar':'immersive-vr',
      procedureGuide:{visible:guideVisible,step:guideStep,actions:guideActions},
      procedureStation:station.snapshot(),
      nativeQuadEvidence:nativeQuad?.snapshot()||null,
      xrCapabilityEvidence:xrCapabilities.snapshot(),
      mrPlacementEvidence:mrPlacement.snapshot(),
      instrumentInputs:{tipH:scope.deflH,tipV:scope.deflV,shaftRollRad:scope.roll},
      controllerActions:{...attempts},
      viewHistory:viewHistory.status(),pivotMode,axisMode,
      orientationLocked,spatialCT:ctSpatial?.snapshot()||null
    };
    regionalEvidenceFinal=regionalEvidence.snapshot();stopPanelEdit();panelPinned.clear();if(regionalOverlay)regionalOverlay.visible=false;regionalDisplay=false;
    motionReport={...motionEvidence};ctx.setReferenceMotion?.(1);ebusProbe.reset();probeGrip.reset();ebusVisual.hide();radialEnabled=false;radialPanel.visible=false;ebusScreen.visible=false;endoscopyTower.hide();ultrasoundTower.hide();monitorImage.scale.setScalar(1);monitorImage.position.x=0;
    // Capture status before disposal; no raw room/headset coordinates.
    if(inspectMode)setInspection(false);
    opticalIndex=0;ctx.setEndoscopeLighting?.(opticalModes[0].gain,opticalModes[0].glare);
    nativeQuad?.reset();
    if(physicalControl?.snapshot().armed)physicalControl.stop('XR_SESSION_ENDED');
    mrPlacement.reset();
    mrReticle.visible=false;
    replay.pause(performance.now());
    insertedShaft.hide();channel.reset();accessory.group.visible=false;
    toolProxyOverlay.visible=false;toolProxy.reset();videoScope.group.visible=false;shaftClutch.reset();clutchAxis=null;clutchHeld=false;
    xrCapabilities.end();
    procedureMetrics.suspend();health.end();reportButton.style.display='block';
    active=false;launchDock.hidden=false;rBro.setAnimationLoop(null);rBro.xr.setReferenceSpace?.(null);workspaceRelayout=false;handGrasped.fill(false);scope.velocity=0;
    camScope.fov=originalScopeFov;camScope.updateProjectionMatrix();
    if(ctSpatial){ctSpatial.dispose();ctSpatial=null;}
    leavePatientStation();worldMenu.visible=false;compactDock.visible=false;scopeScreen.visible=false;room.visible=false;routeOverlay.visible=false;
    mGrp.visible=originalModelVisibility;
    if(guidePlane)guidePlane.visible=false;
    // Restore the exact pre-session desktop scene and renderer blend state.
    scene.background=backgroundBeforeXR;scene.fog=fogBeforeXR;
    for(const [o,visible] of mrHidden)o.visible=visible;mrHidden=[];
    embodiment.reset();resetEmbodiedGrip();handSamples={};handMenuBlocked={left:false,right:false};
    if(clearBeforeXR)rBro.setClearColor(clearBeforeXR,clearAlphaBeforeXR);
    xrPresentation='vr';
    if(saved){
      mGrp.position.copy(saved.position);mGrp.quaternion.copy(saved.quaternion);
      mGrp.scale.copy(saved.scale);setExtIdle(saved.extIdle);
    }
    for(let i=0;i<2;i++)grab[i]=null;
    grabSource.fill(null);handSeen.fill(false);handPinched.fill(false);
    contactMarkers.forEach(m=>m.visible=false);
    summonGeneration++;summonTween=null;gesture=null;scopeSteering=null;
    laserLines.forEach(line=>{if(line)line.visible=false});
    laserTips.forEach(tip=>{if(tip)tip.visible=false});
    controllerSources.fill(null);controllerSide.fill(null);
    squeezeHeld.fill(false);aimTargets.fill(null);hoverPanels.fill(null);
    placed=false;viewerValid=false;
    button.textContent='ENTER PROCEDURAL XR';
    arButton.textContent='ENTER MR · PASSTHROUGH PREVIEW';
    syncTraceButton();inputStatusRefresh();
    onResize();
  });
  paint();
  const diagnostics=()=>({source:'original-pneumovr-scene',modelRole:getMeshMode(),
    mode,active,placed,viewerValid,monitor:scopeScreen.visible,
    quantitativeProcedure:procedureMetrics.snapshot(),
    originalSegments:SEGS.length,scopeBranch:scope.segId,clinicalMode:getClinMode(),
    scopeSteeringActive:Boolean(scopeSteering),
    monitorSource:'same-renderer-offscreen',monitorStatus,monitorFrames,monitorError,
    opticalDisplay:depthOpticsAvailable&&opticsEnabled?'PROCESSED_MESH':'RAW_SOURCE',
    monitorRetryFailures:monitorFailures,monitorRecoveries,
    videoBudget:monitorBudget.snapshot(),
    clinicalAndPhysicalReview:'EXTERNAL_EVIDENCE_REQUIRED_LOCAL_VALIDATION_LAB',
    monitorResolution:[scopeTarget.width,scopeTarget.height],
    physicalInstrumentBench:physicalBench?.snapshot(performance.now())||null,
    procedureStation:station.snapshot(),
    XR17CDisplayPath:displayPath.snapshot(),
    nativeQuadEvidence:endedXRGeometry?.nativeQuadEvidence||nativeQuad?.snapshot()||null,
    calibratedSimulatorInput:physicalControl?.snapshot()||null,
    XR17CInstrumentTrace:traceSummary(),
    XR17CTrackedControllerProxy:toolProxy.snapshot(),
    instrumentInputs:{tipH:scope.deflH,tipV:scope.deflV,shaftRollRad:scope.roll},
    contact:lastContact,grabFeedback,grabAttempts:{...attempts},
    activeControllerSides:controllerSide.slice(),
    menuHoverActions:hoverPanels.slice(),
    spatialMenuVisible:Boolean(worldMenu.visible),
    proceduralEquipment:{endoscopy:endoscopyTower.snapshot(),ultrasound:ultrasoundTower.snapshot(),independentEBUSDisplay:true,primaryImagePreserved:true,hardwareControl:false},
    spatialInterface:{main:mainSpatialSurface.snapshot(),guide:guideSpatialSurface.snapshot(),rail:railSurface.snapshot(),monitor:{depthM:monitorHousing.depthM,videoUnchanged:monitorHousing.videoUnchanged,ventCount:monitorHousing.ventCount,beveledChassis:monitorHousing.beveledChassis},laboratory:laboratory.snapshot(),monitorControls:monitorControlSurface.snapshot(),readingMode,monitorFocus,visualEvidence,composition:compositionEvidence},
    patientStation:{...patientStation.snapshot(),operatorZone:patientZone},
    spatialCalibration:spatialFinal||{...spatialCal.snapshot(),benchAligned,occlusion:planeMask.snapshot()},
    roomShellVisible:Boolean(room.visible),
    stagePresentation:stage.snapshot(),xrFoveation:qualityStatus,
    xrPresentation,
    mrPlacementEvidence:mrPlacement.snapshot(),
    xrCapabilityEvidence:xrCapabilities.snapshot(),
    procedureGuideVisible:Boolean(guidePlane?.visible),
    viewHistory:viewHistory.status(),pivotMode,axisMode,
    orientationLocked,spatialCT:ctSpatial?.snapshot()||null,
    scienceIntegrity:scienceSnapshot(),
    A4SourceProcedureStatus:getMeshMode()==='patient'
      ?'BLOCKED_NO_VERIFIED_CT_AIRWAY':'ORIGINAL_PROCEDURAL_REFERENCE_ONLY',
    panelPage:menuPage,
    worldSpaceSpanM:boundsCache?+(modelExtent()*mGrp.scale.x).toFixed(3):null,
    scaleClampEvents:scaleClamps,workspaceClampEvents:workspaceClamps,
    hoverStatus:aimTargets.map(t=>t?{hit:t.hit,reason:t.reason,distanceM:t.distanceM}:null),
    summonActive:Boolean(summonTween),
    handTracking:handSeen.slice(),grabSources:grabSource.slice(),
    physics:physicsSnapshot,physicsError,
    sessionHealth:health.snapshot(),
    referencePhysicsStatus:global.PulmoXRReferencePhysicsStatus||'not-loaded',
    referenceFocus:getReferenceFocus?.()||null,
    sourceAudit:getPatientAudit?.()||null,
    scaleMode,sourceScaleEligible:Boolean(getPatientAudit?.()?.sourceScaleEligible),
    clearance:clearanceState(),
    originalPathologyCount:Object.keys(ctx.pathoMeshes||{}).length,
    originalAnatomyMeshes:getAnatomyMeshes().length});
  global.pulmoProceduralXRDebug=diagnostics;
  return {button,diagnostics};
}
global.PulmoImmersiveXR={install};
})(window);
