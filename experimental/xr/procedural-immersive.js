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
    setClinMode,getClinMode,fireAction,enterBranch,advance,retract,hideBranchSel,
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
  let mrReticle=null,originalModelVisibility=true,mrMenuPositioned=false;
  const instrument=global.PulmoXRBronchoscopyStation;
  const originalScopeFov=camScope.fov;
  // External instrument inputs are evidence only, never scope control.
  const physicalBench=global.PulmoXRInstrumentTelemetry?.create?.()||null;
  const physicalCalibration=global.PulmoXRPhysicalCalibration?.create({parseFrame:global.PulmoXRInstrumentTelemetry.parseFrame});
  const physicalControl=physicalCalibration&&global.PulmoXRCalibratedControl?.create({calibration:physicalCalibration});
  let serialInput=null;
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
  const fingerChains=[
    ['wrist','thumb-metacarpal','thumb-phalanx-proximal','thumb-phalanx-distal','thumb-tip'],
    ...['index','middle','ring','pinky'].map(f=>['wrist',f+'-finger-metacarpal',f+'-finger-phalanx-proximal',f+'-finger-phalanx-intermediate',f+'-finger-phalanx-distal',f+'-finger-tip'])
  ];
  const jointNames=[...new Set(fingerChains.flat())];
  const handVisuals=[0,1].map(()=>{
    const group=new THREE.Group();group.visible=false;scene.add(group);
    const pts=new THREE.BufferGeometry();
    const bones=new THREE.BufferGeometry();
    pts.setAttribute('position',new THREE.BufferAttribute(new Float32Array(jointNames.length*3),3));
    bones.setAttribute('position',new THREE.BufferAttribute(new Float32Array(fingerChains.reduce((n,a)=>n+a.length-1,0)*6),3));
    const joints=new THREE.Points(pts,new THREE.PointsMaterial({color:0xc0e8e5,size:.012,transparent:true,opacity:.9,depthTest:true}));
    const links=new THREE.LineSegments(bones,new THREE.LineBasicMaterial({color:0x8fbac2,transparent:true,opacity:.72}));
    joints.frustumCulled=false;links.frustumCulled=false;group.add(joints,links);
    return {group,pts,bones,joints};
  });
  function updateHandVisual(i,source,frame,space){
    const v=handVisuals[i],poses=new Map();
    for(const name of jointNames){
      const joint=source.hand.get(name);
      const p=joint&&frame.getJointPose(joint,space)?.transform?.position;
      if(p&&[p.x,p.y,p.z].every(Number.isFinite))poses.set(name,p);
    }
    let n=0;for(const p of poses.values())v.pts.attributes.position.setXYZ(n++,p.x,p.y,p.z);
    v.pts.setDrawRange(0,n);v.pts.attributes.position.needsUpdate=true;
    let k=0;for(const chain of fingerChains)for(let j=1;j<chain.length;j++){
      const a=poses.get(chain[j-1]),b=poses.get(chain[j]);if(!a||!b)continue;
      v.bones.attributes.position.setXYZ(k++,a.x,a.y,a.z);
      v.bones.attributes.position.setXYZ(k++,b.x,b.y,b.z);
    }
    v.bones.setDrawRange(0,k);v.bones.attributes.position.needsUpdate=true;
    v.joints.material.color.setHex(grabSource[i]==='hand'?0x86edd4:0xc0e8e5);
    v.group.visible=n>0;
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
  function resetLobes(){
    lobeGrabs.fill(null);selectedLobe=null;spread=false;
    for(const l of lobes){l.mesh.position.copy(l.home);l.mesh.quaternion.copy(l.homeQuat);l.mesh.updateMatrixWorld(true);}
  }
  function setInspection(enabled){
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
    if(!['anatomy','procedure'].includes(next))return false;
    // A4: the actual original bronchoscope traverses SEGS (a reference
    // tree), not the uploaded CT GLB. Never silently call that patient-specific.
    if(next==='procedure'&&xrPresentation==='ar'&&!placed){
      grabFeedback='PLACE REFERENCE ANATOMY BEFORE ENTERING BRONCHOSCOPY';
      paint();return false;
    }
    if(next==='procedure'&&getMeshMode()==='patient'){
      grabFeedback='CT-SPECIFIC BRONCHOSCOPY BLOCKED / NO REGISTERED AIRWAY';
      showToast('The CT mesh has no reviewed airway centreline. Return to the procedural reference before bronchoscopy.','warn');
      paint();return false;
    }
    if(mode==='anatomy'&&next!=='anatomy'){finishViewChange();if(inspectMode)setInspection(false);}
    summonGeneration++;
    const previousMode=mode;
    mode=next;grab[0]=grab[1]=null;grabSource.fill(null);
    gesture=null;scopeSteering=null;summonTween=null;
    grabFeedback='AIM AT LUNG / HOLD TRIGGER OR GRIP';
    if(scopeScreen)scopeScreen.visible=next==='procedure';
    if(guidePlane)guidePlane.visible=next==='procedure'&&guideVisible;
    if(next==='anatomy'){scope.velocity=0;if(active&&physicalControl?.snapshot().armed&&previousMode==='procedure')physicalControl.stop('PROCEDURE_EXIT');}
    updateRouteOverlay();
    if(next==='procedure'){
      camScope.fov=station.snapshot().fovDegrees;
      camScope.updateProjectionMatrix();
      positionPanels();
    }
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
    if(!viewerValid)return false;
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
  function positionPanels(){
    if(!viewerValid)return;
    if(worldMenu){
      // Independent world-space panel, placed only on intentional layout
      // changes or recenter. It does not follow the viewer every XR frame.
      worldMenu.position.copy(head).addScaledVector(forward,.91).addScaledVector(right,-.72);
      worldMenu.position.y-=.02;
      worldMenu.lookAt(head);
      worldMenu.updateWorldMatrix(true,false);
    }
    const pose=station.snapshot().pose;
    function place(object,spec){
      if(!object)return;
      object.position.copy(head).addScaledVector(forward,spec.depth)
        .addScaledVector(right,spec.right);
      object.position.y+=spec.y;
      object.scale.setScalar(spec.scale);
      object.lookAt(head);object.updateWorldMatrix(true,false);
    }
    place(scopeScreen,pose.monitor);
    place(guidePlane,pose.guide);
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
    const side=controllerSide[i]||controllerSources[i]?.handedness;
    if(source==='controller'&&mode==='procedure'&&active&&side==='right'){
      if(input!=='grip')return false;
      const g=trackedController(i);if(!g)return false;
      g.updateWorldMatrix(true,false);
      scopeSteering={i,origin:g.getWorldQuaternion(new THREE.Quaternion()),
        deflH:scope.deflH,deflV:scope.deflV};
      grabFeedback='RIGHT GRIP / STEERING SOURCE SCOPE';return true;
    }
    if(!active||!placed||mode!=='anatomy'||grab[i])return false;
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
        const pressed=handPinched[i]?pinchDistance<.055:pinchDistance<=.026;
        if(pressed&&!handPinched[i]){
          const hit=panelIntersection(rayFrom(i));
          if(hit?.action){attempts.menuActions++;activate(hit.action);}
          else if(mode==='anatomy'){
            if(xrPresentation==='ar'&&!placed){confirmOrManualMR();}
            else startGrab(i,'hand');
          }else if(mode==='procedure'&&scope.atBranch){
            enterBranch(source.handedness==='right'?Math.min(1,(SEG_MAP[scope.segId]?.children.length||1)-1):0);
          }
        }else if(!pressed&&handPinched[i]&&grabSource[i]==='hand')release(i,'hand');
        handPinched[i]=pressed;
      }
    }
    for(let i=0;i<2;i++)if(!found[i]||!valid[i]){
      if(handSeen[i])health.event('handTrackingLost');
      handSeen[i]=false;handPinched[i]=false;handVisuals[i].group.visible=false;
      if(contactMarkers[i])contactMarkers[i].visible=false;
      if(grabSource[i]==='hand')release(i,'hand');
    }
  }
  function updateSteering(){
    if(mode!=='procedure'||!scopeSteering||physicalControl?.snapshot().armed)return;
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
      {object:guidePlane,actions:guideHitboxes,role:'procedure-guide'}
    ]){
      if(!panel.object?.visible)continue;
      panel.object.updateWorldMatrix(true,false);
      const hit=r.intersectObject(panel.object,false)[0];
      if(!hit?.uv||!Number.isFinite(hit.distance))continue;
      const x=hit.uv.x,y=1-hit.uv.y;
      const item=panel.actions.find(a=>x>=a.u&&x<=a.u+a.w&&
        y>=a.v&&y<=a.v+a.h);
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
    if(!active||!rayFrom(i))return;
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
    }else if(side==='right'){
      if(scope.atBranch){enterBranch(Math.min(1,(SEG_MAP[scope.segId]?.children.length||1)-1))}
      else fireAction();
    }else if(scope.atBranch)enterBranch(0);
  }
  const modes=['nav','biopsy','measure','annotate'];
  function activate(action){
    grabFeedback='MENU SELECTED / '+String(action||'').toUpperCase();
    if(action==='more'||action==='back'||action==='precision'||action==='back-expert'){
      menuPage=action==='more'||action==='back-expert'?'advanced':
        action==='precision'?(mode==='procedure'?'station':'precision'):'primary';
      actionHitbox=panelLayouts[menuPage];hoverPanels.fill(null);
      paint();return;
    }
    if(action==='mr-confirm'&&xrPresentation==='ar'){
      const result=mrPlacement.confirm();
      grabFeedback=result.accepted?'SURFACE SELECTED / CONFIRMING':
        'NO REAL SURFACE HIT / SELECT UNREGISTERED PREVIEW';
      paint();return;
    }
    if(action==='mr-reposition'&&xrPresentation==='ar'){
      release(0);release(1);scope.velocity=0;
      placed=false;mGrp.visible=false;mrPlacement.reposition();
      if(mrReticle)mrReticle.visible=false;
      grabFeedback='MR SCAN / AIM HEADSET AT A REAL SURFACE';
      paint();return;
    }
    if(action==='mr-preview'&&xrPresentation==='ar'){
      mrPlacement.preview();mGrp.visible=true;
      frameGroup();grabFeedback='UNREGISTERED FLOATING PREVIEW / NOT ROOM ANCHORED';
      paint();return;
    }
    if(action==='mr-status'&&xrPresentation==='ar'){
      grabFeedback='MR / '+mrPlacement.snapshot().state.toUpperCase()+
        ' / '+mrPlacement.snapshot().anchorStatus.toUpperCase();
      paint();return;
    }
    if(action==='inspect'){menuPage='inspection';actionHitbox=panelLayouts.inspection;paint();return;}
    if(action==='inspect-toggle'){if(inspectMode&&inspectionUnit==='lung'){inspectionUnit='lobe';spreadLobes('lobe');}else{inspectionUnit='lobe';setInspection(!inspectMode);}paint();return;}
    if(action==='inspect-spread'){spreadLobes('lobe');paint();return;}
    if(action==='inspect-lungs'){spreadLobes('lung');paint();return;}
    if(action==='inspect-restore'){setInspection(false);paint();return;}
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
      stage.setMode(modeNext);
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
      if(guidePlane)guidePlane.visible=mode==='procedure'&&guideVisible;
    }
    if(action==='guide-hide'){guideVisible=false;if(guidePlane)guidePlane.visible=false}
    if(mode==='procedure'&&!physicalControl?.snapshot().armed&&action==='scope-forward'){advance(.045);scope.velocity=0;}
    if(mode==='procedure'&&!physicalControl?.snapshot().armed&&action==='scope-back'){retract(.045);scope.velocity=0;}
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
      if(current.branches[index]?.enabled){
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
      positionPanels();
      worldMenu.visible=true;
      grabFeedback='SPATIAL CONTROLS READY / AIM AND TRIGGER';
    }
    attempts.panelToggles++;
    haptic(i,.20,36);paint();return true;
  }
  function applyAnalogAnatomy(side,x,y,dt){
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
        if(edge(0)&&!grab[i])onTrigger(i);
        if(!buttons[0].pressed&&grab[i]?.input==='trigger')release(i,'controller','trigger');
      }
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
        applyAnalogAnatomy(side,x,y,dt);
        continue;
      }
      if(mode!=='procedure')continue;
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
  const plinth=new THREE.Mesh(new THREE.CylinderGeometry(.72,.78,.055,64),
    new THREE.MeshStandardMaterial({color:0x253742,metalness:.16,roughness:.86,
      transparent:true,opacity:.45}));
  plinth.position.set(0,.03,0);room.add(plinth);
  const stage=global.PulmoXRStagePresence.create(THREE,room,{floor,grid,plinth,rings:spatialRings});
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
      {id:'exit',label:'EXIT XR',u:.52,v:.87,w:.43,h:.095}
    ],
    inspection:[
      {id:'inspect-toggle',label:'INDIVIDUAL LOBES',u:.05,v:.54,w:.43,h:.12},
      {id:'inspect-spread',label:'SEPARATE LOBES',u:.52,v:.54,w:.43,h:.12},
      {id:'inspect-lungs',label:'SEPARATE LUNGS',u:.05,v:.69,w:.43,h:.12},
      {id:'inspect-restore',label:'REASSEMBLE',u:.52,v:.69,w:.43,h:.12},
      {id:'back-expert',label:'BACK',u:.05,v:.84,w:.43,h:.12},
      {id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}
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
      {id:'trace-open',label:'INSTRUMENT TRACE',u:.05,v:.76,w:.43,h:.095},
      {id:'optical-light',label:'LIGHTING',u:.52,v:.76,w:.43,h:.095},
      {id:'back-expert',label:'BACK',u:.05,v:.87,w:.43,h:.095},
      {id:'exit',label:'EXIT XR',u:.52,v:.87,w:.43,h:.095}
    ],
    mr:[
      {id:'mr-confirm',label:'PLACE ON SURFACE',u:.05,v:.54,w:.43,h:.12},
      {id:'mr-reposition',label:'REPOSITION',u:.52,v:.54,w:.43,h:.12},
      {id:'mr-preview',label:'FLOATING PREVIEW',u:.05,v:.69,w:.43,h:.12},
      {id:'mr-status',label:'PLACEMENT STATUS',u:.52,v:.69,w:.43,h:.12},
      {id:'back',label:'BACK',u:.05,v:.84,w:.43,h:.12},
      {id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}
    ],
    trace:[
      {id:'trace-play',label:'PLAY TRACE',u:.05,v:.54,w:.43,h:.095},
      {id:'trace-pause',label:'PAUSE',u:.52,v:.54,w:.43,h:.095},
      {id:'trace-prev',label:'PREV SAMPLE',u:.05,v:.65,w:.43,h:.095},
      {id:'trace-next',label:'NEXT SAMPLE',u:.52,v:.65,w:.43,h:.095},
      {id:'tool-proxy',label:'TRACKED TOOL PROXY',u:.05,v:.76,w:.43,h:.095},
      {id:'trace-clear',label:'CLEAR TRACE',u:.52,v:.76,w:.43,h:.095},
      {id:'trace-back',label:'BACK TO STATION',u:.05,v:.87,w:.43,h:.095},
      {id:'exit',label:'EXIT XR',u:.52,v:.87,w:.43,h:.095}
    ],
    precision:[
      {id:'undo',label:'UNDO',u:.05,v:.54,w:.43,h:.12},
      {id:'redo',label:'REDO',u:.52,v:.54,w:.43,h:.12},
      {id:'pivot',label:'PIVOT CENTER',u:.05,v:.69,w:.43,h:.12},
      {id:'axis',label:'AXIS YAW',u:.52,v:.69,w:.43,h:.12},
      {id:'back-expert',label:'BACK',u:.05,v:.84,w:.43,h:.12},
      {id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}
    ]
  };
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
    c.clearRect(0,0,w,h);c.fillStyle='#14222c';c.fillRect(0,0,w,h);
    c.fillStyle='#7ea5b3';c.fillRect(0,0,w,8);
    c.textAlign='left';c.fillStyle='#afc9d1';c.font='600 27px system-ui';
    c.fillText(('PULMOSIMU XR-14 / '+(hoverPanels.find(Boolean)?('MENU: '+hoverPanels.find(Boolean)):active?grabFeedback:'ORIGINAL ENGINE')).slice(0,58),48,64);
    c.fillStyle='#e8f0f2';c.font='600 48px system-ui';
    c.fillText(menuPage==='mr'?'Mixed-reality placement':
      menuPage==='trace'?'Physical instrument trace':
      menuPage==='station'?'Bronchoscopy workstation':
      menuPage==='inspection'?'Anatomical disassembly':
      menuPage==='precision'?'Precision anatomy controls':
      menuPage==='advanced'?'Research instruments':
      menuPage==='ct'?'Source CT / spatial sections':
      mode==='procedure'?'Bronchoscopy':'Anatomical exploration',48,131);
    c.fillStyle='#d3e0e5';c.font='28px system-ui';
    c.fillText((menuPage==='trace'?
      (replay.snapshot().loaded?
        'READ-ONLY RECORD · '+replay.snapshot().reviewIndex+' / '+(replay.snapshot().total-1):
        'NO BENCH TRACE · LOAD FROM DESKTOP FIRST')
      :mode==='procedure'
      ?('SEG '+scope.segId+' · '+(SEG_MAP[scope.segId]?.name||'—'))
      :menuPage==='ct'&&ctSpatial
        ?'CT '+ctAxis.toUpperCase()+' / slice '+ctSpatial.snapshot().sections[ctAxis]
        :inspectMode?'INSPECTION · '+(lobes.find(l=>l.id===selectedLobe)?.name||'GRAB A LOBE'):'Original anatomical scene · '+scaleMode+' scale'+(referenceFocus?' · '+referenceFocus:'' )).slice(0,51),48,189);
    c.fillStyle='#a8c4cc';c.font='23px system-ui';
    c.fillText(menuPage==='trace'
       ?'PHYSICAL MEASUREMENT REVIEW · NEVER DRIVES THE ORIGINAL SCOPE'
       :mode==='procedure'
       ?'L STICK: INSERT / ROLL · R STICK: TIP · R GRIP: STEER'
       :xrPresentation==='ar'&&!placed
         ?'MR: SCAN SURFACE · TRIGGER TO PLACE · MENU: OPTIONS'
         :'GRAB TO MOVE · R STICK: ROTATE / DEPTH · LEFT Y: MENU',48,236);
    c.fillStyle='#95bdc8';c.font='23px system-ui';
    c.fillText((menuPage==='trace'
      ?('TRACE QA '+replay.snapshot().engineeringStatus+' / '+
        (replay.snapshot().observation?.tracking?'TRACKING OK':'TRACKING NOT VERIFIED'))
      :xrPresentation==='ar'?
      'MR '+mrPlacement.snapshot().state.toUpperCase()+' / '+
      mrPlacement.snapshot().anchorStatus.toUpperCase()+' · UNVERIFIED ROOM DEPTH'
      :source).slice(0,83),48,274);
    c.fillStyle='#d8e1e4';c.font='22px system-ui';
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
    c.fillText((menuPage==='trace'&&traceObs
      ?'MEAS INS '+traceObs.measured.insertionMm.toFixed(1)+' mm · ERR '+
        traceObs.residual.insertionMm.toFixed(1)+' mm · TIP H '+traceObs.measured.tipHDeg.toFixed(1)+'°'
      :menuPage==='trace'
        ?'NO IMPORTED EXTERNAL REFERENCE RECORD · SOURCE SCOPE UNCHANGED'
        :clinicalLabel).slice(0,83),48,311);
    for(const a of actionHitbox){
      const x=a.u*w,y=a.v*h,width=a.w*w,height=a.h*h;
      const selected=a.id===mode||(a.id==='scale'&&scaleMode==='source-derived');
      const hovered=hoverPanels.includes(a.id);
      const available=(a.id!=='scale'||Boolean((audit?.sourceScaleEligible||getCTStudy?.())&&role==='patient')) &&
        (a.id!=='undo'||viewHistory.status().undo>0) &&
        (a.id!=='redo'||viewHistory.status().redo>0) &&
        (!a.id.startsWith('trace-')||a.id==='trace-open'||a.id==='trace-back'||replay.snapshot().loaded) &&
        (a.id!=='procedure'||role==='reference');
      const settings=station.snapshot();
      const label=a.id==='trace-play'&&(replay.snapshot().playing)?'PLAYING TRACE':
        a.id==='airway-map'?(routeVisible?'HIDE AIRWAY MAP':'SHOW AIRWAY MAP'):
        a.id==='optical-light'?'LIGHT '+opticalModes[opticalIndex].name:
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
      c.fillStyle=hovered?'#476774':selected?'#345463':available?'#213641':'#1c2a32';
      c.fillRect(x,y,width,height);
      c.strokeStyle=hovered?'#f1d3a5':selected?'#adcdd5':available?'#4b6673':'#314652';
      c.lineWidth=2;c.strokeRect(x,y,width,height);
      c.fillStyle=available?'#e5eef1':'#6f8791';
      c.font='600 27px system-ui';
      c.fillText(label,x+23,y+height*.66);
    }
    panelTex.needsUpdate=true;
    paintGuide();
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
  let monitorStatus='not-started',monitorError=null,monitorFrames=0;
  let monitorFailures=0,monitorRecoveries=0,monitorRetryAt=0;
  let monitorBudget=instrument.videoCostGovernor();
  const monitorRoot=new THREE.Group();
  monitorRoot.name='xr16-independent-endoscopy-display';
  const monitorFrame=new THREE.Mesh(new THREE.PlaneGeometry(1.00,.665),
    new THREE.MeshBasicMaterial({color:0x0b161e,side:THREE.DoubleSide}));
  monitorFrame.position.z=-.012;monitorRoot.add(monitorFrame);
  // Keep anatomical source pixels clear: identification and instrument data
  // are physically above/below the image, never drawn across the lumen.
  const monitorImage=new THREE.Mesh(new THREE.PlaneGeometry(.94,.529),
    new THREE.MeshBasicMaterial({map:scopeTexture,side:THREE.DoubleSide,depthTest:false}));
  monitorImage.position.y=-.012;
  monitorImage.visible=false;monitorImage.renderOrder=30;monitorRoot.add(monitorImage);
  function monitorStrip(y){
    const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=74;
    const ctx=canvas.getContext('2d');
    const texture=new THREE.CanvasTexture(canvas);texture.encoding=THREE.sRGBEncoding;
    const mesh=new THREE.Mesh(new THREE.PlaneGeometry(.94,.056),
      new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,
        depthTest:false,depthWrite:false}));
    mesh.position.set(0,y,.004);mesh.renderOrder=31;
    monitorRoot.add(mesh);
    return {canvas,ctx,texture};
  }
  const topStrip=monitorStrip(.294),bottomStrip=monitorStrip(-.303);
  function paintMonitorChrome(){
    const settings=station.snapshot();
    const seg=SEG_MAP[scope.segId];
    function draw(strip,left,rightText,small=false){
      const c=strip.ctx,w=strip.canvas.width,h=strip.canvas.height;
      c.clearRect(0,0,w,h);c.fillStyle='#14242d';c.fillRect(0,0,w,h);
      c.fillStyle='#8fb5c0';c.fillRect(0,0,w,3);
      c.fillStyle='#e3edf0';c.font='600 '+(small?23:27)+'px system-ui';
      c.textAlign='left';c.fillText(left.slice(0,75),18,49);
      c.textAlign='right';c.fillStyle='#a4c1ca';
      c.fillText(rightText.slice(0,45),w-17,49);
      strip.texture.needsUpdate=true;
    }
    draw(topStrip,(seg?.name||'REFERENCE AIRWAY').toUpperCase(),'GEN '+(seg?.gen??'—')+' · '+monitorStatus.toUpperCase());
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
    c.clearRect(0,0,w,h);
    c.fillStyle='#112028';c.fillRect(0,0,w,h);
    c.fillStyle='#6a9eab';c.fillRect(0,0,w,6);
    c.textAlign='left';
    c.fillStyle='#a9c3cc';c.font='600 23px system-ui';
    c.fillText('PULMOSIMU  /  PROCEDURE GUIDE',42,50);
    c.textAlign='right';c.fillStyle='#bad1da';
    c.fillText('STEP '+(guideStep+1)+'/5  ·  '+guideStages[guideStep],w-42,50);
    c.textAlign='left';c.fillStyle='#f0f5f7';
    c.font='550 42px system-ui';c.fillText(state.heading,42,116);
    c.font='550 31px system-ui';
    c.fillText(state.location.slice(0,51),42,165);
    c.fillStyle='#9cb7c2';c.font='26px system-ui';
    c.fillText('GEN '+(state.generation??'—')+
      '    SEGMENT '+scope.segId+'    PATH '+state.progressPercent+'%',42,213);
    c.fillStyle='#304651';c.fillRect(42,236,w-84,8);
    c.fillStyle='#a4d3d9';c.fillRect(42,236,(w-84)*state.progressPercent/100,8);
    c.fillStyle='#dce8eb';c.font='600 26px system-ui';
    c.fillText(state.tool+'   /   '+state.monitor,42,289);
    c.fillStyle='#aebfc7';c.font='24px system-ui';
    c.fillText(route.labels.join(' > ').slice(0,73),42,337);
    c.fillStyle=state.clearance.includes('BLOCKING')?'#e8b99d':'#93adba';
    c.font='23px system-ui';
    c.fillText('ROUTE '+Math.round(route.distanceMm)+' MM · '+state.clearance.slice(0,47),42,378);
    const branches=Array.from({length:Math.max(2,state.branches.length)},(_,i)=>({id:'guide-branch'+i,label:state.branches[i]?.label||'NO SOURCE BRANCH',disabled:!state.branches[i]?.enabled,u:.04+i*.92/Math.max(2,state.branches.length),v:.60,w:.88/Math.max(2,state.branches.length),h:.14}));
    guideHitboxes=[
      ...branches,
      {id:'scope-back',label:'RETRACT',disabled:false,u:.04,v:.76,w:.43,h:.10},
      {id:'scope-forward',label:'ADVANCE',disabled:false,u:.52,v:.76,w:.43,h:.10},
      {id:'guide-prev',label:'PREVIOUS',disabled:guideStep===0,u:.04,v:.88,w:.20,h:.08},
      {id:'guide-next',label:'NEXT STEP',disabled:guideStep===4,u:.27,v:.88,w:.20,h:.08},
      {id:'guide-tool',label:'TOOL MODE',disabled:false,u:.50,v:.88,w:.21,h:.08},
      {id:'guide-hide',label:'HIDE',disabled:false,u:.74,v:.88,w:.22,h:.08}
    ];
    for(const a of guideHitboxes){
      const x=a.u*w,y=a.v*h,width=a.w*w,height=a.h*h;
      const hovered=hoverPanels.includes(a.id);
      c.fillStyle=a.disabled?'#17252e':hovered?'#436b76':'#223843';
      c.fillRect(x,y,width,height);
      c.strokeStyle=hovered?'#e4cba5':a.disabled?'#273740':'#4b6975';
      c.lineWidth=2;c.strokeRect(x,y,width,height);
      c.fillStyle=a.disabled?'#647d88':'#e6f0f3';
      c.font='600 23px system-ui';
      c.font='600 '+(a.w<.31?20:23)+'px system-ui';
      c.fillText(a.label.slice(0,25),x+12,y+height*.65,width-24);
    }
    c.fillStyle='#839da7';c.font='19px system-ui';
    c.fillText(state.provenance,42,652);
    // Real current instrument commands, deliberately separate from the
    // endoscopic image. Angles are reference actuator inputs, not measured
    // patient mechanics.
    c.fillStyle='#c7d7dc';c.font='23px system-ui';
    c.fillText('TIP H '+Math.round(scope.deflH*CFG.deflMax)+'°  /  V '+
      Math.round(scope.deflV*CFG.deflMax)+'°   ·   SHAFT '+
      Math.round(scope.roll*180/Math.PI)+'°',42,505);
    guideTex.needsUpdate=true;
    paintMonitorChrome();
  }
  function captureFeed(now){
    if(mode!=='procedure'||now-lastFeed<station.snapshot().capture.intervalMs)return;
    if(monitorStatus==='unavailable'&&now<monitorRetryAt)return;
    lastFeed=now;
    const previousTarget=rBro.getRenderTarget();
    const previousXR=rBro.xr.enabled;
    const previousAspect=camScope.aspect;
    const menuWasVisible=worldMenu.visible,roomWasVisible=room.visible;
    const guideWasVisible=guidePlane?.visible||false;
    const monitorWasVisible=scopeScreen.visible;
    const originalBg=scene.background;
    const hidden=[...(ctx.getEndoscopeHiddenObjects?.()||[]),routeOverlay,toolProxyOverlay,...handVisuals.map(v=>v.group),...contactMarkers,...laserTips,...laserLines].filter(Boolean).map(o=>[o,o.visible]);
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
      camScope.updateProjectionMatrix();
      rBro.setRenderTarget(scopeTarget);
      rBro.clear(true,true,true);
      rBro.render(scene,camScope);
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
      camScope.aspect=previousAspect;camScope.updateProjectionMatrix();
      scene.background=originalBg;
      worldMenu.visible=menuWasVisible;
      if(guidePlane)guidePlane.visible=guideWasVisible;
      room.visible=roomWasVisible;
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
      laser.material.color.setHex(choice.kind==='panel'?0xeecb96:
        choice.kind==='anatomy'?0x80dfd8:0x7896a1);
      laser.material.opacity=choice.kind==='none'?.25:.95;
      const point=choice.kind==='panel'?panel?.point:
        choice.kind==='anatomy'?organ?.point:null;
      if(point){
        tip.position.copy(point);tip.visible=true;
        tip.material.color.setHex(choice.kind==='panel'?0xeecb96:0x80dfd8);
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
    release:'E1–E4 · Integrated PulmoSIMU Research XR (UNVALIDATED)',
    session:health.snapshot(),
    interactionAttempts:{...attempts},
    proceduralGuidance:{visible:guideVisible,manualDemoStep:guideStep,
      actions:guideActions,lastSourceState:guideLastState},
    procedureStation:station.snapshot(),
    XR17CDisplayPath:displayPath.snapshot(),
    nativeQuadEvidence:endedXRGeometry?.nativeQuadEvidence||nativeQuad?.snapshot()||null,
    calibratedSimulatorInput:physicalControl?.snapshot()||null,
    XR17CInstrumentTrace:traceSummary(),
    XR17CTrackedControllerProxy:toolProxy.snapshot(),
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
    const a=physicalControl?.snapshot(),s=serialInput?.status();
    inputStatus.textContent=(calibrationLoaded?'Calibration loaded':'Calibration required')+' · '+
      (s?.state||'DISCONNECTED')+' · '+(a?.reason||'UNAVAILABLE');
  };
  if(physicalCalibration&&physicalControl&&global.PulmoXRReadOnlySerial){
   serialInput=global.PulmoXRReadOnlySerial.create({navigatorObject:navigator,
    onLine:(line,now)=>{if(physicalControl.snapshot().armed){
      const result=physicalCalibration.ingest(line,now);
      if(!result.accepted)physicalControl.stop(result.reason);
    }},
    onState:s=>{if(s.state==='DISCONNECTED'&&physicalControl.snapshot().armed)
      physicalControl.stop('SERIAL_DISCONNECTED');inputStatusRefresh();}
   });
  }
  instrumentSetup.querySelector('input').addEventListener('change',async e=>{
    try{physicalControl?.stop('CALIBRATION_REPLACED');calibrationLoaded=false;
     const file=e.target.files?.[0];if(!file||file.size>100000)throw Error('INVALID_CALIBRATION_FILE');
     physicalCalibration.configure(JSON.parse(await file.text()));calibrationLoaded=true;
     inputStatusRefresh();
    }catch(err){inputStatus.textContent='Calibration rejected: '+String(err.message).slice(0,100)}
    finally{e.target.value=''}
  });
  instrumentSetup.querySelector('[data-op="connect"]').addEventListener('click',async()=>{
    if(serialConnecting)return;
    serialConnecting=true;
    try{if(!serialInput)throw Error('LOCAL_SERIAL_UNAVAILABLE');await serialInput.connect()}
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
    physicalControl?.stop('OPERATOR_STOP');void serialInput?.disconnect();inputStatusRefresh();
  });
  addEventListener('pagehide',()=>{physicalControl?.stop('PAGE_EXIT');void serialInput?.disconnect()});
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
  arButton.textContent='ENTER MR · PASSTHROUGH PREVIEW';
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
  function refreshXRAvailability(info){
    const caps=info||xrCapabilities.snapshot(),q=caps.sessions;
    button.disabled=q.vr==='unsupported'||q.vr==='unavailable'||q.vr==='error';
    if(button.disabled)button.textContent='PROCEDURAL VR · UNAVAILABLE';
    else button.textContent='ENTER PROCEDURAL XR';
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
    health.begin(rBro.xr.getSession()?.frameRate??null);
    displayPath.inspect({session:rBro.xr.getSession(),renderer:rBro,
      gl:rBro.getContext?.()||null});
    nativeQuad?.bind(rBro);
    replay.pause(performance.now());
    traceButton.style.display='none';instrumentSetup.style.display='none';
    qualityStatus=configureComfortRendering();
    reportButton.style.display='none';
    active=true;placed=false;viewerValid=false;lastFrame=0;lastFeed=0;
    lastPanelUpdate=0;scaleMode='comfort';endedXRGeometry=null;
    guideVisible=true;guideStep=0;guideActions=0;
    station.reset();
    toolProxy.reset();toolProxyOverlay.visible=false;
    physicalBench?.reset();
    scopeTarget.setSize(station.snapshot().capture.width,station.snapshot().capture.height);
    camScope.fov=station.snapshot().fovDegrees;camScope.updateProjectionMatrix();
    viewHistory.clear();analogChangedAt=0;pivotMode='center';axisMode='yaw';
    orientationLocked=false;orientationAnchor=null;ctAxis='axial';
    if(ctSpatial){ctSpatial.dispose();ctSpatial=null;}
    menuPage='primary';actionHitbox=panelLayouts.primary;
    stage.setMode('studio');
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
    worldMenu.visible=true;scopeScreen.visible=false;room.visible=xrPresentation!=='ar';
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
        // Tracking loss: never continue the last object or instrument command.
        health.event('viewerPoseLost');health.endFrame({poseAvailable:false});
        nativeQuad?.update({visible:false});
        if(physicalControl?.snapshot().armed)physicalControl.stop('TRACKING_LOST');
        viewerValid=false;scope.velocity=0;release(0);release(1);
        toolProxyOverlay.visible=false;
        if(mrReticle)mrReticle.visible=false;
        handSeen.fill(false);handPinched.fill(false);handVisuals.forEach(v=>v.group.visible=false);
        contactMarkers.forEach(marker=>marker.visible=false);
        laserLines.forEach(line=>{if(line)line.visible=false});
        laserTips.forEach(tip=>{if(tip)tip.visible=false});
        return;
      }
      if(!placed&&xrPresentation!=='ar'){
        room.position.copy(head).setY(head.y-1.18);
        const center=head.clone().addScaledVector(forward,1.5);
        plinth.position.x=center.x-head.x;plinth.position.z=center.z-head.z;
        frameGroup();
      }
      if(xrPresentation==='ar'){
        const hit=mrPlacement.scan(frame,rBro.xr.getReferenceSpace());
        mrReticle.visible=Boolean(hit?.kind==='candidate'&&!placed);
        if(hit?.kind==='candidate')
          mrReticle.position.set(hit.point.x,hit.point.y+.006,hit.point.z);
        if(hit?.kind==='placed'){
          mrReticle.visible=false;
          placeOnMRSurface(hit.point);
        }else if(hit?.kind==='anchor-delta'&&placed){
          const delta=hit.delta;
          if([delta.x,delta.y,delta.z].every(Number.isFinite)&&
             Math.hypot(delta.x,delta.y,delta.z)<.15){
            mGrp.position.add(new THREE.Vector3(delta.x,delta.y,delta.z));
            mGrp.updateWorldMatrix(true,false);
          }
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
      poll(dt);
      if(analogChangedAt&&!grab.some(Boolean)&&performance.now()-analogChangedAt>380)
        finishViewChange();
      updateTrackedHands(frame);updateSummon();updateGrab();updateSteering();
      updateToolProxy();updatePointerAim(time);
      // The identical SI-unit R3 solver used in the research laboratory;
      // explicitly reference-only, never CT-fitted or mesh-deforming.
      if(getMeshMode()==='reference'&&global.PulmoXRReferencePhysics?.step){
        try{physicsSnapshot=global.PulmoXRReferencePhysics.step(dt);physicsError=null}
        catch(error){physicsSnapshot=null;physicsError=String(error?.message||error).slice(0,140)}
      }else physicsSnapshot=null;
      // Only a calibrated, live, local device can request simulator motion.
      // Left grip is a dead-man; no replay, no hardware write, no patient input.
      if(physicalControl?.snapshot().armed){
        scope.velocity=0;scopeSteering=null;
        const deadman=Array.from(rBro.xr.getSession()?.inputSources||[]).some(s=>
          s.handedness==='left'&&s.gamepad?.buttons?.[1]?.pressed===true);
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
      if(mode==='procedure'&&Math.abs(scope.velocity)>.0001){
        const step=Math.min(1.5,Math.max(.1,dt*60));
        const distance=scope.velocity*step;
        if(distance>0)advance(distance);else retract(-distance);
        scope.velocity*=Math.pow(scope.velocityDecay,step);
      }
      updateScope();updateRouteOverlay();
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
      const nativeVisible=mode==='procedure'&&monitorStatus==='live'&&xrPresentation==='vr'&&nativeLayersOptIn;
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
    // Snapshot before restoring the original desktop scene scale.
    endedXRGeometry={
      modelSpanM:boundsCache?+(modelExtent()*mGrp.scale.x).toFixed(3):null,
      scaleMode,scaleClamps,workspaceClamps,
      finalGrabStatus:grabFeedback,
      stagePresentation:stage.snapshot(),
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
    // Capture status before disposal; no raw room/headset coordinates.
    if(inspectMode)setInspection(false);
    opticalIndex=0;ctx.setEndoscopeLighting?.(opticalModes[0].gain,opticalModes[0].glare);
    nativeQuad?.reset();
    if(physicalControl?.snapshot().armed)physicalControl.stop('XR_SESSION_ENDED');
    mrPlacement.reset();
    mrReticle.visible=false;
    replay.pause(performance.now());
    toolProxyOverlay.visible=false;toolProxy.reset();
    xrCapabilities.end();
    health.end();reportButton.style.display='block';
    active=false;rBro.setAnimationLoop(null);scope.velocity=0;
    camScope.fov=originalScopeFov;camScope.updateProjectionMatrix();
    if(ctSpatial){ctSpatial.dispose();ctSpatial=null;}
    worldMenu.visible=false;scopeScreen.visible=false;room.visible=false;routeOverlay.visible=false;
    mGrp.visible=originalModelVisibility;
    if(guidePlane)guidePlane.visible=false;
    // Restore the exact pre-session desktop scene and renderer blend state.
    scene.background=backgroundBeforeXR;scene.fog=fogBeforeXR;
    for(const [o,visible] of mrHidden)o.visible=visible;mrHidden=[];
    handVisuals.forEach(v=>v.group.visible=false);
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
    originalSegments:SEGS.length,scopeBranch:scope.segId,clinicalMode:getClinMode(),
    scopeSteeringActive:Boolean(scopeSteering),
    monitorSource:'same-renderer-offscreen',monitorStatus,monitorFrames,monitorError,
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
