/* PulmoSIMU · Procedural Immersive XR
 * Three.js r128 classic-script adapter for the ORIGINAL PNEUMO·VR scene.
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
  const limits=global.PulmoXRInteractionMath;
  const health=global.PulmoXRSessionHealth.createSessionHealth();
  const station=global.PulmoXRBronchoscopyStation.create();
  const instrument=global.PulmoXRBronchoscopyStation;
  const originalScopeFov=camScope.fov;
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
  let clearBeforeXR=null,clearAlphaBeforeXR=1,arButton=null;
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
  const surfaceTarget=()=>getAnatomyMeshes().filter(o=>o&&o.visible&&o.isObject3D);
  const trackedController=i=>controllerSources[i]?.gripSpace?grips[i]:hands[i];
  const effectiveGrip=i=>grabSource[i]==='hand'?handProxies[i]:trackedController(i);
  const release=(i,source=null,input=null)=>{
    if(source&&grabSource[i]&&grabSource[i]!==source)return;
    if(input&&grab[i]&&grab[i].input!==input)return;
    const endedGrab=grab[i];
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
  function setMode(next){
    if(!['anatomy','procedure'].includes(next))return false;
    // A4: the actual original bronchoscope traverses SEGS (a reference
    // tree), not the uploaded CT GLB. Never silently call that patient-specific.
    if(next==='procedure'&&getMeshMode()==='patient'){
      grabFeedback='CT-SPECIFIC BRONCHOSCOPY BLOCKED / NO REGISTERED AIRWAY';
      showToast('The CT mesh has no reviewed airway centreline. Return to the procedural reference before bronchoscopy.','warn');
      paint();return false;
    }
    if(mode==='anatomy'&&next!=='anatomy')finishViewChange();
    summonGeneration++;
    mode=next;grab[0]=grab[1]=null;grabSource.fill(null);
    gesture=null;scopeSteering=null;summonTween=null;
    grabFeedback='AIM AT LUNG / HOLD TRIGGER OR GRIP';
    if(scopeScreen)scopeScreen.visible=next==='procedure';
    if(guidePlane)guidePlane.visible=next==='procedure'&&guideVisible;
    if(next==='anatomy')scope.velocity=0;
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
  function frameGroup(distance=1.05,width=.68){
    if(!viewerValid)return false;
    if(placed){finishViewChange();beginViewChange();}
    summonGeneration++;summonTween=null;
    release(0);release(1);
    if(placed){finishViewChange();beginViewChange();}
    const bounds=sourceBounds();boundsCache=bounds;
    const scale=limits.clampScale(Math.min(width/Math.max(.0001,bounds.size.x),
      .65/Math.max(.0001,bounds.size.y)),modelExtent());
    const center=head.clone().addScaledVector(forward,distance).add(new THREE.Vector3(0,-.1,0));
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
    if(!grip||!grab[i])return;
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
    const remote=source==='controller'&&!near.hit
      ?chosen.organ:null;
    if(!near.hit&&!remote?.hit){
      attempts.rejected++;
      grabFeedback=remote?.reason==='no-anatomy-source'
        ?'NO ANATOMY SOURCE / SELECT REFERENCE'
        :'AIM LASER AT LUNG THEN HOLD TRIGGER OR GRIP';
      lastContact={hit:false,reason:remote?.reason||near.reason||'no-contact',
        distanceM:null};
      paint();return false;
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
        proxy.position.copy(pA).add(pB).multiplyScalar(.5);
        const wrist=source.hand.get('wrist');
        const w=wrist?frame.getJointPose(wrist,space):null;
        if(w?.transform?.orientation){
          const o=w.transform.orientation;
          proxy.quaternion.set(o.x,o.y,o.z,o.w).normalize();
        }else proxy.quaternion.identity();
        proxy.updateMatrixWorld(true);valid[i]=true;
        handSeen[i]=true;
        if(contactMarkers[i]){
          contactMarkers[i].position.copy(proxy.position);
          contactMarkers[i].visible=mode==='anatomy';
        }
        const pinchDistance=pA.distanceTo(pB);
        const pressed=handPinched[i]?pinchDistance<.055:pinchDistance<=.026;
        if(pressed&&!handPinched[i]&&mode==='anatomy')startGrab(i,'hand');
        else if(!pressed&&handPinched[i]&&grabSource[i]==='hand')release(i,'hand');
        handPinched[i]=pressed;
      }
    }
    for(let i=0;i<2;i++)if(!found[i]||!valid[i]){
      if(handSeen[i])health.event('handTrackingLost');
      handSeen[i]=false;handPinched[i]=false;
      if(contactMarkers[i])contactMarkers[i].visible=false;
      if(grabSource[i]==='hand')release(i,'hand');
    }
  }
  function updateSteering(){
    if(mode!=='procedure'||!scopeSteering)return;
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
    if(action==='guide-next')guideStep=Math.min(4,guideStep+1);
    if(action==='guide-prev')guideStep=Math.max(0,guideStep-1);
    if(action==='guide-tool'){
      setClinMode(modes[(modes.indexOf(getClinMode())+1)%modes.length]);
      guideActions++;
    }
    if(action==='guide-branch0'||action==='guide-branch1'){
      const index=action==='guide-branch0'?0:1;
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
      const i=controllerSources.indexOf(src);
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
      const x=Math.abs(axes[n]||0)<.13?0:axes[n],y=Math.abs(axes[n+1]||0)<.13?0:axes[n+1];
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
      {id:'tool',label:'INSTRUMENT',u:.05,v:.54,w:.43,h:.12},
      {id:'scale',label:'SOURCE SCALE',u:.52,v:.54,w:.43,h:.12},
      {id:'guide',label:'PROCEDURE GUIDE',u:.05,v:.69,w:.43,h:.12},
      {id:'precision',label:'VIEW CONTROLS',u:.52,v:.69,w:.43,h:.12},
      {id:'back',label:'BACK',u:.05,v:.84,w:.43,h:.12},
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
      {id:'station-layout',label:'LAYOUT',u:.05,v:.54,w:.43,h:.12},
      {id:'station-video',label:'VIDEO DETAIL',u:.52,v:.54,w:.43,h:.12},
      {id:'station-optics',label:'ENDOSCOPE VIEW',u:.05,v:.69,w:.43,h:.12},
      {id:'station-control',label:'FINE CONTROL',u:.52,v:.69,w:.43,h:.12},
      {id:'back-expert',label:'BACK',u:.05,v:.84,w:.43,h:.12},
      {id:'exit',label:'EXIT XR',u:.52,v:.84,w:.43,h:.12}
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
    c.fillText(menuPage==='station'?'Bronchoscopy workstation':
      menuPage==='precision'?'Precision anatomy controls':
      menuPage==='advanced'?'Research instruments':
      menuPage==='ct'?'Source CT / spatial sections':
      mode==='procedure'?'Bronchoscopy':'Anatomical exploration',48,131);
    c.fillStyle='#d3e0e5';c.font='28px system-ui';
    c.fillText((mode==='procedure'
      ?('SEG '+scope.segId+' · '+(SEG_MAP[scope.segId]?.name||'—'))
      :menuPage==='ct'&&ctSpatial
        ?'CT '+ctAxis.toUpperCase()+' / slice '+ctSpatial.snapshot().sections[ctAxis]
        :'Original anatomical scene · '+scaleMode+' scale'+(referenceFocus?' · '+referenceFocus:'' )).slice(0,51),48,189);
    c.fillStyle='#a8c4cc';c.font='23px system-ui';
    c.fillText(mode==='procedure'
       ?'L STICK: INSERT / ROLL · R STICK: TIP · R GRIP: STEER'
       :'GRAB TO MOVE · R STICK: ROTATE / DEPTH · LEFT Y: MENU',48,236);
    c.fillStyle='#95bdc8';c.font='23px system-ui';
    c.fillText(source,48,274);
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
    c.fillText(clinicalLabel.slice(0,83),48,311);
    for(const a of actionHitbox){
      const x=a.u*w,y=a.v*h,width=a.w*w,height=a.h*h;
      const selected=a.id===mode||(a.id==='scale'&&scaleMode==='source-derived');
      const hovered=hoverPanels.includes(a.id);
      const available=(a.id!=='scale'||Boolean((audit?.sourceScaleEligible||getCTStudy?.())&&role==='patient')) &&
        (a.id!=='undo'||viewHistory.status().undo>0) &&
        (a.id!=='redo'||viewHistory.status().redo>0) &&
        (a.id!=='procedure'||role==='reference');
      const settings=station.snapshot();
      const label=a.id==='station-layout'?'LAYOUT '+settings.pose.label:
        a.id==='station-video'?'VIDEO '+settings.capture.label:
        a.id==='station-optics'?'VIEW '+settings.fovDegrees+' DEG':
        a.id==='station-control'?(settings.control==='fine'?'FINE INSERTION':'STANDARD INSERTION'):
        a.id==='scale'&&role==='patient'&&getCTStudy?.()?'CT SECTIONS':
        a.id==='axis'&&orientationLocked?'ORIENTATION LOCKED':
        a.id==='ct-axis'?'PLANE '+ctAxis.toUpperCase():
        a.id==='ct-sections'&&ctSpatial?(ctSpatial.snapshot().visible?'HIDE MPR':'SHOW MPR'):
        a.id==='view'
        ?(mode==='procedure'?'PROCEDURE STATION':
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
    draw(topStrip,'PULMOSIMU / ENDOSCOPIC VIDEO','REFERENCE AIRWAY · '+monitorStatus.toUpperCase());
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
    c.fillText(state.instruction.slice(0,73),42,337);
    c.fillStyle=state.clearance.includes('BLOCKING')?'#e8b99d':'#93adba';
    c.font='23px system-ui';
    c.fillText(state.clearance.slice(0,73),42,378);
    const branches=[
      {id:'guide-branch0',label:state.branches[0]?.label||'NO SOURCE BRANCH',
        disabled:!state.branches[0]?.enabled,u:.04,v:.60,w:.44,h:.14},
      {id:'guide-branch1',label:state.branches[1]?.label||'NO SOURCE BRANCH',
        disabled:!state.branches[1]?.enabled,u:.52,v:.60,w:.44,h:.14}
    ];
    guideHitboxes=[
      ...branches,
      {id:'guide-prev',label:'PREVIOUS',disabled:guideStep===0,u:.04,v:.79,w:.20,h:.11},
      {id:'guide-next',label:'NEXT STEP',disabled:guideStep===4,u:.27,v:.79,w:.20,h:.11},
      {id:'guide-tool',label:'TOOL MODE',disabled:false,u:.50,v:.79,w:.21,h:.11},
      {id:'guide-hide',label:'HIDE',disabled:false,u:.74,v:.79,w:.22,h:.11}
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
      c.fillText(a.label.slice(0,25),x+18,y+height*.65);
    }
    c.fillStyle='#839da7';c.font='19px system-ui';
    c.fillText(state.provenance,42,652);
    // Real current instrument commands, deliberately separate from the
    // endoscopic image. Angles are reference actuator inputs, not measured
    // patient mechanics.
    c.fillStyle='#c7d7dc';c.font='23px system-ui';
    c.fillText('TIP H '+Math.round(scope.deflH*CFG.deflMax)+'°  /  V '+
      Math.round(scope.deflV*CFG.deflMax)+'°   ·   SHAFT '+
      Math.round(scope.roll*180/Math.PI)+'°',42,523);
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
    try{
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
      // THREE r128 WebXRManager binds the XR base-layer framebuffer only at
      // the start of its frame callback. setRenderTarget(null) can replace
      // that binding while this callback is still active. Explicitly restore
      // the real headset framebuffer before the stereo render below.
      if(previousXR&&rBro.xr.isPresenting&&previousTarget===null){
        const layer=rBro.xr.getSession()?.renderState?.baseLayer;
        if(layer?.framebuffer&&rBro.state?.bindXRFramebuffer)
          rBro.state.bindXRFramebuffer(layer.framebuffer);
        else{
          monitorStatus='unavailable';
          monitorError='Unable to restore XR framebuffer after scope pass';
          monitorImage.visible=false;
        }
      }
      camScope.aspect=previousAspect;camScope.updateProjectionMatrix();
      scene.background=originalBg;
      worldMenu.visible=menuWasVisible;
      if(guidePlane)guidePlane.visible=guideWasVisible;
      room.visible=roomWasVisible;
      scopeScreen.visible=monitorWasVisible;
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
      const available=placed&&controllerSources[i]&&!controllerSources[i].hand;
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
    instrumentInputs:{tipH:scope.deflH,tipV:scope.deflV,shaftRollRad:scope.roll},
    videoCapture:{width:scopeTarget.width,height:scopeTarget.height,
      requestedIntervalMs:station.snapshot().capture.intervalMs,
      framesRendered:monitorFrames},
    sessionType:xrPresentation==='ar'?'mixed-reality-passthrough-preview':'immersive-vr',
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
    const desired=presentation==='ar'?'immersive-ar':'immersive-vr';
    try{
      const splash=document.getElementById('splash'),app=document.getElementById('app');
      if(splash)splash.style.display='none';if(app)app.style.display='flex';
      onResize();
      const session=await navigator.xr.requestSession(desired,{
        optionalFeatures:['local-floor','bounded-floor','hand-tracking']
      });
      xrPresentation=presentation;
      try{await rBro.xr.setSession(session)}
      catch(error){try{await session.end()}catch{}throw error}
    }catch(error){
      xrPresentation='vr';
      showToast((presentation==='ar'?'MR passthrough':'Quest XR')+
        ' unavailable: '+String(error?.message||error),'warn');
      console.warn('PulmoSIMU WebXR entry',error);
    }
  }
  button.addEventListener('click',()=>enterXR('vr'));
  arButton.addEventListener('click',()=>enterXR('ar'));
  if(!navigator.xr){
    button.disabled=true;button.textContent='PROCEDURAL XR · QUEST REQUIRED';
  }else{
    navigator.xr.isSessionSupported('immersive-vr').then(ok=>{
      if(!ok){button.disabled=true;button.textContent='PROCEDURAL XR · QUEST REQUIRED'}
    }).catch(()=>{button.disabled=true;button.textContent='PROCEDURAL XR UNAVAILABLE'});
    navigator.xr.isSessionSupported('immersive-ar').then(ok=>{
      arButton.style.display=ok?'block':'none';
    }).catch(()=>{arButton.style.display='none'});
  }
  rBro.xr.addEventListener('sessionstart',()=>{
    // Preserve ordinary opaque 2D/VR rendering and fully restore it at exit.
    backgroundBeforeXR=scene.background;fogBeforeXR=scene.fog;
    clearBeforeXR=rBro.getClearColor(new THREE.Color()).clone();
    clearAlphaBeforeXR=rBro.getClearAlpha?.()??1;
    if(xrPresentation==='ar'){
      scene.background=null;scene.fog=null;
      rBro.setClearColor(0x000000,0);
    }
    health.begin(rBro.xr.getSession()?.frameRate??null);
    qualityStatus=configureComfortRendering();
    reportButton.style.display='none';
    active=true;placed=false;viewerValid=false;lastFrame=0;lastFeed=0;
    lastPanelUpdate=0;scaleMode='comfort';endedXRGeometry=null;
    guideVisible=true;guideStep=0;guideActions=0;
    station.reset();
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
    saved={position:mGrp.position.clone(),quaternion:mGrp.quaternion.clone(),
      scale:mGrp.scale.clone(),extIdle:getExtIdle()};
    setExtIdle(false);vrCam.position.set(0,0,0);vrCam.quaternion.identity();
    worldMenu.visible=true;scopeScreen.visible=false;room.visible=xrPresentation!=='ar';
    if(guidePlane)guidePlane.visible=false;
    room.position.copy(head).setY(-.02);
    button.textContent='EXIT PROCEDURAL XR';
    arButton.textContent='EXIT MR PREVIEW';
    setMode('anatomy');paint();
    rBro.setAnimationLoop((time,frame)=>{
      if(!active||!rBro.xr.isPresenting)return;
      health.startFrame(time);
      if(!headPose(frame)){
        // Tracking loss: never continue the last object or instrument command.
        health.event('viewerPoseLost');health.endFrame({poseAvailable:false});
        viewerValid=false;scope.velocity=0;release(0);release(1);
        handSeen.fill(false);handPinched.fill(false);
        contactMarkers.forEach(marker=>marker.visible=false);
        laserLines.forEach(line=>{if(line)line.visible=false});
        laserTips.forEach(tip=>{if(tip)tip.visible=false});
        return;
      }
      if(!placed){
        room.position.copy(head).setY(head.y-1.18);
        const center=head.clone().addScaledVector(forward,1.5);
        plinth.position.x=center.x-head.x;plinth.position.z=center.z-head.z;
        frameGroup();
      }
      const dt=lastFrame?Math.min(.05,Math.max(0,(time-lastFrame)/1000)):.016;lastFrame=time;
      poll(dt);
      if(analogChangedAt&&!grab.some(Boolean)&&performance.now()-analogChangedAt>380)
        finishViewChange();
      updateTrackedHands(frame);updateSummon();updateGrab();updateSteering();updatePointerAim(time);
      // The identical SI-unit R3 solver used in the research laboratory;
      // explicitly reference-only, never CT-fitted or mesh-deforming.
      if(getMeshMode()==='reference'&&global.PulmoXRReferencePhysics?.step){
        try{physicsSnapshot=global.PulmoXRReferencePhysics.step(dt);physicsError=null}
        catch(error){physicsSnapshot=null;physicsError=String(error?.message||error).slice(0,140)}
      }else physicsSnapshot=null;
      // Advance the ORIGINAL medical procedure engine on XR time, not
      // tab requestAnimationFrame, which can be throttled in a headset.
      if(mode==='procedure'&&Math.abs(scope.velocity)>.0001){
        const step=Math.min(1.5,Math.max(.1,dt*60));
        const distance=scope.velocity*step;
        if(distance>0)advance(distance);else retract(-distance);
        scope.velocity*=Math.pow(scope.velocityDecay,step);
      }
      updateScope();
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
            station.next('video'); // detail -> balanced -> economy
            const video=station.snapshot().capture;
            scopeTarget.setSize(video.width,video.height);
            lastFeed=0;
            grabFeedback='VIDEO QUALITY ADAPTED / HIGH MEASURED CPU COST';
            paint();
          }
        }
      }
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
      instrumentInputs:{tipH:scope.deflH,tipV:scope.deflV,shaftRollRad:scope.roll},
      controllerActions:{...attempts},
      viewHistory:viewHistory.status(),pivotMode,axisMode,
      orientationLocked,spatialCT:ctSpatial?.snapshot()||null
    };
    health.end();reportButton.style.display='block';
    active=false;rBro.setAnimationLoop(null);scope.velocity=0;
    camScope.fov=originalScopeFov;camScope.updateProjectionMatrix();
    if(ctSpatial){ctSpatial.dispose();ctSpatial=null;}
    worldMenu.visible=false;scopeScreen.visible=false;room.visible=false;
    if(guidePlane)guidePlane.visible=false;
    // Restore the exact pre-session desktop scene and renderer blend state.
    scene.background=backgroundBeforeXR;scene.fog=fogBeforeXR;
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
    procedureStation:station.snapshot(),
    instrumentInputs:{tipH:scope.deflH,tipV:scope.deflV,shaftRollRad:scope.roll},
    contact:lastContact,grabFeedback,grabAttempts:{...attempts},
    activeControllerSides:controllerSide.slice(),
    menuHoverActions:hoverPanels.slice(),
    spatialMenuVisible:Boolean(worldMenu.visible),
    roomShellVisible:Boolean(room.visible),
    stagePresentation:stage.snapshot(),xrFoveation:qualityStatus,
    xrPresentation,procedureGuideVisible:Boolean(guidePlane?.visible),
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
