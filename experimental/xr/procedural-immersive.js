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
    getPatientAudit,getMmPerSceneUnit,getScopeDiameterMm,getReferenceFocus,
    setClinMode,getClinMode,fireAction,enterBranch,advance,retract,hideBranchSel,
    updateScope,onResize,showToast,getExtIdle,setExtIdle}=ctx;
  if(!THREE||!scene||!rBro||!mGrp||!scope)throw Error('Procedural XR bridge missing original clinical engine');
  if(!global.PulmoXRSessionHealth?.createSessionHealth)
    throw Error('XR-09 local session health runtime missing');
  if(!global.PulmoXRRemoteTarget?.pick)
    throw Error('XR-10 triangle-backed remote aiming runtime missing');
  const health=global.PulmoXRSessionHealth.createSessionHealth();
  rBro.xr.enabled=true;
  if(rBro.xr.setFramebufferScaleFactor)rBro.xr.setFramebufferScaleFactor(.78);

  const safe=(a,b,v)=>Math.max(a,Math.min(b,v));
  const vec=(a)=>new THREE.Vector3(a[0],a[1],a[2]);
  const hands=[null,null],grips=[null,null],grab=[null,null],controllerSide=[null,null];
  // WebXR getController(index) is connection-order, NOT reliably left=0/right=1.
  const controllerSources=[null,null],laserLines=[null,null],laserTips=[null,null];
  const aimTargets=[null,null],squeezeHeld=[false,false];
  const attempts={acceptedRay:0,acceptedNear:0,rejected:0,trigger:0,grip:0};
  let lastAimTime=0,grabFeedback='AIM AT LUNG / HOLD TRIGGER OR GRIP';
  let summonTween=null;
  const handProxies=[new THREE.Group(),new THREE.Group()];
  const handSeen=[false,false],handPinched=[false,false],grabSource=[null,null];
  const contactMarkers=[];
  let mode='anatomy',active=false,placed=false,viewerValid=false,lastFrame=0;
  let scaleMode='comfort',lastPanelUpdate=0,physicsSnapshot=null;
  let physicsError=null;
  let saved=null,scopeSteering=null;
  let gesture=null,worldMenu=null,scopeScreen=null,scopeTexture=null,scopeCanvas=null;
  let button=null,lastFeed=0,actionHitbox=[],summonGeneration=0;
  const tempMatrix=new THREE.Matrix4(),ray=new THREE.Raycaster();
  const modelBox=new THREE.Box3(),head=new THREE.Vector3(),forward=new THREE.Vector3(),right=new THREE.Vector3();
  const tmp=new THREE.Vector3(),tmp2=new THREE.Vector3(),tempBox=new THREE.Box3();
  const surfaceTarget=()=>getAnatomyMeshes().filter(o=>o&&o.visible&&o.isObject3D);
  const trackedController=i=>controllerSources[i]?.gripSpace?grips[i]:hands[i];
  const effectiveGrip=i=>grabSource[i]==='hand'?handProxies[i]:trackedController(i);
  const release=(i,source=null,input=null)=>{
    if(source&&grabSource[i]&&grabSource[i]!==source)return;
    if(input&&grab[i]&&grab[i].input!==input)return;
    const endedGrab=grab[i];
    grab[i]=null;grabSource[i]=null;
    if(endedGrab){
      grabFeedback='RELEASED '+(endedGrab.method||'ANATOMY');
      // A short trigger TAP on the actual surface summons the lung
      // within reach, while a HOLD translates/rotates it at a distance.
      if(source==='controller'&&input==='trigger'&&mode==='anatomy'&&
        performance.now()-endedGrab.startedAt<400&&
        mGrp.position.distanceTo(endedGrab.startPosition)<.025 &&
        mGrp.quaternion.angleTo(endedGrab.startQuaternion)<.045)
        summon();
    }
    if(scopeSteering?.i===i&&(!source||source==='controller'))scopeSteering=null;
    if(grab[0]&&!grab[1])capture(0);
    if(grab[1]&&!grab[0])capture(1);
    gesture=null;
  };
  function setMode(next){
    if(!['anatomy','procedure'].includes(next))return;
    summonGeneration++;
    mode=next;grab[0]=grab[1]=null;grabSource.fill(null);
    gesture=null;scopeSteering=null;summonTween=null;
    grabFeedback='AIM AT LUNG / HOLD TRIGGER OR GRIP';
    if(scopeScreen)scopeScreen.visible=next==='procedure';
    if(next==='anatomy')scope.velocity=0;
    paint();
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
  function frameGroup(distance=1.55,width=.62){
    if(!viewerValid)return false;
    summonGeneration++;
    release(0);release(1);
    const bounds=sourceBounds();
    const scale=safe(.1,1.2,Math.min(width/bounds.size.x,.65/bounds.size.y));
    const center=head.clone().addScaledVector(forward,distance).add(new THREE.Vector3(0,-.1,0));
    mGrp.quaternion.identity();
    mGrp.scale.setScalar(scale);scaleMode='comfort';
    mGrp.position.copy(center).sub(bounds.center.clone().multiplyScalar(scale));
    mGrp.updateMatrixWorld(true);placed=true;
    positionPanels();
    return true;
  }
  function setSourceScale(){
    const audit=getPatientAudit?.();
    if(!audit||getMeshMode()!=='patient'||audit.status!=='coframed'||
       !audit.sourceScaleEligible||mode!=='anatomy'||grab.some(Boolean))
      return false;
    const sourceScale=audit.sceneUnitMetres;
    if(!Number.isFinite(sourceScale)||sourceScale<.005||sourceScale>2)return false;
    const local=sourceBounds().center;
    const currentCenter=local.clone().multiplyScalar(mGrp.scale.x)
      .applyQuaternion(mGrp.quaternion).add(mGrp.position);
    mGrp.scale.setScalar(sourceScale);
    mGrp.position.copy(currentCenter)
      .sub(local.clone().multiplyScalar(sourceScale).applyQuaternion(mGrp.quaternion));
    scaleMode='source-derived';
    mGrp.updateMatrixWorld(true);
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
      worldMenu.position.copy(head).addScaledVector(forward,1.18).addScaledVector(right,-.65);
      worldMenu.position.y-=.04;
      worldMenu.lookAt(head);
    }
    if(scopeScreen){
      scopeScreen.position.copy(head).addScaledVector(forward,1.34).addScaledVector(right,.74);
      scopeScreen.position.y-=.1;scopeScreen.lookAt(head);
    }
  }
  let lastContact=null;
  function surfaceContact(i,source='controller'){
    const controller=source==='hand'?handProxies[i]:grips[i];
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
  function startGrab(i,source='controller'){
    if(source==='controller'&&mode==='procedure'&&active&&controllerSide[i]==='right'){
      const g=grips[i];if(!g)return false;
      g.updateMatrixWorld(true);
      scopeSteering={i,origin:g.getWorldQuaternion(new THREE.Quaternion()),
        deflH:scope.deflH,deflV:scope.deflV};
      return true;
    }
    if(!active||!placed||mode!=='anatomy'||grab[i]||!surfaceContact(i,source).hit)return false;
    const grip=source==='hand'?handProxies[i]:grips[i];if(!grip)return false;
    summonGeneration++;
    grabSource[i]=source;
    grab[i]={relative:new THREE.Matrix4()};capture(i);
    if(grab[0]&&grab[1]){
      effectiveGrip(0).updateMatrixWorld(true);effectiveGrip(1).updateMatrixWorld(true);
      const a=effectiveGrip(0).getWorldPosition(new THREE.Vector3());
      const b=effectiveGrip(1).getWorldPosition(new THREE.Vector3());
      gesture={a:a.clone(),b:b.clone(),center:a.clone().add(b).multiplyScalar(.5),
        position:mGrp.position.clone(),quat:mGrp.quaternion.clone(),scale:mGrp.scale.x};
    }else gesture=null;
    return true;
  }
  function updateGrab(){
    if(!grab[0]&&!grab[1]||mode!=='anatomy')return;
    mGrp.parent?.updateMatrixWorld(true);
    if(grab[0]&&grab[1]&&gesture){
      effectiveGrip(0).updateMatrixWorld(true);effectiveGrip(1).updateMatrixWorld(true);
      const a=effectiveGrip(0).getWorldPosition(new THREE.Vector3()),b=effectiveGrip(1).getWorldPosition(new THREE.Vector3());
      const initial=gesture.b.clone().sub(gesture.a),current=b.clone().sub(a);
      if(initial.length()<.07||current.length()<.07)return;
      const rotation=new THREE.Quaternion().setFromUnitVectors(initial.normalize(),current.normalize());
      const scaling=safe(.18,1.65,gesture.scale*current.length()/gesture.b.distanceTo(gesture.a));
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
    // Keep the assembly from passing into the user's head.
    const distance=mGrp.position.distanceTo(head);
    if(distance<.38){
      const offset=mGrp.position.clone().sub(head);
      if(offset.length()<.001)offset.copy(forward);
      mGrp.position.copy(head).add(offset.normalize().multiplyScalar(.38));
    }
  }
  function updateTrackedHands(frame){
    const session=rBro.xr.getSession(),space=rBro.xr.getReferenceSpace();
    const found=[false,false],valid=[false,false];
    if(session&&space&&frame?.getJointPose){
      for(const source of session.inputSources){
        if(!source.hand)continue;
        const i=source.handedness==='left'?0:source.handedness==='right'?1:-1;
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
    grips[i].updateMatrixWorld(true);
    const orientation=grips[i].getWorldQuaternion(new THREE.Quaternion());
    // Change in controller orientation relative to the moment of grasp.
    // This is a VR input mapping to the ORIGINAL scope deflection state,
    // not an assertion of mechanical shaft/knob or force-feedback realism.
    const delta=origin.clone().invert().multiply(orientation);
    const euler=new THREE.Euler().setFromQuaternion(delta,'YXZ');
    scope.deflH=safe(-1,1,deflH+euler.y/0.65);
    scope.deflV=safe(-1,1,deflV-euler.x/0.65);
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
  function findPanelHit(r){
    if(!worldMenu?.visible)return null;
    const hit=r.intersectObject(worldMenu,false)[0];if(!hit?.uv)return null;
    const x=hit.uv.x,y=1-hit.uv.y;
    return actionHitbox.find(a=>x>=a.u&&x<=a.u+a.w&&y>=a.v&&y<=a.v+a.h)?.id||null;
  }
  function summon(){
    if(!viewerValid||mode!=='anatomy')return;
    const bounds=sourceBounds(),targetScale=Math.min(mGrp.scale.x,.4/bounds.size.x,.45/bounds.size.y);
    const targetCenter=head.clone().addScaledVector(forward,.82).add(new THREE.Vector3(0,-.07,0));
    const targetPosition=targetCenter.clone().sub(bounds.center.clone().multiplyScalar(targetScale).applyQuaternion(mGrp.quaternion));
    const start=mGrp.position.clone(),old=mGrp.scale.x;
    // No camera motion. Smoothly approach a reachable workbench.
    gesture=null;
    const begin=performance.now(),generation=++summonGeneration;
    const update=()=>{
      if(generation!==summonGeneration||!active||mode!=='anatomy'||grab.some(Boolean))return;
      const t=safe(0,1,(performance.now()-begin)/480);
      const eased=1-Math.pow(1-t,3);
      mGrp.position.copy(start).lerp(targetPosition,eased);
      mGrp.scale.setScalar(old+(targetScale-old)*eased);
      if(t<1)requestAnimationFrame(update);
    };
    update();
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
    const action=findPanelHit(ray);
    const side=controllerSide[i]|| (i===0?'left':'right');
    if(action){activate(action);return}
    if(mode==='anatomy')pickFromScene(ray);
    else if(side==='right'){
      if(scope.atBranch){enterBranch(Math.min(1,(SEG_MAP[scope.segId]?.children.length||1)-1))}
      else fireAction();
    }else if(scope.atBranch)enterBranch(0);
  }
  const modes=['nav','biopsy','measure','annotate'];
  function activate(action){
    if(action==='anatomy')setMode('anatomy');
    if(action==='procedure')setMode('procedure');
    if(action==='reset')frameGroup();
    if(action==='scale'){
      if(!setSourceScale())showToast('Source scale requires co-framed CT metadata (not synthetic)','warn');
    }
    if(action==='tool')setClinMode(modes[(modes.indexOf(getClinMode())+1)%modes.length]);
    if(action==='exit')rBro.xr.getSession()?.end().catch(()=>{});
    paint();
  }
  const mapButtons=[[],[]];
  function poll(dt){
    const session=rBro.xr.getSession();if(!session)return;
    let leftControllerAvailable=false;
    for(const src of session.inputSources){
      if(!src.gamepad)continue;
      const i=src.handedness==='left'?0:src.handedness==='right'?1:-1;
      if(i<0)continue;
      if(i===0)leftControllerAvailable=true;
      const buttons=src.gamepad.buttons||[];
      const edge=idx=>{const pressed=!!buttons[idx]?.pressed,prior=!!mapButtons[i][idx];mapButtons[i][idx]=pressed;return pressed&&!prior};
      const axes=Array.from(src.gamepad.axes||[]);
      const n=axes.length>=4?2:0;
      const x=Math.abs(axes[n]||0)<.13?0:axes[n],y=Math.abs(axes[n+1]||0)<.13?0:axes[n+1];
      if(i===1&&edge(4))setMode(mode==='anatomy'?'procedure':'anatomy');
      if(i===0&&edge(4))frameGroup();
      if(mode!=='procedure')continue;
      if(i===0){
        if(y)scope.velocity=-y*CFG.scopeSpd/900;
        if(x)scope.roll+=x*.03;
      }else{
        if(!scopeSteering){
          scope.deflH=safe(-1,1,scope.deflH*.82+x*.18);
          scope.deflV=safe(-1,1,scope.deflV*.82-y*.18);
        }
      }
      if(i===1&&edge(5))activate('tool');
    }
    // A vanished controller must not leave the original scope advancing.
    if(mode==='procedure'&&!leftControllerAvailable)scope.velocity=0;
  }
  // World-space context for a seated user. This is environmental staging,
  // deliberately independent of anatomy/scopes and ignored by raycasting.
  const room=new THREE.Group();room.visible=false;scene.add(room);
  const floor=new THREE.Mesh(new THREE.CircleGeometry(3.8,64),
    new THREE.MeshStandardMaterial({color:0x14242e,roughness:1,metalness:0}));
  floor.rotation.x=-Math.PI/2;room.add(floor);
  const grid=new THREE.GridHelper(7.2,20,0x4e6671,0x263843);
  grid.material.transparent=true;grid.material.opacity=.24;
  grid.position.y=.003;room.add(grid);
  const plinth=new THREE.Mesh(new THREE.CylinderGeometry(.72,.78,.055,64),
    new THREE.MeshStandardMaterial({color:0x253742,metalness:.16,roughness:.86,
      transparent:true,opacity:.45}));
  plinth.position.set(0,.03,0);room.add(plinth);
  // Plinth only indicates working region, never anatomical scale or a patient metric.
  const panelCanvas=document.createElement('canvas');panelCanvas.width=1024;panelCanvas.height=600;
  const panelCtx=panelCanvas.getContext('2d');
  const panelTex=new THREE.CanvasTexture(panelCanvas);
  panelTex.encoding=THREE.sRGBEncoding;
  worldMenu=new THREE.Mesh(new THREE.PlaneGeometry(.7,.41),
    new THREE.MeshBasicMaterial({map:panelTex,transparent:true,side:THREE.DoubleSide,depthWrite:false,depthTest:false}));
  worldMenu.renderOrder=40;worldMenu.visible=false;scene.add(worldMenu);
  actionHitbox=[
    {id:'anatomy',label:'ANATOMY',u:.05,v:.54,w:.43,h:.12},
    {id:'procedure',label:'BRONCHOSCOPY',u:.52,v:.54,w:.43,h:.12},
    {id:'tool',label:'INSTRUMENT',u:.05,v:.69,w:.43,h:.12},
    {id:'reset',label:'RECENTER',u:.52,v:.69,w:.43,h:.12},
    {id:'scale',label:'SOURCE SCALE',u:.05,v:.84,w:.43,h:.12},
    {id:'exit',label:'EXIT VR',u:.52,v:.84,w:.43,h:.12}
  ];
  function paint(){
    const c=panelCtx,w=1024,h=600;
    const audit=getPatientAudit?.();
    const role=getMeshMode();
    const referenceFocus=role==='reference'?(getReferenceFocus?.()||null):null;
    const source=role==='patient'&&audit?.status==='coframed'
      ?(audit.source==='synthetic'?'SYNTHETIC · COFRAMED ONLY':'CT-DERIVED · COFRAMED / UNVERIFIED')
      :'PROCEDURAL REFERENCE · UNCALIBRATED';
    const clearance=mode==='procedure'?clearanceState():null;
    c.clearRect(0,0,w,h);c.fillStyle='#14222c';c.fillRect(0,0,w,h);
    c.fillStyle='#7ea5b3';c.fillRect(0,0,w,8);
    c.textAlign='left';c.fillStyle='#afc9d1';c.font='600 27px system-ui';
    c.fillText('PULMOSIMU  /  XR-09 UNIFIED ENGINE',48,64);
    c.fillStyle='#e8f0f2';c.font='600 48px system-ui';
    c.fillText(mode==='procedure'?'Bronchoscopy':'Anatomical exploration',48,131);
    c.fillStyle='#d3e0e5';c.font='28px system-ui';
    c.fillText((mode==='procedure'
      ?('SEG '+scope.segId+' · '+(SEG_MAP[scope.segId]?.name||'—'))
      :'Original anatomical scene · '+scaleMode+' scale'+(referenceFocus?' · '+referenceFocus:'' )).slice(0,51),48,189);
    c.fillStyle='#a8c4cc';c.font='23px system-ui';
    c.fillText(mode==='procedure'
       ?'LEFT: ADVANCE · RIGHT GRIP: STEER · TRIGGER: ACT'
       :'TRIGGER: SUMMON · NEAR GRIP: HOLD · BOTH: RESIZE',48,236);
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
      const available=a.id!=='scale'||Boolean(audit?.sourceScaleEligible&&role==='patient');
      c.fillStyle=selected?'#345463':available?'#213641':'#1c2a32';
      c.fillRect(x,y,width,height);
      c.strokeStyle=selected?'#adcdd5':available?'#4b6673':'#314652';
      c.lineWidth=2;c.strokeRect(x,y,width,height);
      c.fillStyle=available?'#e5eef1':'#6f8791';
      c.font='600 27px system-ui';
      c.fillText(a.label,x+23,y+height*.66);
    }
    panelTex.needsUpdate=true;
  }
  // Live bronchoscopy is rendered from the ORIGINAL camScope directly
  // into a render target owned by the SAME VR WebGL context. No canvas
  // readback, second-context upload, video recording or synthetic camera.
  const scopeTarget=new THREE.WebGLRenderTarget(640,360,{
    minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,
    format:THREE.RGBAFormat,depthBuffer:true,stencilBuffer:false
  });
  scopeTexture=scopeTarget.texture;scopeTexture.encoding=THREE.sRGBEncoding;
  let monitorStatus='not-started',monitorError=null,monitorFrames=0;
  const monitorRoot=new THREE.Group();
  const monitorFrame=new THREE.Mesh(new THREE.PlaneGeometry(.78,.49),
    new THREE.MeshBasicMaterial({color:0x0b161e,side:THREE.DoubleSide}));
  monitorFrame.position.z=-.012;monitorRoot.add(monitorFrame);
  const monitorImage=new THREE.Mesh(new THREE.PlaneGeometry(.73,.411),
    new THREE.MeshBasicMaterial({map:scopeTexture,side:THREE.DoubleSide,depthTest:false}));
  monitorImage.visible=false;monitorImage.renderOrder=30;monitorRoot.add(monitorImage);
  scopeScreen=monitorRoot;scopeScreen.visible=false;scene.add(scopeScreen);
  function captureFeed(now){
    if(mode!=='procedure'||now-lastFeed<90||monitorStatus==='unavailable')return;
    lastFeed=now;
    const previousTarget=rBro.getRenderTarget();
    const previousXR=rBro.xr.enabled;
    const previousAspect=camScope.aspect;
    const menuWasVisible=worldMenu.visible,roomWasVisible=room.visible;
    const monitorWasVisible=scopeScreen.visible;
    try{
      scopeScreen.visible=false;worldMenu.visible=false;room.visible=false;
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
      worldMenu.visible=menuWasVisible;
      room.visible=roomWasVisible;
      scopeScreen.visible=monitorWasVisible;
    }
  }
  for(let i=0;i<2;i++){
    scene.add(handProxies[i]);
    const marker=new THREE.Mesh(new THREE.SphereGeometry(.009,8,6),
      new THREE.MeshBasicMaterial({color:0xc0dce0,transparent:true,opacity:.76,depthTest:false}));
    marker.visible=false;marker.renderOrder=36;scene.add(marker);contactMarkers.push(marker);
  }
  for(let i=0;i<2;i++){
    const controller=rBro.xr.getController(i),grip=rBro.xr.getControllerGrip(i);
    hands[i]=controller;grips[i]=grip;scene.add(controller,grip);
    const line=new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3(0,0,-1)]),
      new THREE.LineBasicMaterial({color:i?0xb6d1d9:0x789ba9,transparent:true,opacity:.45}));
    line.scale.z=1.9;controller.add(line);
    controller.addEventListener('connected',event=>{
      controllerSide[i]=event.data?.handedness||null;
      mapButtons[i].length=0;health.event('controllerConnected');
    });
    controller.addEventListener('selectstart',event=>{
      // A tracked-hand selectstart is the same physical pinch handled by
      // XRJointPose. Never treat it as an unrelated distance-ray click.
      if(!event.data?.hand)onTrigger(i);
    });
    controller.addEventListener('squeezestart',()=>startGrab(i,'controller'));
    controller.addEventListener('squeezeend',()=>release(i,'controller'));
    controller.addEventListener('disconnected',()=>{
      health.event('controllerDisconnected');
      release(i,'controller');controllerSide[i]=null;mapButtons[i].length=0;
      if(i===0)scope.velocity=0;
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
  const localSessionReport=()=>({
    session:health.snapshot(),
    endoscopeMonitor:{status:monitorStatus,framesRendered:monitorFrames},
    integration:'XR-09 original procedural scene; unvalidated research build'
  });
  reportButton.addEventListener('click',()=>{
    const blob=new Blob([JSON.stringify(localSessionReport(),null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download='pulmosimu-xr09-session-report.json';
    document.body.appendChild(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(url),2000);
  });
  global.pulmoXRSessionReport=localSessionReport;
  button.addEventListener('click',async()=>{
    if(rBro.xr.isPresenting){await rBro.xr.getSession()?.end();return}
    try{
      // The splash screen blocks all other UI. For WebXR, open the existing
      // model without requesting webcam/MediaPipe desktop tracking.
      const splash=document.getElementById('splash'),app=document.getElementById('app');
      if(splash)splash.style.display='none';if(app)app.style.display='flex';
      onResize();
      const session=await navigator.xr.requestSession('immersive-vr',{
        optionalFeatures:['local-floor','bounded-floor','hand-tracking']
      });
      await rBro.xr.setSession(session);
    }catch(e){showToast('Quest XR unavailable: '+e.message,'warn');console.warn('Procedural XR',e)}
  });
  if(!navigator.xr){
    button.disabled=true;button.textContent='PROCEDURAL XR · QUEST REQUIRED';
  }else{
    navigator.xr.isSessionSupported('immersive-vr').then(ok=>{
      if(!ok){button.disabled=true;button.textContent='PROCEDURAL XR · QUEST REQUIRED'}
    }).catch(()=>{button.disabled=true;button.textContent='PROCEDURAL XR UNAVAILABLE'});
  }
  rBro.xr.addEventListener('sessionstart',()=>{
    health.begin(rBro.xr.getSession()?.frameRate??null);
    reportButton.style.display='none';
    active=true;placed=false;viewerValid=false;lastFrame=0;lastFeed=0;
    lastPanelUpdate=0;scaleMode='comfort';
    physicsSnapshot=null;physicsError=null;
    global.PulmoXRReferencePhysics?.reset?.();
    handPinched.fill(false);handSeen.fill(false);grabSource.fill(null);
    contactMarkers.forEach(m=>m.visible=false);
    monitorStatus='not-started';monitorError=null;monitorFrames=0;
    monitorImage.visible=false;
    saved={position:mGrp.position.clone(),quaternion:mGrp.quaternion.clone(),
      scale:mGrp.scale.clone(),extIdle:getExtIdle()};
    setExtIdle(false);vrCam.position.set(0,0,0);vrCam.quaternion.identity();
    worldMenu.visible=true;scopeScreen.visible=false;room.visible=true;
    room.position.copy(head).setY(-.02);
    button.textContent='EXIT PROCEDURAL XR';setMode('anatomy');paint();
    rBro.setAnimationLoop((time,frame)=>{
      if(!active||!rBro.xr.isPresenting)return;
      health.startFrame(time);
      if(!headPose(frame)){
        // Tracking loss: never continue the last object or instrument command.
        health.event('viewerPoseLost');health.endFrame({poseAvailable:false});
        viewerValid=false;scope.velocity=0;release(0);release(1);
        handSeen.fill(false);handPinched.fill(false);
        contactMarkers.forEach(marker=>marker.visible=false);
        return;
      }
      if(!placed){
        room.position.copy(head).setY(head.y-1.18);
        const center=head.clone().addScaledVector(forward,1.5);
        plinth.position.x=center.x-head.x;plinth.position.z=center.z-head.z;
        frameGroup();
      }
      const dt=lastFrame?Math.min(.05,Math.max(0,(time-lastFrame)/1000)):.016;lastFrame=time;
      poll(dt);updateTrackedHands(frame);updateGrab();updateSteering();
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
        const monitorStarted=performance.now(),beforeStatus=monitorStatus;
        captureFeed(time);monitorCpuMs=performance.now()-monitorStarted;
        if(beforeStatus!=='unavailable'&&monitorStatus==='unavailable')
          health.event('monitorUnavailable');
      }
      // Stereoscopic source render. CPU call time is NOT a GPU timing.
      const renderStarted=performance.now();
      try{rBro.render(scene,vrCam)}
      finally{health.endFrame({renderMs:performance.now()-renderStarted,
        monitorMs:monitorCpuMs})}
    });
  });
  rBro.xr.addEventListener('sessionend',()=>{
    health.end();reportButton.style.display='block';
    active=false;rBro.setAnimationLoop(null);scope.velocity=0;
    worldMenu.visible=false;scopeScreen.visible=false;room.visible=false;
    if(saved){
      mGrp.position.copy(saved.position);mGrp.quaternion.copy(saved.quaternion);
      mGrp.scale.copy(saved.scale);setExtIdle(saved.extIdle);
    }
    for(let i=0;i<2;i++)grab[i]=null;
    grabSource.fill(null);handSeen.fill(false);handPinched.fill(false);
    contactMarkers.forEach(m=>m.visible=false);
    summonGeneration++;gesture=null;scopeSteering=null;placed=false;viewerValid=false;
    button.textContent='ENTER PROCEDURAL XR';onResize();
  });
  paint();
  const diagnostics=()=>({source:'original-pneumovr-scene',modelRole:getMeshMode(),
    mode,active,placed,viewerValid,monitor:scopeScreen.visible,
    originalSegments:SEGS.length,scopeBranch:scope.segId,clinicalMode:getClinMode(),
    scopeSteeringActive:Boolean(scopeSteering),
    monitorSource:'same-renderer-offscreen',monitorStatus,monitorFrames,monitorError,
    contact:lastContact,
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
