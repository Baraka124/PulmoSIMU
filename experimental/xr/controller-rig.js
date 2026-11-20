/**
 * PulmoSIMU XR-02 — tracked-controller anatomical manipulation.
 * Interacts exclusively with the experimental reference renderer. Does not
 * mutate the PNEUMO·VR procedure, camera, biopsy, or clinical physiology.
 */
import * as THREE from 'three';
import {XRControllerModelFactory} from 'three/addons/webxr/XRControllerModelFactory.js';
import {bimanualPose,clampDisplayScale,dominantThumbstick} from './spatial-interaction.js';
import {createWelcomeSequence} from './experience-sequence.js';
import {fitSeatedAnatomy,COMFORT_DEFAULTS} from './comfort-framing.js';
import {SPATIAL_LIMITS,decideInteraction,pinchActive,workZoneTarget,workZoneScale,easedTransform,frameMetrics} from './spatial-intent.js';

export function createSpatialRig({
  renderer,scene,camera,anatomy,onSelect,onHover,onTouch,onMode,onReset,onIsolate,onProcedure,onRouteStep,onSectionStep,onNextBranch,getInfo
}){
  const controllers=[],grips=[],grabbing=[false,false],handAnchors=[null,null];
  const grabbingSource=[null,null],pinching=[false,false],handSeen=[false,false],handProxies=[],handMarkers=[];
  let transition=null,metricsSamples=[],metrics={fps:null,p95Ms:null,slowFrames:0};
  let twoHandStart=null,started=false,menuVisible=true,mode='anatomy',previousButtons=[[],[]];
  const welcome=createWelcomeSequence();
  let menuVersion='',hoverAction=null;
  const raycaster=new THREE.Raycaster(),basis=new THREE.Matrix4();
  const headPosition=new THREE.Vector3(),headForward=new THREE.Vector3(),headRight=new THREE.Vector3(),q=new THREE.Quaternion();
  const tmpVec=new THREE.Vector3(),tmpMatrix=new THREE.Matrix4(),tmpBox=new THREE.Box3();
  const scaleLimits={minScale:COMFORT_DEFAULTS.minUserScale,maxScale:COMFORT_DEFAULTS.maxUserScale};
  const activeGrip=i=>grabbingSource[i]==='hand'?handProxies[i]:grips[i];
  function surfaceDistance(position){
    anatomy.updateMatrixWorld(true);
    let min=Infinity;
    anatomy.traverse(obj=>{
      if(!obj.isMesh||!obj.userData.regionId||!obj.visible||!obj.geometry)return;
      tmpBox.setFromObject(obj);
      min=Math.min(min,tmpBox.distanceToPoint(position));
    });
    return min;
  }
  let validViewerPose=false,firstFrameBounds=null,firstFrame=null;
  const controllerFactory=new XRControllerModelFactory();

  const panelCanvas=document.createElement('canvas');
  panelCanvas.width=1024;panelCanvas.height=970;
  const ctx=panelCanvas.getContext('2d');
  const tex=new THREE.CanvasTexture(panelCanvas);tex.colorSpace=THREE.SRGBColorSpace;
  const panel=new THREE.Mesh(new THREE.PlaneGeometry(.69,.655),
    new THREE.MeshBasicMaterial({map:tex,transparent:true,side:THREE.DoubleSide,depthWrite:false,depthTest:false}));
  panel.renderOrder=45;panel.visible=false;scene.add(panel);

  const actions=[
    {id:'anatomy',label:'EXPLORE',x:65,y:418,w:434,h:89},
    {id:'path',label:'AIRWAY GUIDE',x:526,y:418,w:434,h:89},
    {id:'section',label:'SECTION VIEW',x:65,y:521,w:434,h:89},
    {id:'next',label:'NEXT BRANCH',x:526,y:521,w:434,h:89},
    {id:'reset',label:'RECENTER',x:65,y:624,w:434,h:89},
    {id:'isolate',label:'ISOLATE LOBE',x:526,y:624,w:434,h:89},
    {id:'procedure',label:'PROCEDURES',x:65,y:727,w:434,h:89},
    {id:'help',label:'FIRST STEPS',x:526,y:727,w:434,h:89}
  ];
  const tutorialActions=[
    {id:'tour-next',label:'NEXT STEP',x:65,y:710,w:434,h:108},
    {id:'tour-skip',label:'EXPLORE NOW',x:526,y:710,w:434,h:108}
  ];
  const activeActions=()=>welcome.get().active?tutorialActions:actions;
  function textWrap(text,x,y,maxWidth,lineHeight,maxLines=5){
    const words=String(text).split(/\s+/);let line='',row=0;
    for(const word of words){
      const candidate=line?line+' '+word:word;
      if(ctx.measureText(candidate).width>maxWidth&&line){
        ctx.fillText(line,x,y+row*lineHeight);row++;line=word;
        if(row>=maxLines)return;
      }else line=candidate;
    }
    if(line&&row<maxLines)ctx.fillText(line,x,y+row*lineHeight);
  }

  function rect(x,y,w,h,fill,stroke,r=12){
    ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();
    if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke()}
  }
  function drawPanel(force=false){
    const info=getInfo?.()||{};
    const tour=welcome.get();
    const key=JSON.stringify([mode,menuVisible,info.region,info.name,info.isolated,info.progress,info.branch,info.generation,info.section,hoverAction,tour.active,tour.index]);
    if(!force&&key===menuVersion)return;
    menuVersion=key;ctx.clearRect(0,0,1024,970);
    rect(0,0,1024,970,'rgba(18,31,39,.96)','#526974',16);
    rect(0,0,1024,9,'#9cbec8',null,0);
    if(tour.active){
      ctx.textAlign='left';ctx.fillStyle='#b8d3db';ctx.font='600 29px system-ui';
      ctx.fillText('PULMOSIMU XR-06  /  FIRST SESSION',65,91);
      ctx.fillStyle='#ebf1f4';ctx.font='600 53px system-ui';
      textWrap(tour.title,65,190,860,64,2);
      ctx.fillStyle='#adc7d0';ctx.font='500 29px system-ui';
      ctx.fillText('STEP '+(tour.index+1)+' OF '+tour.total,65,284);
      rect(65,335,894,280,'rgba(45,67,78,.52)',null,14);
      ctx.fillStyle='#f0f4f5';ctx.font='35px system-ui';
      textWrap(tour.text,93,405,825,51,4);
      ctx.fillStyle='#afcbd4';ctx.font='600 26px system-ui';
      textWrap(tour.hint,65,667,885,38,2);
      for(const a of tutorialActions){
        const hovered=hoverAction===a.id;
        rect(a.x,a.y,a.w,a.h,hovered?'#43626f':'#2b434e',hovered?'#bbdce3':'#647f8a',12);
        ctx.fillStyle='#f0f6f8';ctx.font='600 34px system-ui';ctx.fillText(a.label,a.x+28,a.y+66);
      }
      ctx.fillStyle='#9bb7c2';ctx.font='28px system-ui';
      ctx.fillText('POINT + TRIGGER TO CONTINUE   ·   A: HIDE / SHOW PANEL',65,895);
      ctx.fillStyle='#8ca8b3';ctx.font='24px system-ui';
      ctx.fillText('SCHEMATIC REFERENCE · NOT CLINICAL / CALIBRATED',65,939);
      tex.needsUpdate=true;return;
    }
    ctx.textAlign='left';ctx.fillStyle='#a8c2cb';ctx.font='500 31px system-ui';
    ctx.fillText('PULMOSIMU XR-06  /  ANATOMY WORKBENCH',65,88);
    ctx.fillStyle='#ecf2f4';ctx.font='600 62px system-ui';
    ctx.fillText((info.name||'Reference anatomy').slice(0,26),65,190);
    ctx.fillStyle='#c1d3d9';ctx.font='34px system-ui';
    ctx.fillText('Region '+(info.region||'—')+'     ·     '+(mode==='path'?'AIRWAY GUIDE':mode==='section'?'SECTION VIEW':'ANATOMY'),65,262);
    rect(65,305,894,94,'rgba(42,66,77,.42)',null,10);
    ctx.fillStyle='#cfdee3';ctx.font='26px system-ui';
    ctx.fillText(('BRANCH '+(info.branch||'—')+'   ·   '+(info.generation==null?'REFERENCE':'G'+info.generation)).slice(0,56),88,341);
    ctx.fillText(mode==='section'?'LEFT THUMBSTICK: SWEEP REFERENCE SECTION':
      mode==='path'?'LEFT THUMBSTICK: FOLLOW REFERENCE AIRWAY':
      'TRIGGER: SUMMON    ·    NEAR GRIP: HOLD',88,378);
    for(const a of activeActions()){
      const active=a.id===mode || (a.id==='isolate'&&info.isolated);
      const hovered=hoverAction===a.id;
      rect(a.x,a.y,a.w,a.h,active?'#36535f':hovered?'#2c4450':'#21343d',
        active?'#a6ced7':hovered?'#7eabb9':'#405864',12);
      ctx.fillStyle=active?'#ffffff':'#dae7ec';
      ctx.font='600 31px system-ui';ctx.fillText(a.label,a.x+25,a.y+56);
    }
    ctx.fillStyle='#b7cad2';ctx.font='29px system-ui';
    ctx.fillText('TRIGGER: SELECT   ·   B: SWITCH MODE   ·   X: RECENTER',65,877);
    ctx.fillStyle='#94aeb8';ctx.font='25px system-ui';
    ctx.fillText('PROCEDURAL REFERENCE  ·  NOT VALIDATED HUMAN ANATOMY',65,933);
    tex.needsUpdate=true;
  }

  // Crucial XR-04: the WebXR frame owns the viewer pose. Reading an XR camera
  // before its first renderer update (or inheriting the desktop camera's
  // position) can place the model around the user instead of in front.
  function readViewerPose(frame){
    if(!frame||!renderer.xr.isPresenting)return false;
    const space=renderer.xr.getReferenceSpace();
    if(!space)return false;
    const pose=frame.getViewerPose?.(space);
    if(!pose?.transform)return false;
    const p=pose.transform.position,o=pose.transform.orientation;
    if(![p.x,p.y,p.z,o.x,o.y,o.z,o.w].every(Number.isFinite))return false;
    headPosition.set(p.x,p.y,p.z);
    q.set(o.x,o.y,o.z,o.w).normalize();
    headForward.set(0,0,-1).applyQuaternion(q);headForward.y=0;
    if(headForward.lengthSq()<.01)headForward.set(0,0,-1);
    headForward.normalize();
    headRight.set(1,0,0).applyQuaternion(q);headRight.y=0;
    if(headRight.lengthSq()<.01)headRight.set(1,0,0);
    headRight.normalize();
    validViewerPose=true;
    return true;
  }
  function headPose(){
    if(renderer.xr.isPresenting)return validViewerPose;
    camera.updateMatrixWorld(true);
    camera.getWorldPosition(headPosition);camera.getWorldQuaternion(q);
    headForward.set(0,0,-1).applyQuaternion(q);headForward.y=0;
    if(headForward.lengthSq()<.01)headForward.set(0,0,-1);
    headForward.normalize();
    headRight.set(1,0,0).applyQuaternion(q);headRight.y=0;
    if(headRight.lengthSq()<.01)headRight.set(1,0,0);
    headRight.normalize();
    return true;
  }
  function sourceBounds(){
    // Measure only the actual reference airway and schematic lobar meshes.
    // Transform each local mesh box into anatomy-root coordinates so grab
    // rotation, model scale, and desktop camera do not bias the source bounds.
    anatomy.updateMatrixWorld(true);
    const inverseRoot=new THREE.Matrix4().copy(anatomy.matrixWorld).invert();
    const box=new THREE.Box3().makeEmpty();
    const matrix=new THREE.Matrix4();
    anatomy.traverse(mesh=>{
      if(!mesh.isMesh||!mesh.userData.regionId||!mesh.geometry)return;
      if(!mesh.geometry.boundingBox)mesh.geometry.computeBoundingBox();
      matrix.multiplyMatrices(inverseRoot,mesh.matrixWorld);
      box.union(mesh.geometry.boundingBox.clone().applyMatrix4(matrix));
    });
    if(box.isEmpty())return {size:[1,1,1],center:[0,0,0]};
    return {size:box.getSize(new THREE.Vector3()).toArray(),
      center:box.getCenter(new THREE.Vector3()).toArray()};
  }
  function placePanel(){
    if(!headPose())return;
    panel.position.copy(headPosition).addScaledVector(headForward,1.28)
      .addScaledVector(headRight,.59);
    panel.position.y-=.05;
    panel.lookAt(headPosition);
    panel.visible=renderer.xr.isPresenting&&menuVisible;
    drawPanel(true);
  }
  function recenter(){
    if(renderer.xr.isPresenting){
      if(!headPose())return false;
      const bounds=sourceBounds();
      const fit=fitSeatedAnatomy({
        head:headPosition.toArray(),forward:headForward.toArray(),
        bounds:bounds.size,localCenter:bounds.center
      });
      // The full *visible* anatomical envelope, not an arbitrary group
      // origin, starts 2.15 m ahead and approx. 0.56 m across.
      anatomy.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,-1),headForward);
      anatomy.scale.setScalar(fit.scale);
      anatomy.position.fromArray(fit.center).sub(
        new THREE.Vector3(...fit.localCenter)
          .multiplyScalar(fit.scale).applyQuaternion(anatomy.quaternion)
      );
      firstFrame={head:headPosition.toArray(),...fit};
      firstFrameBounds=bounds;
    }else{
      anatomy.position.set(0,1.27,-1.38);
      anatomy.quaternion.identity();
      anatomy.scale.setScalar(1);
    }
    grabbing.fill(false);grabbingSource.fill(null);
    twoHandStart=null;handAnchors.fill(null);transition=null;
    if(renderer.xr.isPresenting)placePanel();
    onReset?.();
    return true;
  }
  function keepComfortable(){
    if(!headPose())return;
    const offset=tmpVec.copy(anatomy.position).sub(headPosition);
    const distance=offset.length();
    const minDistance=Math.max(.52,.42+Math.max(...(firstFrameBounds?.size||[1,1,1]))*anatomy.scale.x*.28);
    if(distance<minDistance||distance>3.5){
      if(distance<.001)offset.copy(headForward);
      offset.normalize().multiplyScalar(THREE.MathUtils.clamp(distance,minDistance,3.5));
      anatomy.position.copy(headPosition).add(offset);
    }
    const s=clampDisplayScale(anatomy.scale.x,scaleLimits.minScale,scaleLimits.maxScale);
    anatomy.scale.setScalar(s);
  }

  function startSummon(){
    if(!renderer.xr.isPresenting||!validViewerPose||mode!=='anatomy'||grabbing.some(Boolean))return false;
    const bounds=sourceBounds();
    const fitScale=workZoneScale(bounds.size,anatomy.scale.x);
    const scale=Math.max(scaleLimits.minScale,fitScale);
    const targetCenter=new THREE.Vector3(...workZoneTarget({
      head:headPosition.toArray(),forward:headForward.toArray()
    }));
    const rootPos=targetCenter.sub(
      new THREE.Vector3(...bounds.center)
        .multiplyScalar(scale).applyQuaternion(anatomy.quaternion)
    );
    transition={elapsed:0,duration:SPATIAL_LIMITS.summonDurationS,
      from:{position:anatomy.position.toArray(),scale:anatomy.scale.x},
      to:{position:rootPos.toArray(),scale}};
    return true;
  }
  function updateTransition(dt){
    if(!transition)return;
    const t=transition;t.elapsed+=Math.min(Math.max(dt,0),.07);
    const state=easedTransform(t.from,t.to,t.elapsed/t.duration);
    anatomy.position.fromArray(state.position);anatomy.scale.setScalar(state.scale);
    if(state.finished)transition=null;
  }
  function readHands(frame){
    const session=renderer.xr.getSession(),space=renderer.xr.getReferenceSpace();
    if(!session||!frame?.getJointPose||!space)return;
    const seen=[false,false];
    handSeen.fill(false); // joint tracking validity must be refreshed each XRFrame
    for(const source of session.inputSources){
      if(!source.hand)continue;
      const i=source.handedness==='left'?0:source.handedness==='right'?1:-1;
      if(i<0)continue;
      seen[i]=true;
      const tip=source.hand.get('index-finger-tip'),thumb=source.hand.get('thumb-tip');
      if(!tip||!thumb)continue;
      const indexPose=frame.getJointPose(tip,space),thumbPose=frame.getJointPose(thumb,space);
      if(!indexPose?.transform||!thumbPose?.transform)continue;
      const a=indexPose.transform.position,b=thumbPose.transform.position;
      const index=new THREE.Vector3(a.x,a.y,a.z),thumbPoint=new THREE.Vector3(b.x,b.y,b.z);
      const middle=index.clone().add(thumbPoint).multiplyScalar(.5);
      handProxies[i].position.copy(middle);
      const wrist=source.hand.get('wrist');
      const wristPose=wrist?frame.getJointPose(wrist,space):null;
      if(wristPose?.transform?.orientation){
        const o=wristPose.transform.orientation;
        handProxies[i].quaternion.set(o.x,o.y,o.z,o.w);
      }else handProxies[i].quaternion.identity();
      handProxies[i].updateMatrixWorld(true);
      handMarkers[i].position.copy(middle);
      handMarkers[i].visible=true;
      handSeen[i]=true;
      const active=pinchActive(pinching[i],index.distanceTo(thumbPoint));
      if(active&&!pinching[i]){
        if(surfaceDistance(middle)<=SPATIAL_LIMITS.nearSurfaceM){
          onTouch?.(middle,SPATIAL_LIMITS.nearSurfaceM);
          beginGesture(i,'hand');
        }
      }else if(!active&&pinching[i]&&grabbingSource[i]==='hand')endGesture(i);
      pinching[i]=active;
    }
    for(let i=0;i<2;i++)if(!seen[i]||!handSeen[i]){
      handMarkers[i].visible=false;handSeen[i]=false;pinching[i]=false;
      if(grabbingSource[i]==='hand')endGesture(i);
    }
  }
  function captureHandAnchor(index){
    const grip=activeGrip(index);grip.updateMatrixWorld(true);
    anatomy.updateMatrixWorld(true);
    handAnchors[index]=new THREE.Matrix4().copy(grip.matrixWorld).invert().multiply(anatomy.matrixWorld);
  }
  function beginGesture(index,source='controller'){
    if(mode!=='anatomy'||!renderer.xr.isPresenting||grabbing[index])return false;
    if(grabbingSource[index]&&grabbingSource[index]!==source)return false;
    const grip=source==='hand'?handProxies[index]:grips[index];
    if(!grip)return false;
    grip.updateMatrixWorld(true);
    const point=grip.getWorldPosition(new THREE.Vector3());
    if(decideInteraction({surfaceDistanceM:surfaceDistance(point)})!=='near')return false;
    transition=null;
    grabbing[index]=true;grabbingSource[index]=source;
    if(grabbing[0]&&grabbing[1]){
      anatomy.updateMatrixWorld(true);
      const left=activeGrip(0).getWorldPosition(new THREE.Vector3());
      const right=activeGrip(1).getWorldPosition(new THREE.Vector3());
      twoHandStart={left:left.toArray(),right:right.toArray(),position:anatomy.position.toArray(),
        quaternion:anatomy.quaternion.toArray(),scale:anatomy.scale.x};
    }else{twoHandStart=null;captureHandAnchor(index)}
    drawPanel(true);
    return true;
  }
  function endGesture(index){
    if(!grabbing[index]){grabbingSource[index]=null;return}
    grabbing[index]=false;twoHandStart=null;handAnchors[index]=null;
    grabbingSource[index]=null;
    const other=index?0:1;
    if(grabbing[other])captureHandAnchor(other);
  }
  function updateGrabs(){
    if(!renderer.xr.isPresenting||mode!=='anatomy')return;
    if(grabbing[0]&&grabbing[1]&&twoHandStart){
      activeGrip(0).updateMatrixWorld(true);activeGrip(1).updateMatrixWorld(true);
      const left=activeGrip(0).getWorldPosition(new THREE.Vector3()).toArray();
      const right=activeGrip(1).getWorldPosition(new THREE.Vector3()).toArray();
      const pose=bimanualPose(twoHandStart,left,right,scaleLimits);
      anatomy.position.fromArray(pose.position);
      anatomy.quaternion.fromArray(pose.quaternion);
      anatomy.scale.setScalar(pose.scale);
      keepComfortable();
    }else{
      const i=grabbing[0]?0:grabbing[1]?1:-1;
      if(i<0||!handAnchors[i])return;
      activeGrip(i).updateMatrixWorld(true);
      tmpMatrix.multiplyMatrices(activeGrip(i).matrixWorld,handAnchors[i]);
      tmpMatrix.decompose(anatomy.position,anatomy.quaternion,anatomy.scale);
      keepComfortable();
    }
  }
  function setMode(next){
    if(!['anatomy','path','section'].includes(next)||next===mode)return;
    mode=next;
    grabbing.fill(false);grabbingSource.fill(null);
    transition=null;twoHandStart=null;handAnchors.fill(null);
    onMode?.(next);drawPanel(true);
  }
  function setPanelVisible(visible){
    menuVisible=Boolean(visible);panel.visible=renderer.xr.isPresenting&&menuVisible;
    drawPanel(true);
  }
  function markTourDone(){
    try{sessionStorage.setItem('pulmosimu.xr.tour.xr05.completed','1')}catch{}
  }
  function perform(id){
    if(id==='tour-next'){
      welcome.next();if(!welcome.get().active)markTourDone();
      drawPanel(true);return;
    }
    if(id==='tour-skip'){welcome.skip();markTourDone();drawPanel(true);return}
    if(id==='help'){welcome.restart();drawPanel(true);return}
    if(['anatomy','path','section'].includes(id))setMode(id);
    else if(id==='next')onNextBranch?.();
    else if(id==='reset')recenter();
    else if(id==='isolate')onIsolate?.();
    else if(id==='hide')setPanelVisible(false);
    else if(id==='procedure'){
      // A deliberate workflow transition: end immersive mode before navigating.
      const session=renderer.xr.getSession();
      if(session)session.end().then(()=>onProcedure?.()).catch(()=>{});
      else onProcedure?.();
    }
    drawPanel(true);
  }
  function pointerRay(controller){
    controller.updateMatrixWorld(true);
    basis.identity().extractRotation(controller.matrixWorld);
    raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
    raycaster.ray.direction.set(0,0,-1).applyMatrix4(basis).normalize();
    raycaster.near=0;raycaster.far=4;
    return raycaster;
  }
  function actionHit(ray){
    if(!panel.visible)return null;
    const hit=ray.intersectObject(panel,false)[0];
    if(!hit?.uv)return null;
    const x=hit.uv.x*1024,y=(1-hit.uv.y)*970;
    return activeActions().find(a=>x>=a.x&&x<=a.x+a.w&&y>=a.y&&y<=a.y+a.h)?.id||null;
  }
  function trigger(controller,source){
    if(mode==='anatomy'&&grabbing.some(Boolean))return;
    // Quest hand selectstart is also a pinch event. Give spatial panel
    // priority and leave anatomy pinch to the direct-contact joint path.
    if(source?.hand){
      const action=actionHit(pointerRay(controller));
      if(action)perform(action);
      return;
    }
    const ray=pointerRay(controller),action=actionHit(ray);
    const picked=action?(perform(action),true):Boolean(onSelect?.(ray,mode));
    if(picked&&!action&&mode==='anatomy'){
      const hand=controller.getWorldPosition(new THREE.Vector3());
      if(decideInteraction({surfaceDistanceM:surfaceDistance(hand),rayHit:true})==='far')startSummon();
    }
    if(picked){
      // Haptics are optional on the Quest WebXR Gamepad API; a missing motor
      // must never prevent selection or make controller input fail.
      try{
        const pad=source?.gamepad;
        if(pad?.hapticActuators?.[0]?.pulse)pad.hapticActuators[0].pulse(.24,38).catch?.(()=>{});
        else pad?.vibrationActuator?.playEffect?.('dual-rumble',{
          duration:38,startDelay:0,strongMagnitude:.2,weakMagnitude:.12}).catch?.(()=>{});
      }catch{}
    }
    drawPanel(true);
  }
  for(let i=0;i<2;i++){
    const proxy=new THREE.Group();scene.add(proxy);handProxies.push(proxy);
    const marker=new THREE.Mesh(new THREE.SphereGeometry(.008,10,8),
      new THREE.MeshBasicMaterial({color:0xb7d9df,transparent:true,opacity:.85,depthTest:false}));
    marker.visible=false;marker.renderOrder=31;scene.add(marker);handMarkers.push(marker);
  }
  for(let i=0;i<2;i++){
    const controller=renderer.xr.getController(i),grip=renderer.xr.getControllerGrip(i);
    scene.add(controller);scene.add(grip);controllers.push(controller);grips.push(grip);
    const beam=new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3(0,0,-1)]),
      new THREE.LineBasicMaterial({color:i?0xc6d8e0:0x9fb9c4,transparent:true,opacity:.65}));
    beam.scale.z=1.8;controller.add(beam);
    try{grip.add(controllerFactory.createControllerModel(grip))}catch{}
    controller.addEventListener('connected',event=>{
      controller.userData.handedness=event.data?.handedness||null;
    });
    controller.addEventListener('selectstart',event=>trigger(controller,event.data));
    controller.addEventListener('squeezestart',()=>beginGesture(i,'controller'));
    controller.addEventListener('squeezeend',()=>endGesture(i));
    controller.addEventListener('disconnected',()=>endGesture(i));
  }
  function buttonEdge(index,buttonNumber,source){
    const pressed=Boolean(source.gamepad?.buttons?.[buttonNumber]?.pressed);
    const prev=Boolean(previousButtons[index]?.[buttonNumber]);
    previousButtons[index][buttonNumber]=pressed;
    return pressed&&!prev;
  }
  function pollGamepads(dt){
    const session=renderer.xr.getSession();
    if(!session)return;
    for(const src of session.inputSources){
      if(!src.gamepad)continue;
      const i=controllers.findIndex(c=>c.userData.handedness===src.handedness);
      if(i<0)continue;
      if(src.handedness==='right'&&buttonEdge(i,4,src))setPanelVisible(!menuVisible);
      if(src.handedness==='right'&&buttonEdge(i,5,src))setMode(mode==='anatomy'?'path':mode==='path'?'section':'anatomy');
      if(src.handedness==='left'&&buttonEdge(i,4,src))recenter();
      if((mode==='path'||mode==='section')&&src.handedness==='left'){
        const [,vertical]=dominantThumbstick(src.gamepad.axes);
        if(vertical!==0){
          if(mode==='path')onRouteStep?.(-vertical*Math.min(dt,.05)*.24);
          else onSectionStep?.(-vertical*Math.min(dt,.05)*.8);
        }
      }
    }
  }
  renderer.xr.addEventListener('sessionstart',()=>{
    started=false;validViewerPose=false;firstFrame=null;firstFrameBounds=null;
    menuVisible=true;previousButtons=[[],[]];
    try{
      if(sessionStorage.getItem('pulmosimu.xr.tour.xr05.completed')==='1')welcome.skip();
      else welcome.restart();
    }catch{welcome.restart()}
    grabbing.fill(false);grabbingSource.fill(null);pinching.fill(false);
    handSeen.fill(false);transition=null;metricsSamples=[];metrics={fps:null,p95Ms:null,slowFrames:0};handAnchors.fill(null);twoHandStart=null;
    handMarkers.forEach(m=>m.visible=false);
    panel.visible=false;
  });
  renderer.xr.addEventListener('sessionend',()=>{
    started=false;validViewerPose=false;grabbing.fill(false);grabbingSource.fill(null);
    pinching.fill(false);transition=null;twoHandStart=null;handAnchors.fill(null);
    handMarkers.forEach(m=>m.visible=false);
    panel.visible=false;mode='anatomy';onMode?.('anatomy');
    anatomy.position.set(0,1.27,-1.38);anatomy.quaternion.identity();anatomy.scale.setScalar(1);
  });
  function update(dt=.016,frame=null){
    if(!renderer.xr.isPresenting)return;
    if(!readViewerPose(frame))return; // do not guess placement from desktop camera
    if(!started){started=true;recenter()}
    const measurement=frameMetrics(metricsSamples,dt,90);
    metricsSamples=measurement.samples;metrics={fps:measurement.fps,p95Ms:measurement.p95Ms,slowFrames:measurement.slowFrames};
    pollGamepads(dt);readHands(frame);updateTransition(dt);updateGrabs();
    // Keep this static readable panel in world space; don't attach it to hands.
    drawPanel();
    const hovered=controllers.map(c=>actionHit(pointerRay(c))).find(Boolean)||null;
    if(hovered!==hoverAction){hoverAction=hovered;drawPanel(true)}
    const right=controllers.find(c=>c.userData.handedness==='right');
    if(right&&!hovered&&!grabbing.some(Boolean))onHover?.(pointerRay(right),mode);
    else onHover?.(null,mode);
  }
  return {update,setMode,recenter,setPanelVisible,
    getMode:()=>mode,getPanelVisible:()=>menuVisible,
    debugState:()=>({mode,menuVisible,grabbing:[...grabbing],scale:anatomy.scale.x,started,
      validViewerPose,firstFrame,firstFrameBounds,tour:welcome.get(),
      metrics,transition:transition?{elapsed:transition.elapsed,duration:transition.duration}:null,
      intent:grabbing.some(Boolean)?'near-grab':transition?'summoning':'idle',
      handTracking:handSeen.slice(),grabbingSource:grabbingSource.slice(),
      modelPosition:anatomy.position.toArray(),headPosition:headPosition.toArray(),
      hands:controllers.map(c=>c.userData.handedness||null)})};
}
