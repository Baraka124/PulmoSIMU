/**
 * PulmoSIMU XR-02 — tracked-controller anatomical manipulation.
 * Interacts exclusively with the experimental reference renderer. Does not
 * mutate the PNEUMO·VR procedure, camera, biopsy, or clinical physiology.
 */
import * as THREE from 'three';
import {XRControllerModelFactory} from 'three/addons/webxr/XRControllerModelFactory.js';
import {bimanualPose,clampDisplayScale,dominantThumbstick} from './spatial-interaction.js';
import {createWelcomeSequence} from './experience-sequence.js';

export function createSpatialRig({
  renderer,scene,camera,anatomy,onSelect,onMode,onReset,onIsolate,onProcedure,onRouteStep,getInfo
}){
  const controllers=[],grips=[],grabbing=[false,false],handAnchors=[null,null];
  let twoHandStart=null,started=false,menuVisible=true,mode='anatomy',previousButtons=[[],[]];
  const welcome=createWelcomeSequence();
  let menuVersion='',hoverAction=null;
  const raycaster=new THREE.Raycaster(),basis=new THREE.Matrix4();
  const headPosition=new THREE.Vector3(),headForward=new THREE.Vector3(),headRight=new THREE.Vector3(),q=new THREE.Quaternion();
  const tmpVec=new THREE.Vector3(),tmpMatrix=new THREE.Matrix4();
  const scaleLimits={minScale:.55,maxScale:1.9};
  const controllerFactory=new XRControllerModelFactory();

  const panelCanvas=document.createElement('canvas');
  panelCanvas.width=1024;panelCanvas.height=970;
  const ctx=panelCanvas.getContext('2d');
  const tex=new THREE.CanvasTexture(panelCanvas);tex.colorSpace=THREE.SRGBColorSpace;
  const panel=new THREE.Mesh(new THREE.PlaneGeometry(.69,.655),
    new THREE.MeshBasicMaterial({map:tex,transparent:true,side:THREE.DoubleSide,depthWrite:false,depthTest:false}));
  panel.renderOrder=45;panel.visible=false;scene.add(panel);

  const actions=[
    {id:'anatomy',label:'EXPLORE',x:65,y:442,w:434,h:108},
    {id:'path',label:'AIRWAY GUIDE',x:526,y:442,w:434,h:108},
    {id:'reset',label:'RECENTER',x:65,y:576,w:434,h:108},
    {id:'isolate',label:'ISOLATE LOBE',x:526,y:576,w:434,h:108},
    {id:'procedure',label:'PROCEDURES',x:65,y:710,w:434,h:108},
    {id:'hide',label:'HIDE PANEL',x:526,y:710,w:434,h:108}
  ];
  const tutorialActions=[
    {id:'tour-next',label:'NEXT STEP',x:65,y:710,w:434,h:108},
    {id:'tour-skip',label:'EXPLORE NOW',x:526,y:710,w:434,h:108}
  ];
  const activeActions=()=>welcome.get().active?tutorialActions:actions;
  function textWrap(text,x,y,maxWidth,lineHeight,maxLines=5){
    const words=String(text).split(/\\s+/);let line='',row=0;
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
    const key=JSON.stringify([mode,menuVisible,info.region,info.name,info.isolated,info.progress,hoverAction,tour.active,tour.index]);
    if(!force&&key===menuVersion)return;
    menuVersion=key;ctx.clearRect(0,0,1024,970);
    rect(0,0,1024,970,'rgba(18,31,39,.96)','#526974',16);
    rect(0,0,1024,9,'#9cbec8',null,0);
    if(tour.active){
      ctx.textAlign='left';ctx.fillStyle='#b8d3db';ctx.font='600 29px system-ui';
      ctx.fillText('PULMOSIMU  /  YOUR FIRST XR SESSION',65,91);
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
    ctx.fillText('PULMOSIMU  /  SPATIAL LAB',65,88);
    ctx.fillStyle='#ecf2f4';ctx.font='600 62px system-ui';
    ctx.fillText((info.name||'Reference anatomy').slice(0,26),65,190);
    ctx.fillStyle='#c1d3d9';ctx.font='34px system-ui';
    ctx.fillText('Region   '+(info.region||'—')+'     ·     '+(mode==='path'?'AIRWAY GUIDE':'ANATOMY'),65,262);
    rect(65,305,894,95,'rgba(42,66,77,.42)',null,10);
    ctx.fillStyle='#cfdee3';ctx.font='29px system-ui';
    ctx.fillText(mode==='path'?'LEFT STICK: ADVANCE ROUTE   ·   NO CAMERA MOTION':
      'SIDE GRIP: HOLD   ·   BOTH GRIPS: RESIZE & ROTATE',89,361);
    for(const a of activeActions()){
      const active=a.id===mode || (a.id==='isolate'&&info.isolated);
      const hovered=hoverAction===a.id;
      rect(a.x,a.y,a.w,a.h,active?'#36535f':hovered?'#2c4450':'#21343d',
        active?'#a6ced7':hovered?'#7eabb9':'#405864',12);
      ctx.fillStyle=active?'#ffffff':'#dae7ec';
      ctx.font='600 34px system-ui';ctx.fillText(a.label,a.x+26,a.y+68);
    }
    ctx.fillStyle='#b7cad2';ctx.font='29px system-ui';
    ctx.fillText('TRIGGER: SELECT    ·    A: TOGGLE PANEL    ·    B: SWITCH MODE',65,877);
    ctx.fillStyle='#94aeb8';ctx.font='25px system-ui';
    ctx.fillText('PROCEDURAL REFERENCE  ·  NOT VALIDATED HUMAN ANATOMY',65,933);
    tex.needsUpdate=true;
  }
  function headPose(){
    const xrCamera=renderer.xr.getCamera(camera);
    xrCamera.updateMatrixWorld(true);
    xrCamera.getWorldPosition(headPosition);
    xrCamera.getWorldQuaternion(q);
    headForward.set(0,0,-1).applyQuaternion(q);headForward.y=0;
    if(headForward.lengthSq()<.02)headForward.set(0,0,-1);
    headForward.normalize();
    headRight.set(1,0,0).applyQuaternion(q);headRight.y=0;
    if(headRight.lengthSq()<.02)headRight.set(1,0,0);
    headRight.normalize();
  }
  function placePanel(){
    headPose();
    panel.position.copy(headPosition).addScaledVector(headForward,1.42)
      .addScaledVector(headRight,.82);
    panel.position.y-=.04;
    panel.lookAt(headPosition);
    panel.visible=renderer.xr.isPresenting&&menuVisible;
    drawPanel(true);
  }
  function recenter(){
    if(renderer.xr.isPresenting){
      headPose();
      anatomy.position.copy(headPosition).addScaledVector(headForward,1.44);
      anatomy.position.y-=.18;
      anatomy.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,-1),headForward);
    }else{
      anatomy.position.set(0,1.27,-1.38);anatomy.quaternion.identity();
    }
    anatomy.scale.setScalar(1);
    grabbing.fill(false);twoHandStart=null;handAnchors.fill(null);
    if(renderer.xr.isPresenting)placePanel();
    onReset?.();
  }
  function keepComfortable(){
    headPose();
    const offset=tmpVec.copy(anatomy.position).sub(headPosition);
    const distance=offset.length();
    if(distance<.68||distance>3.3){
      if(distance<.001)offset.copy(headForward);
      offset.normalize().multiplyScalar(THREE.MathUtils.clamp(distance,.68,3.3));
      anatomy.position.copy(headPosition).add(offset);
    }
    const s=clampDisplayScale(anatomy.scale.x,scaleLimits.minScale,scaleLimits.maxScale);
    anatomy.scale.setScalar(s);
  }
  function captureHandAnchor(index){
    const grip=grips[index];grip.updateMatrixWorld(true);
    anatomy.updateMatrixWorld(true);
    handAnchors[index]=new THREE.Matrix4().copy(grip.matrixWorld).invert().multiply(anatomy.matrixWorld);
  }
  function beginGesture(index){
    if(mode!=='anatomy'||!renderer.xr.isPresenting)return;
    grabbing[index]=true;grips[index].updateMatrixWorld(true);
    if(grabbing[0]&&grabbing[1]){
      anatomy.updateMatrixWorld(true);
      const left=grips[0].getWorldPosition(new THREE.Vector3()),right=grips[1].getWorldPosition(new THREE.Vector3());
      twoHandStart={left:left.toArray(),right:right.toArray(),position:anatomy.position.toArray(),
        quaternion:anatomy.quaternion.toArray(),scale:anatomy.scale.x};
    }else{twoHandStart=null;captureHandAnchor(index)}
    drawPanel(true);
  }
  function endGesture(index){
    if(!grabbing[index])return;
    grabbing[index]=false;twoHandStart=null;handAnchors[index]=null;
    const other=index?0:1;
    if(grabbing[other])captureHandAnchor(other);
  }
  function updateGrabs(){
    if(!renderer.xr.isPresenting||mode!=='anatomy')return;
    if(grabbing[0]&&grabbing[1]&&twoHandStart){
      grips.forEach(g=>g.updateMatrixWorld(true));
      const left=grips[0].getWorldPosition(new THREE.Vector3()).toArray();
      const right=grips[1].getWorldPosition(new THREE.Vector3()).toArray();
      const pose=bimanualPose(twoHandStart,left,right,scaleLimits);
      anatomy.position.fromArray(pose.position);
      anatomy.quaternion.fromArray(pose.quaternion);
      anatomy.scale.setScalar(pose.scale);
      keepComfortable();
    }else{
      const i=grabbing[0]?0:grabbing[1]?1:-1;
      if(i<0||!handAnchors[i])return;
      grips[i].updateMatrixWorld(true);
      tmpMatrix.multiplyMatrices(grips[i].matrixWorld,handAnchors[i]);
      tmpMatrix.decompose(anatomy.position,anatomy.quaternion,anatomy.scale);
      keepComfortable();
    }
  }
  function setMode(next){
    if(!['anatomy','path'].includes(next)||next===mode)return;
    mode=next;
    grabbing.fill(false);twoHandStart=null;handAnchors.fill(null);
    onMode?.(next);drawPanel(true);
  }
  function setPanelVisible(visible){
    menuVisible=Boolean(visible);panel.visible=renderer.xr.isPresenting&&menuVisible;
    drawPanel(true);
  }
  function perform(id){
    if(id==='tour-next'){welcome.next();drawPanel(true);return}
    if(id==='tour-skip'){welcome.skip();drawPanel(true);return}
    if(id==='anatomy'||id==='path')setMode(id);
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
  function trigger(controller){
    if(mode==='anatomy'&&grabbing.some(Boolean))return;
    const ray=pointerRay(controller),action=actionHit(ray);
    if(action)perform(action);
    else onSelect?.(ray,mode);
    drawPanel(true);
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
    controller.addEventListener('selectstart',()=>trigger(controller));
    controller.addEventListener('squeezestart',()=>beginGesture(i));
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
      if(src.handedness==='right'&&buttonEdge(i,5,src))setMode(mode==='anatomy'?'path':'anatomy');
      if(src.handedness==='left'&&buttonEdge(i,4,src))recenter();
      if(mode==='path'&&src.handedness==='left'){
        const [,vertical]=dominantThumbstick(src.gamepad.axes);
        if(vertical!==0)onRouteStep?.(-vertical*Math.min(dt,.05)*.24);
      }
    }
  }
  renderer.xr.addEventListener('sessionstart',()=>{
    started=false;menuVisible=true;previousButtons=[[],[]];
    welcome.restart();
    grabbing.fill(false);handAnchors.fill(null);twoHandStart=null;
    panel.visible=false;
  });
  renderer.xr.addEventListener('sessionend',()=>{
    started=false;grabbing.fill(false);twoHandStart=null;handAnchors.fill(null);
    panel.visible=false;mode='anatomy';onMode?.('anatomy');
    anatomy.position.set(0,1.27,-1.38);anatomy.quaternion.identity();anatomy.scale.setScalar(1);
  });
  function update(dt=.016){
    if(!renderer.xr.isPresenting)return;
    if(!started){started=true;recenter();placePanel()}
    pollGamepads(dt);updateGrabs();
    // Keep this static readable panel in world space; don't attach it to hands.
    drawPanel();
    const hovered=controllers.map(c=>actionHit(pointerRay(c))).find(Boolean)||null;
    if(hovered!==hoverAction){hoverAction=hovered;drawPanel(true)}
  }
  return {update,setMode,recenter,setPanelVisible,
    getMode:()=>mode,getPanelVisible:()=>menuVisible,
    debugState:()=>({mode,menuVisible,grabbing:[...grabbing],scale:anatomy.scale.x,started,tour:welcome.get(),hands:controllers.map(c=>c.userData.handedness||null)})};
}
