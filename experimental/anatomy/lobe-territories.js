/* PulmoSIMU A2.2 — source-surface pulmonary lobe territory workbench.
 * Lobe boundaries are adjustable ATLAS-SEED hypotheses, not CT-derived
 * fissure detections. This view never alters the source mesh or bronchial
 * reference graph, and never creates gaps, fake vessels, or procedural data.
 */
(function(global){
'use strict';
const names=Object.freeze({
  RUL:'Right upper',RML:'Right middle',RLL:'Right lower',
  LUL:'Left upper',LLL:'Left lower'
});
const palette=Object.freeze({
  RUL:'#AFC0C4',RML:'#CCB495',RLL:'#829CA9',
  LUL:'#C8A9A7',LLL:'#A4AF9D'
});
const DEFAULT=Object.freeze({rightMajor:0,rightMinor:0,leftMajor:0,selected:'all'});
const clamp=(a,low,high)=>Math.max(low,Math.min(high,Number(a)||0));
const parameters=(a={})=>Object.freeze({
  rightMajor:clamp(a.rightMajor,-35,35),
  rightMinor:clamp(a.rightMinor,-30,30),
  leftMajor:clamp(a.leftMajor,-35,35),
  selected:Object.hasOwn(names,a.selected)?a.selected:'all'
});
function classifyRAS(ras,raw={}){
  const [x,y,z]=ras.map(Number),p=parameters(raw);
  if(![x,y,z].every(Number.isFinite))throw Error('Non-finite source RAS coordinates');
  // RAS: +X patient right, +Y anterior, +Z superior (millimetres).
  // These editable planes are atlas INITIATION, never detected fissures.
  const obliqueR=-394-.69*(y-110)+.12*(x-65)+p.rightMajor;
  const obliqueL=-399-.73*(y-110)-.10*(Math.abs(x)-58)+p.leftMajor;
  const horizontalR=-348-.12*(y-110)+p.rightMinor;
  if(x<0)return z<obliqueL?'LLL':'LUL';
  if(z<obliqueR)return 'RLL';
  // Middle territory is antero-inferior to the upper lobe, while the
  // posterobasal surface above the major plane remains RUL for review.
  if(y>=104&&z<horizontalR)return 'RML';
  return 'RUL';
}
function matrixFromManifest(THREE,manifest){
  const m=manifest?.model_from_ras_mm;
  if(!Array.isArray(m)||m.length!==4||m.some(x=>!Array.isArray(x)||x.length!==4||
    !x.every(Number.isFinite)))throw Error('An exact model-from-RAS transform is required');
  const M=new THREE.Matrix4().set(...m.flat());
  if(Math.abs(M.determinant())<1e-16)throw Error('Singular CT coordinate transform');
  return M.invert();
}
function create(THREE,root,modelRoot,manifest){
  if(!THREE?.Matrix4||!root?.isObject3D||!modelRoot?.isObject3D)
    throw Error('The original source model must be loaded');
  const modelToRAS=matrixFromManifest(THREE,manifest);
  const records=[],point=new THREE.Vector3(),modelPoint=new THREE.Vector3();
  const inv=new THREE.Matrix4(),rel=new THREE.Matrix4();
  modelRoot.updateMatrixWorld(true);
  root.updateMatrixWorld(true);
  inv.copy(modelRoot.matrixWorld).invert();
  root.traverse(mesh=>{
    if(!mesh.isMesh||!mesh.geometry?.attributes?.position)return;
    const base=mesh.geometry;
    mesh.updateMatrixWorld(true);
    rel.multiplyMatrices(inv,mesh.matrixWorld);
    const localToRAS=modelToRAS.clone().multiply(rel);
    // Clone topology before adding vertex colours, so returning to the
    // genuine GLB restores its exact original geometry and materials.
    const working=base.clone(),originalMaterial=mesh.material;
    const p=working.getAttribute('position');
    const ras=new Float32Array(p.count*3);
    for(let i=0;i<p.count;i++){
      point.fromBufferAttribute(p,i);
      modelPoint.copy(point).applyMatrix4(localToRAS);
      ras[3*i]=modelPoint.x;ras[3*i+1]=modelPoint.y;ras[3*i+2]=modelPoint.z;
    }
    const colors=new Float32Array(p.count*3);
    working.setAttribute('color',new THREE.BufferAttribute(colors,3));
    const atlasMaterial=new THREE.MeshStandardMaterial({
      vertexColors:true,roughness:.90,metalness:0,side:THREE.DoubleSide
    });
    records.push({mesh,base,originalMaterial,working,atlasMaterial,ras,colors});
  });
  if(!records.length)throw Error('No CT-derived mesh geometry available');
  let active=false,settings=parameters(DEFAULT);
  const modelColors=Object.fromEntries(Object.keys(names).map(k=>[k,new THREE.Color(palette[k])]));
  const dim=new THREE.Color('#465663'),seam=new THREE.Color('#43515c');
  const color=new THREE.Color();
  const counts={};
  function update(raw={}){
    settings=parameters({...settings,...raw});
    Object.keys(names).forEach(k=>counts[k]=0);
    for(const rec of records){
      for(let i=0;i<rec.ras.length;i+=3){
        const x=rec.ras[i],y=rec.ras[i+1],z=rec.ras[i+2];
        const label=classifyRAS([x,y,z],settings);counts[label]++;
        const major=label==='LLL'||label==='LUL'
          ?z-(-399-.73*(y-110)-.10*(Math.abs(x)-58)+settings.leftMajor)
          :z-(-394-.69*(y-110)+.12*(x-65)+settings.rightMajor);
        const minor=x>=0?z-(-348-.12*(y-110)+settings.rightMinor):999;
        color.copy(modelColors[label]);
        // A restrained seam indicates the atlas plane, not a fissure
        // extracted or proven by CT. Do NOT alter lung topology.
        const near=Math.abs(major)<2.6||
          (label!=='RLL'&&x>=0&&y>=104&&Math.abs(minor)<2.6);
        if(near)color.lerp(seam,.62);
        if(settings.selected!=='all'&&settings.selected!==label)color.lerp(dim,.83);
        rec.colors[i]=color.r;rec.colors[i+1]=color.g;rec.colors[i+2]=color.b;
      }
      rec.working.getAttribute('color').needsUpdate=true;
    }
    return snapshot();
  }
  function enable(){
    for(const rec of records){
      rec.mesh.geometry=rec.working;rec.mesh.material=rec.atlasMaterial;
    }
    active=true;return update();
  }
  function disable(){
    for(const rec of records){
      rec.mesh.geometry=rec.base;rec.mesh.material=rec.originalMaterial;
    }
    active=false;return snapshot();
  }
  function dispose(){
    disable();
    for(const rec of records){
      rec.working.dispose();rec.atlasMaterial.dispose();
    }
  }
  function snapshot(){return Object.freeze({
    mode:active?'editable-atlas-territories':'original-source',
    label:'ATLAS-SEEDED LOBE TERRITORIES / NOT FISSURE SEGMENTATION',
    settings,vertices:Object.values(counts).reduce((a,b)=>a+b,0),
    counts:{...counts},sourceSurfacePreserved:true
  });}
  return Object.freeze({enable,disable,update,snapshot,dispose});
}
global.PulmoCTLobeTerritories=Object.freeze({
  DEFAULT,names,palette,parameters,classifyRAS,create
});
})(typeof window!=='undefined'?window:globalThis);
