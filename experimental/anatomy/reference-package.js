/* INF05B — exact v1 reference study package contract. No auth or storage. */
(function(root){
"use strict";
const spec=Object.freeze({
 "source.nii":["ct_nifti",null],
 "source.ingest.json":["manifest",null],
 "airway-mask/mask.raw":["mask","airway_lumen"],
 "airway-mask/mask.json":["manifest","airway_lumen"],
 "airway-mask/topology.json":["quality_report","airway_lumen"],
 "airway-mask/airway-skeleton.json":["airway_graph","airway_lumen"],
 "airway-surface/airway.glb":["glb","airway_lumen"],
 "airway-surface/reconstruction.json":["manifest","airway_lumen"],
 "airway-surface/fidelity.json":["quality_report","airway_lumen"],
 "study-package.json":["manifest",null]
});
const required=Object.keys(spec).filter(p=>p!=="source.ingest.json");
const kinds=["airway_lumen","pulmonary_artery","pulmonary_vein","RUL","RML","RLL","LUL","LLL","lung"],lobes=["RUL","RML","RLL","LUL","LLL"],parts=["mask/mask.raw","mask/mask.json","mask/topology.json","surface.glb","surface.json"];
const multiSpec=Object.freeze({"source.nii":["ct_nifti",null],"pulmonary.glb":["glb",null],"reconstruction.json":["manifest",null],"study-package.json":["manifest",null],"structures/airway_lumen/mask/mask.raw":["mask","airway_lumen"],"structures/airway_lumen/mask/mask.json":["manifest","airway_lumen"],"structures/airway_lumen/mask/topology.json":["quality_report","airway_lumen"],"structures/airway_lumen/surface.glb":["glb","airway_lumen"],"structures/airway_lumen/surface.json":["manifest","airway_lumen"],"structures/pulmonary_artery/mask/mask.raw":["mask","pulmonary_artery"],"structures/pulmonary_artery/mask/mask.json":["manifest","pulmonary_artery"],"structures/pulmonary_artery/mask/topology.json":["quality_report","pulmonary_artery"],"structures/pulmonary_artery/surface.glb":["glb","pulmonary_artery"],"structures/pulmonary_artery/surface.json":["manifest","pulmonary_artery"],"structures/pulmonary_vein/mask/mask.raw":["mask","pulmonary_vein"],"structures/pulmonary_vein/mask/mask.json":["manifest","pulmonary_vein"],"structures/pulmonary_vein/mask/topology.json":["quality_report","pulmonary_vein"],"structures/pulmonary_vein/surface.glb":["glb","pulmonary_vein"],"structures/pulmonary_vein/surface.json":["manifest","pulmonary_vein"],"structures/RUL/mask/mask.raw":["mask","RUL"],"structures/RUL/mask/mask.json":["manifest","RUL"],"structures/RUL/mask/topology.json":["quality_report","RUL"],"structures/RUL/surface.glb":["glb","RUL"],"structures/RUL/surface.json":["manifest","RUL"],"structures/RML/mask/mask.raw":["mask","RML"],"structures/RML/mask/mask.json":["manifest","RML"],"structures/RML/mask/topology.json":["quality_report","RML"],"structures/RML/surface.glb":["glb","RML"],"structures/RML/surface.json":["manifest","RML"],"structures/RLL/mask/mask.raw":["mask","RLL"],"structures/RLL/mask/mask.json":["manifest","RLL"],"structures/RLL/mask/topology.json":["quality_report","RLL"],"structures/RLL/surface.glb":["glb","RLL"],"structures/RLL/surface.json":["manifest","RLL"],"structures/LUL/mask/mask.raw":["mask","LUL"],"structures/LUL/mask/mask.json":["manifest","LUL"],"structures/LUL/mask/topology.json":["quality_report","LUL"],"structures/LUL/surface.glb":["glb","LUL"],"structures/LUL/surface.json":["manifest","LUL"],"structures/LLL/mask/mask.raw":["mask","LLL"],"structures/LLL/mask/mask.json":["manifest","LLL"],"structures/LLL/mask/topology.json":["quality_report","LLL"],"structures/LLL/surface.glb":["glb","LLL"],"structures/LLL/surface.json":["manifest","LLL"],"structures/lung/mask/mask.raw":["mask","lung"],"structures/lung/mask/mask.json":["manifest","lung"],"structures/lung/mask/topology.json":["quality_report","lung"],"structures/lung/surface.glb":["glb","lung"],"structures/lung/surface.json":["manifest","lung"],"structures/airway_lumen/mask/airway-skeleton.json":["airway_graph","airway_lumen"]});
const isMulti=inventory=>Object.hasOwn(inventory,"pulmonary.glb");
function structures(inventory){return kinds.filter(k=>Object.hasOwn(inventory,"structures/"+k+"/surface.glb"));}
function pathsFor(inventory,mode){return isMulti(inventory)?["source.nii","pulmonary.glb","reconstruction.json"]:mode==="immersive"?immersivePaths:viewerPaths;}
function checkMulti(inventory){
 need(["source.nii","pulmonary.glb","reconstruction.json","study-package.json"].every(p=>Object.hasOwn(inventory,p)),"Missing combined source assets");
 const ks=structures(inventory);need(ks.length>0,"No complete anatomical structure");
 for(const k of kinds){const count=parts.filter(p=>Object.hasOwn(inventory,"structures/"+k+"/"+p)).length;need(count===0||count===parts.length,"Incomplete structure "+k);}
 need(!Object.hasOwn(inventory,"structures/airway_lumen/mask/airway-skeleton.json")||ks.includes("airway_lumen"),"Orphan airway graph");
 need(inventory["reconstruction.json"].bytes<=2097152,"Reconstruction manifest exceeds limit");
}
function checkSurface(inventory,m){
 if(!isMulti(inventory))return true;
 const ks=structures(inventory),expected=ks.filter(k=>k!=="lung"||!lobes.every(l=>ks.includes(l)));
 need(m?.validation==="UNREVIEWED_MULTI_STRUCTURE_SURFACES"&&m.source_sha256===inventory["source.nii"].sha256&&m.mesh_sha256===inventory["pulmonary.glb"].sha256,"Combined surface identity mismatch");
 need(m.structure_inventory&&Object.keys(m.structure_inventory).length===ks.length&&ks.every(k=>m.structure_inventory[k]===inventory["structures/"+k+"/surface.glb"].sha256),"Structure surface identity mismatch");
 need(Array.isArray(m.combined_structure_names)&&m.combined_structure_names.length===expected.length&&new Set(m.combined_structure_names).size===expected.length&&expected.every(k=>m.combined_structure_names.includes(k)),"Combined structure names mismatch");
 const a=m.model_from_ras_mm;need(Array.isArray(a)&&a.length===4&&a.every(r=>Array.isArray(r)&&r.length===4&&r.every(Number.isFinite))&&a[3].every((v,i)=>v===(i===3?1:0))&&[[.001,0,0],[0,0,.001],[0,-.001,0]].every((r,i)=>r.every((v,j)=>Math.abs(v-a[i][j])<1e-10)),"Common metric frame mismatch");
 return true;
}
const viewerPaths=Object.freeze(["source.nii","airway-surface/airway.glb","airway-surface/reconstruction.json","airway-mask/mask.json","airway-mask/mask.raw","airway-mask/airway-skeleton.json","airway-surface/fidelity.json"]);
const immersivePaths=Object.freeze(["source.nii","airway-surface/airway.glb","airway-surface/reconstruction.json"]);
const uuid=x=>typeof x==="string"&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(x);
const hash=x=>typeof x==="string"&&/^[0-9a-f]{64}$/.test(x);
const need=(ok,msg)=>{if(!ok)throw Error(msg);};
function plan(study,assets){
 need(uuid(study?.id)&&study.source_kind==="public_reference"&&["review_pending","research_ready"].includes(study.status),"Published reference study required");
 need(Array.isArray(assets)&&assets.length>=9&&assets.length<=50,"Incomplete reference inventory");
 const multi=assets.some(a=>a?.object_key===study.id+"/v1/pulmonary.glb"),schema=multi?multiSpec:spec;
 if(!multi)need([9,10].includes(assets.length),"Invalid airway inventory");
 const prefix=study.id+"/v1/",out=Object.create(null);let total=0;
 for(const a of assets){
  need(typeof a.object_key==="string"&&a.object_key.startsWith(prefix),"Cross-study asset rejected");
  const path=a.object_key.slice(prefix.length),s=Object.hasOwn(schema,path)?schema[path]:null;
  need(s&&!Object.hasOwn(out,path)&&a.kind===s[0]&&(a.structure??null)===s[1]&&a.version===1&&hash(a.sha256)&&Number.isSafeInteger(a.bytes)&&a.bytes>0&&a.bytes<=50000000,"Invalid reference asset");
  total+=a.bytes;out[path]=Object.freeze({path,object_key:a.object_key,kind:a.kind,structure:a.structure??null,version:1,bytes:a.bytes,sha256:a.sha256});
 }
 need(total<=200262144,"Reference package exceeds memory budget");
 if(multi){checkMulti(out);need(out["study-package.json"].bytes<=262144,"Package manifest size limit");}
 else{need(required.every(p=>Object.hasOwn(out,p)),"Reference package missing");need(out["study-package.json"].bytes<=262144&&out["airway-mask/mask.json"].bytes<=65536&&out["airway-surface/reconstruction.json"].bytes<=2097152&&out["airway-surface/fidelity.json"].bytes<=262144,"Manifest size limit exceeded");}
 return Object.freeze(out);
}
function manifest(study,inventory,m){
 need(m?.schema===(isMulti(inventory)?"pulmo.multi-structure-package/1":"pulmo.study-package/1")&&m.study_id===study.id&&m.status==="review_pending"&&m.clinical_navigation_allowed===false&&m.source_sha256===inventory["source.nii"].sha256,"Package identity or review gate mismatch");
 need(Array.isArray(m.assets)&&m.assets.length===Object.keys(inventory).length-1,"Package inventory mismatch");
 const seen=new Set();
 for(const a of m.assets){
  const p=a?.path,b=Object.hasOwn(inventory,p)?inventory[p]:null;
  need(b&&p!=="study-package.json"&&!seen.has(p)&&["path","object_key","kind","version","bytes","sha256"].every(k=>a[k]===b[k])&&(a.structure??null)===b.structure,"Package asset mismatch");
  seen.add(p);
 }
 if(isMulti(inventory)){const ks=structures(inventory);need(Array.isArray(m.available_structures)&&m.available_structures.length===ks.length&&new Set(m.available_structures).size===ks.length&&ks.every(k=>m.available_structures.includes(k)),"Package structures mismatch");}
 return true;
}
root.PulmoReferencePackage=Object.freeze({plan,manifest,viewerPaths,immersivePaths,isMulti,pathsFor,checkSurface});
})(typeof window!=="undefined"?window:globalThis);
