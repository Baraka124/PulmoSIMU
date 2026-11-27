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
const viewerPaths=Object.freeze(["source.nii","airway-surface/airway.glb","airway-surface/reconstruction.json","airway-mask/mask.json","airway-mask/mask.raw","airway-mask/airway-skeleton.json","airway-surface/fidelity.json"]);
const uuid=x=>typeof x==="string"&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(x);
const hash=x=>typeof x==="string"&&/^[0-9a-f]{64}$/.test(x);
const need=(ok,msg)=>{if(!ok)throw Error(msg);};
function plan(study,assets){
 need(uuid(study?.id)&&study.source_kind==="public_reference"&&["review_pending","research_ready"].includes(study.status),"Published reference study required");
 need(Array.isArray(assets)&&[9,10].includes(assets.length),"Incomplete reference inventory");
 const prefix=study.id+"/v1/",out=Object.create(null);let total=0;
 for(const a of assets){
  need(typeof a.object_key==="string"&&a.object_key.startsWith(prefix),"Cross-study asset rejected");
  const path=a.object_key.slice(prefix.length),s=Object.hasOwn(spec,path)?spec[path]:null;
  need(s&&!Object.hasOwn(out,path)&&a.kind===s[0]&&(a.structure??null)===s[1]&&a.version===1&&hash(a.sha256)&&Number.isSafeInteger(a.bytes)&&a.bytes>0&&a.bytes<=50000000,"Invalid reference asset");
  total+=a.bytes;out[path]=Object.freeze({path,object_key:a.object_key,kind:a.kind,structure:a.structure??null,version:1,bytes:a.bytes,sha256:a.sha256});
 }
 need(total<=200262144&&required.every(p=>Object.hasOwn(out,p)),"Reference package missing or exceeds memory budget");
 need(out["study-package.json"].bytes<=262144&&out["airway-mask/mask.json"].bytes<=65536&&out["airway-surface/reconstruction.json"].bytes<=2097152&&out["airway-surface/fidelity.json"].bytes<=262144,"Manifest size limit exceeded");
 return Object.freeze(out);
}
function manifest(study,inventory,m){
 need(m?.schema==="pulmo.study-package/1"&&m.study_id===study.id&&m.status==="review_pending"&&m.clinical_navigation_allowed===false&&m.source_sha256===inventory["source.nii"].sha256,"Package identity or review gate mismatch");
 need(Array.isArray(m.assets)&&m.assets.length===Object.keys(inventory).length-1,"Package inventory mismatch");
 const seen=new Set();
 for(const a of m.assets){
  const p=a?.path,b=Object.hasOwn(inventory,p)?inventory[p]:null;
  need(b&&p!=="study-package.json"&&!seen.has(p)&&["path","object_key","kind","version","bytes","sha256"].every(k=>a[k]===b[k])&&(a.structure??null)===b.structure,"Package asset mismatch");
  seen.add(p);
 }
 return true;
}
root.PulmoReferencePackage=Object.freeze({plan,manifest,viewerPaths});
})(typeof window!=="undefined"?window:globalThis);
