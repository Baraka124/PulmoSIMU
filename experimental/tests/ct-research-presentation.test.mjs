import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {webcrypto} from 'node:crypto';
const source=readFileSync(new URL('../anatomy/research-view.js',import.meta.url),'utf8');
const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
const w={};
runInNewContext(source,{window:w,ArrayBuffer,Uint8Array,crypto:webcrypto});
const bridge=w.PulmoCTResearchBridge;
const sha='5b234ac92a9a5d58d15f3f43144d31612d10ff6e60777d8c273cd52b36b13087';
const sample={
 geometry_class:'PRELIMINARY_CT_DERIVED_AIRSPACE_MASK',
 source_sha256:sha,mesh_sha256:sha,
 model_from_ras_mm:[[.001,0,0,0],[0,0,.001,.39],[0,-.001,0,.122],[0,0,0,1]],
 quality_gates:{right_lung_mask_leakage:'FLAGGED_FOR_REVIEW'}
};
test('CT research study sidecar is NEVER mistaken for patient airway graph',()=>{
 assert.equal(bridge.isResearchManifest(sample),true);
 assert.equal(bridge.isResearchManifest({...sample,segments:[{id:0}]}),false);
 assert.equal(bridge.isResearchManifest({...sample,mesh_sha256:'bad-hash'}),false);
 const status=bridge.describe(sample);
 assert.equal(status.bronchoscopyAllowed,false);
 assert.equal(status.anatomyValidated,false);
 assert.match(status.warning,/QA FLAGGED/);
 assert.ok(main.includes('if(research?.isResearchManifest(meta))'));
 assert.ok(main.includes('await research.verifyGLB(gltfBuf,meta)'));
 assert.ok(main.includes('patientData.meta=null;patientData.xrAudit=null;'));
 assert.ok(main.includes("setViewMode('research')"));
 assert.ok(main.includes('Unsupported JSON: use a CT source manifest'));
});
test('CT research GLB requires exact hash before being displayed',async()=>{
 const buf=new TextEncoder().encode('CT airspace source, example test mesh bytes').buffer;
 const hash=Array.from(new Uint8Array(await webcrypto.subtle.digest('SHA-256',buf)),
    v=>v.toString(16).padStart(2,'0')).join('');
 const ok=await bridge.verifyGLB(buf,{...sample,mesh_sha256:hash},webcrypto);
 assert.equal(ok.verifiedBytes,true);
 assert.equal(ok.clinicalValidated,false);
 assert.equal(ok.airwayRegistrationVerified,false);
 await assert.rejects(bridge.verifyGLB(buf,sample,webcrypto),/SHA-256 does not match/);
 await assert.rejects(bridge.verifyGLB(buf,{...sample,segments:[]},webcrypto),/Not a recognized/);
});
test('research presentation isolates reference pathologies, restores state and blocks CT bronchoscopy',()=>{
 for(const term of [
  "document.body.classList.toggle('ct-source-mode',research)",
  "mGrp.children.forEach(child=>{",
  'ctResearchHidden.set(child,child.visible);child.visible=false',
  'ctResearchHidden.forEach((visible,object)=>',
  "Object.values(pathoMeshes).forEach(p=>{if(p.mesh)p.mesh.visible=patientGraph})",
  "getMeshMode:()=>meshMode==='research'?'patient':meshMode",
  "meshMode==='patient'||meshMode==='research'",
  "body.ct-source-mode #rp,",
  "body.ct-source-mode #seg-card,",
  "body.ct-source-mode #bro-view,",
  "body.ct-source-mode #sp-panel,",
  "window.resetCTResearchView=()=>",
  "_focusCameraOnGLB(patientData.gltf)",
  "window.toggleCTResearchMaterial=()=>",
  "child.material=patientData.researchMaterials.get(child)"
 ]) assert.ok(main.includes(term),'missing CT research-mode invariant '+term);
 assert.ok(main.includes('research:'+'\x27'+'CT-derived airspace surface · research only'));
 assert.ok(main.includes('CT RESEARCH SURFACE · UNVALIDATED'));
});
test('clean original experience is preserved and source import is explicitly research-only',()=>{
 assert.ok(main.includes("setViewMode('reference')"));
 assert.ok(main.includes("setViewMode('research')"));
 assert.ok(main.includes("setView('overview')"));
 assert.ok(main.includes("getPatientAudit:()=>patientData.xrAudit||null"));
 assert.ok(main.includes("if(next==='procedure'"),'XR procedure guard lives in canonical immersive adapter');
});
