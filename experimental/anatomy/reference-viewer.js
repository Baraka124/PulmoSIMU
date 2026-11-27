/* INF05B — one-use same-origin in-memory handoff. The viewer receives no API credentials. */
(function(root){
"use strict";
function open({client,study,assets,container,status,active}){
 const origin=location.origin,protocol="pulmo.reference/1",id=crypto.randomUUID();
 let cancelled=false,frame=null,listener=null,timer=null,rejectWait=null;
 function close(){cancelled=true;clearTimeout(timer);if(listener)window.removeEventListener("message",listener);if(rejectWait)rejectWait(Error("Viewer closed"));frame?.remove();frame=null;}
 const guard=()=>{if(cancelled||!active()||!client.valid())throw Error("Study opening cancelled");};
 const wait=(expected,send)=>new Promise((resolve,reject)=>{
  rejectWait=reject;
  const done=(error)=>{clearTimeout(timer);window.removeEventListener("message",listener);listener=null;rejectWait=null;error?reject(error):resolve();};
  listener=e=>{if(e.origin!==origin||e.source!==frame?.contentWindow||e.data?.protocol!==protocol||e.data?.nonce!==id)return;
   if(e.data.type==="rejected")done(Error("Source laboratory rejected the package"));else if(e.data.type===expected)done();
  };
  window.addEventListener("message",listener);
  timer=setTimeout(()=>done(Error("Source laboratory did not respond")),120000);
  try{send();}catch(e){done(e);}
 });
 async function run(){
  const U=root.PulmoReferencePackage,inventory=U.plan(study,assets);
  guard();status("Verifying study package…");
  const m=JSON.parse(await (await client.download(study.id,inventory["study-package.json"])).text());guard();U.manifest(study,inventory,m);
  const files={};let n=0;
  for(const path of U.viewerPaths){guard();status("Verifying source assets · "+(++n)+" / "+U.viewerPaths.length);files[path]=await client.download(study.id,inventory[path]);}
  guard();
  frame=document.createElement("iframe");frame.title="Source Anatomy Laboratory";frame.className="reference-frame";frame.referrerPolicy="no-referrer";
  frame.src="./anatomy-inspector.html?referenceEmbed=1#"+id;
  await wait("ready",()=>container.append(frame));guard();
  status("Binding CT, surface and airway topology…");
  await wait("loaded",()=>frame.contentWindow.postMessage({protocol,nonce:id,type:"package",study:{id:study.id,source_kind:study.source_kind,status:study.status},assets,manifest:m,files},origin));
  guard();frame.classList.add("reference-ready");status("Source-linked reference study · review pending");
 }
 const promise=run().catch(e=>{close();throw e;});
 return {promise,close};
}
root.PulmoReferenceViewer=Object.freeze({open});
})(window);
