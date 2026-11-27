/* INF03: fixed-origin read-only catalogue; credentials in memory only. */
(function(root){
"use strict";
const origin="https://jcspwnszghckrsfguvcl.supabase.co",key="sb_publishable_XOL4Iegkrz0KqhQ4q6m36A_hsKU-QJM";
function create(fetcher=globalThis.fetch.bind(globalThis),now=()=>Date.now()){
 let session=null,generation=0;
 const clear=()=>{generation++;session=null;};
 const valid=()=>!!session&&session.until>now();
 async function request(path,{method="GET",body,token}={}){
  const headers={apikey:key,Accept:"application/json"};
  if(token)headers.Authorization="Bearer "+token;
  if(body!==undefined)headers["Content-Type"]="application/json";
  const res=await fetcher(origin+path,{method,headers,body:body===undefined?undefined:JSON.stringify(body),cache:"no-store",credentials:"omit",redirect:"error",referrerPolicy:"no-referrer",signal:typeof AbortSignal!=="undefined"&&AbortSignal.timeout?AbortSignal.timeout(20000):undefined});
  if(!res.ok){const e=new Error("Request failed ("+res.status+")");e.status=res.status;throw e;}
  if(res.status===204)return null;
  const data=await res.text();return data?JSON.parse(data):null;
 }
 async function signIn(email,password){
  clear();const epoch=generation;
  const data=await request("/auth/v1/token?grant_type=password",{method:"POST",body:{email,password}});
  if(epoch!==generation)throw Error("Session changed");
  if(typeof data?.access_token!=="string"||!Number.isFinite(data.expires_in)||data.expires_in<=0||typeof data.user?.id!=="string")throw Error("Invalid session response");
  session={token:data.access_token,until:now()+Math.min(data.expires_in*1000,3600000)};return true;
 }
 async function read(path){
  if(!valid()){clear();throw Error("Sign in required");}
  const epoch=generation,token=session.token;
  try{const data=await request(path,{token});if(epoch!==generation||!valid())throw Error("Session changed");if(!Array.isArray(data))throw Error("Unexpected catalogue response");return data;}
  catch(e){if(epoch===generation&&(e.status===401||e.status===403||!valid()))clear();throw e;}
 }
 async function signOut(){const token=session?.token;clear();if(token)await request("/auth/v1/logout?scope=local",{method:"POST",token});}
 const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
 function studyPath(table,fields,id){if(typeof id!=="string"||!uuid.test(id))throw Error("Invalid study ID");return "/rest/v1/"+table+"?select="+fields+"&study_id=eq."+id+"&order=created_at.desc&limit=100";}
 async function download(id,asset){
  if(!valid())throw Error("Sign in required");
  if(!uuid.test(id)||asset?.version!==1||typeof asset.object_key!=="string"||!asset.object_key.startsWith(id+"/v1/")||!/^([0-9a-f-]{36})\/v1\/[a-z0-9_./-]+$/i.test(asset.object_key)||asset.object_key.includes("..")||!Number.isSafeInteger(asset.bytes)||asset.bytes<=0||asset.bytes>50000000||!(/^[0-9a-f]{64}$/).test(asset.sha256))throw Error("Invalid asset metadata");
  const epoch=generation,token=session.token;
  const res=await fetcher(origin+"/storage/v1/object/authenticated/pulmo-study-assets/"+asset.object_key,{headers:{apikey:key,Authorization:"Bearer "+token},cache:"no-store",credentials:"omit",redirect:"error",referrerPolicy:"no-referrer",signal:typeof AbortSignal!=="undefined"&&AbortSignal.timeout?AbortSignal.timeout(120000):undefined});
  if(!res.ok){if(epoch===generation&&(res.status===401||res.status===403))clear();throw Error("Asset unavailable");}
  if(!res.body)throw Error("Streaming response required");
  const reader=res.body.getReader(),parts=[];let length=0;
  try{while(true){const {done,value}=await reader.read();if(done)break;length+=value.byteLength;if(length>asset.bytes||epoch!==generation||!valid())throw Error("Asset transfer rejected");parts.push(value);}}
  finally{await reader.cancel().catch(()=>{});}
  if(length!==asset.bytes||epoch!==generation||!valid())throw Error("Asset length or session mismatch");
  const blob=new Blob(parts),bytes=await blob.arrayBuffer(),digest=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),x=>x.toString(16).padStart(2,"0")).join("");
  if(digest!==asset.sha256||epoch!==generation||!valid())throw Error("Asset integrity verification failed");
  return blob;
 }
 return Object.freeze({signIn,signOut,clear,valid,download,list:()=>read("/rest/v1/studies?select=id,title,source_kind,modality,status,created_at&order=created_at.desc&limit=100"),assets:id=>read(studyPath("study_assets","id,kind,structure,version,bytes,sha256,object_key,created_at",id)),jobs:id=>read(studyPath("processing_jobs","id,kind,status,stage,progress,created_at",id))});
}
root.PulmoStudyCatalogue=Object.freeze({create});
})(globalThis);
