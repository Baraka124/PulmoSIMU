#!/usr/bin/env python3
"""Publish a checked, approved public reference package to its pre-provisioned draft study.
Run only in a trusted environment. No patient upload service."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import uuid

HOST="jcspwnszghckrsfguvcl.supabase.co"
BUCKET="pulmo-study-assets"
LIMIT=50_000_000
REQUIRED={
"source.nii":("ct_nifti",None),
"airway-mask/mask.raw":("mask","airway_lumen"),
"airway-mask/mask.json":("manifest","airway_lumen"),
"airway-mask/topology.json":("quality_report","airway_lumen"),
"airway-mask/airway-skeleton.json":("airway_graph","airway_lumen"),
"airway-surface/airway.glb":("glb","airway_lumen"),
"airway-surface/reconstruction.json":("manifest","airway_lumen"),
"airway-surface/fidelity.json":("quality_report","airway_lumen")}
OPTIONAL={"source.ingest.json":("manifest",None)}
def sha(data): return hashlib.sha256(data).hexdigest()
def check_package(directory):
    root=Path(directory).resolve(strict=True)
    manifest=root/"study-package.json"
    if manifest.is_symlink() or manifest.stat().st_size>262144: raise ValueError("Invalid package manifest")
    raw=manifest.read_bytes();m=json.loads(raw)
    sid=str(uuid.UUID(m["study_id"]))
    if m.get("schema")!="pulmo.study-package/1" or m.get("status")!="review_pending" or m.get("clinical_navigation_allowed") is not False: raise ValueError("Invalid review gate")
    assets=m["assets"]
    if not isinstance(assets,list) or len(assets) not in (8,9): raise ValueError("Invalid inventory")
    seen=set(); files={}
    for a in assets:
        path=a["path"]; expected=(REQUIRED|OPTIONAL).get(path)
        if not expected or path in seen or (a["kind"],a.get("structure"))!=expected or a.get("version")!=1 or a["object_key"]!=sid+"/v1/"+path: raise ValueError("Invalid asset record")
        p=root/path
        if p.resolve()!=p or not p.is_file() or not 0<p.stat().st_size<=LIMIT: raise ValueError("Invalid or oversized asset")
        b=p.read_bytes()
        if len(b)!=a["bytes"] or sha(b)!=a["sha256"]: raise ValueError("Asset digest mismatch")
        seen.add(path);files[path]=b
    if not REQUIRED.keys()<=seen or sum(map(len,files.values()))>200_000_000: raise ValueError("Incomplete or oversized package")
    if sha(files["source.nii"])!=m["source_sha256"]: raise ValueError("CT identity mismatch")
    mask=json.loads(files["airway-mask/mask.json"]); mesh=json.loads(files["airway-surface/reconstruction.json"])
    if mask["ct_sha256"]!=m["source_sha256"] or mesh["source_sha256"]!=m["source_sha256"] or mask["mask_sha256"]!=sha(files["airway-mask/mask.raw"]) or mesh["mask_sha256"]!=mask["mask_sha256"] or mesh["mesh_sha256"]!=sha(files["airway-surface/airway.glb"]): raise ValueError("Source chain mismatch")
    files["study-package.json"]=raw
    records=assets+[{"path":"study-package.json","kind":"manifest","structure":None,"version":1,"object_key":sid+"/v1/study-package.json","sha256":sha(raw),"bytes":len(raw)}]
    return sid,records,files
class API:
    def __init__(self,key):
        if not key.startswith("sb_secret_"): raise ValueError("Use server-only modern Supabase secret key")
        self.key=key
    def call(self,method,path,body=None,content_type=None,max_bytes=262144,extra=None):
        c=http.client.HTTPSConnection(HOST,timeout=120)
        headers={"apikey":self.key,"Accept":"application/json"}
        if content_type: headers["Content-Type"]=content_type
        if extra: headers.update(extra)
        try:
            c.request(method,path,body=body,headers=headers)
            r=c.getresponse()
            data=r.read(max_bytes+1)
            if len(data)>max_bytes: raise ValueError("Response too large")
            return r.status,data
        finally: c.close()
def publish(api,sid,records,files,provenance):
    code,raw=api.call("GET","/rest/v1/studies?id=eq."+sid+"&select=id,source_kind,status")
    if code!=200 or json.loads(raw)!=[{"id":sid,"source_kind":"public_reference","status":"draft"}]: raise ValueError("Pre-provisioned draft reference study required")
    for a in records:
        key=a["object_key"]; data=files[a["path"]]
        url="/storage/v1/object/"+BUCKET+"/"+key
        # Never overwrite; existing bytes may be retried only if exactly identical.
        code,_=api.call("POST",url,data,"application/octet-stream",extra={"x-upsert":"false"})
        if code not in (200,201,400,409): raise RuntimeError("Storage upload failed: "+str(code))
        code,stored=api.call("GET","/storage/v1/object/authenticated/"+BUCKET+"/"+key,max_bytes=len(data))
        if code!=200 or len(stored)!=len(data) or sha(stored)!=a["sha256"]: raise RuntimeError("Stored bytes failed integrity verification")
    body=json.dumps({"p_study":sid,"p_assets":records,"p_provenance":provenance}).encode()
    code,_=api.call("POST","/rest/v1/rpc/publish_reference_package",body,"application/json")
    if code not in (200,204): raise RuntimeError("Catalogue commit failed: "+str(code))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("package");p.add_argument("--source-url",required=True);p.add_argument("--license",required=True);p.add_argument("--confirm-redistribution",action="store_true");p.add_argument("--publish",action="store_true")
    a=p.parse_args()
    if not a.confirm_redistribution or not a.source_url.startswith("https://") or not 0<len(a.license)<=200: p.error("Explicit approved source and redistribution confirmation required")
    sid,records,files=check_package(a.package)
    if a.publish:
        publish(API(os.environ["SUPABASE_SECRET_KEY"]),sid,records,files,{"source_url":a.source_url,"license":a.license,"redistribution_confirmed":True})
    print(json.dumps({"study_id":sid,"assets":len(records),"published":a.publish,"review":"PENDING"}))
