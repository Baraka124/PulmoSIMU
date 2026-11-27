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
MULTI = {k:tuple(v) for k,v in {"source.nii":["ct_nifti",None],"pulmonary.glb":["glb",None],"reconstruction.json":["manifest",None],"study-package.json":["manifest",None],"structures/airway_lumen/mask/mask.raw":["mask","airway_lumen"],"structures/airway_lumen/mask/mask.json":["manifest","airway_lumen"],"structures/airway_lumen/mask/topology.json":["quality_report","airway_lumen"],"structures/airway_lumen/surface.glb":["glb","airway_lumen"],"structures/airway_lumen/surface.json":["manifest","airway_lumen"],"structures/pulmonary_artery/mask/mask.raw":["mask","pulmonary_artery"],"structures/pulmonary_artery/mask/mask.json":["manifest","pulmonary_artery"],"structures/pulmonary_artery/mask/topology.json":["quality_report","pulmonary_artery"],"structures/pulmonary_artery/surface.glb":["glb","pulmonary_artery"],"structures/pulmonary_artery/surface.json":["manifest","pulmonary_artery"],"structures/pulmonary_vein/mask/mask.raw":["mask","pulmonary_vein"],"structures/pulmonary_vein/mask/mask.json":["manifest","pulmonary_vein"],"structures/pulmonary_vein/mask/topology.json":["quality_report","pulmonary_vein"],"structures/pulmonary_vein/surface.glb":["glb","pulmonary_vein"],"structures/pulmonary_vein/surface.json":["manifest","pulmonary_vein"],"structures/RUL/mask/mask.raw":["mask","RUL"],"structures/RUL/mask/mask.json":["manifest","RUL"],"structures/RUL/mask/topology.json":["quality_report","RUL"],"structures/RUL/surface.glb":["glb","RUL"],"structures/RUL/surface.json":["manifest","RUL"],"structures/RML/mask/mask.raw":["mask","RML"],"structures/RML/mask/mask.json":["manifest","RML"],"structures/RML/mask/topology.json":["quality_report","RML"],"structures/RML/surface.glb":["glb","RML"],"structures/RML/surface.json":["manifest","RML"],"structures/RLL/mask/mask.raw":["mask","RLL"],"structures/RLL/mask/mask.json":["manifest","RLL"],"structures/RLL/mask/topology.json":["quality_report","RLL"],"structures/RLL/surface.glb":["glb","RLL"],"structures/RLL/surface.json":["manifest","RLL"],"structures/LUL/mask/mask.raw":["mask","LUL"],"structures/LUL/mask/mask.json":["manifest","LUL"],"structures/LUL/mask/topology.json":["quality_report","LUL"],"structures/LUL/surface.glb":["glb","LUL"],"structures/LUL/surface.json":["manifest","LUL"],"structures/LLL/mask/mask.raw":["mask","LLL"],"structures/LLL/mask/mask.json":["manifest","LLL"],"structures/LLL/mask/topology.json":["quality_report","LLL"],"structures/LLL/surface.glb":["glb","LLL"],"structures/LLL/surface.json":["manifest","LLL"],"structures/lung/mask/mask.raw":["mask","lung"],"structures/lung/mask/mask.json":["manifest","lung"],"structures/lung/mask/topology.json":["quality_report","lung"],"structures/lung/surface.glb":["glb","lung"],"structures/lung/surface.json":["manifest","lung"],"structures/airway_lumen/mask/airway-skeleton.json":["airway_graph","airway_lumen"]}.items()}
STRUCTURES = ["airway_lumen","pulmonary_artery","pulmonary_vein","RUL","RML","RLL","LUL","LLL","lung"]
LOBES = ["RUL","RML","RLL","LUL","LLL"]
PARTS = ["mask/mask.raw","mask/mask.json","mask/topology.json","surface.glb","surface.json"]

def check_multi(directory):
    root=Path(directory).resolve(strict=True);p=root/"study-package.json"
    if p.is_symlink() or p.stat().st_size>262144:raise ValueError("Invalid package manifest")
    raw=p.read_bytes();m=json.loads(raw);sid=str(uuid.UUID(m["study_id"]))
    if m.get("schema")!="pulmo.multi-structure-package/1" or m.get("status")!="review_pending" or m.get("clinical_navigation_allowed") is not False:raise ValueError("Invalid package review gate")
    assets=m["assets"];files={};total=0
    if not isinstance(assets,list) or not 8<=len(assets)<=49:raise ValueError("Invalid multi inventory")
    for a in assets:
        path=a["path"];expected=MULTI.get(path)
        if path=="study-package.json" or not expected or path in files or (a["kind"],a.get("structure"))!=expected or a.get("version")!=1 or a["object_key"]!=sid+"/v1/"+path:raise ValueError("Invalid asset record")
        f=root/path
        if f.resolve()!=f or not f.is_file() or not 0<f.stat().st_size<=LIMIT:raise ValueError("Invalid or oversized asset")
        total+=f.stat().st_size
        if total>200_000_000:raise ValueError("Package exceeds transfer budget")
        data=f.read_bytes()
        if len(data)!=a["bytes"] or sha(data)!=a["sha256"]:raise ValueError("Asset digest mismatch")
        files[path]=data
    if not {"source.nii","pulmonary.glb","reconstruction.json"}<=files.keys():raise ValueError("Missing common source")
    ct=sha(files["source.nii"]);meta=json.loads(files["reconstruction.json"])
    if m["source_sha256"]!=ct or meta["source_sha256"]!=ct or meta["mesh_sha256"]!=sha(files["pulmonary.glb"]) or meta.get("validation")!="UNREVIEWED_MULTI_STRUCTURE_SURFACES" or meta["model_from_ras_mm"]!=m["model_from_ras_mm"]:raise ValueError("Combined source identity mismatch")
    seen=[]
    for k in STRUCTURES:
        prefix="structures/"+k+"/";count=sum(prefix+x in files for x in PARTS)
        if not count:continue
        if count!=len(PARTS):raise ValueError("Incomplete structure "+k)
        seen.append(k);mask=json.loads(files[prefix+"mask/mask.json"]);report=json.loads(files[prefix+"surface.json"]);topology=json.loads(files[prefix+"mask/topology.json"])
        masksha=sha(files[prefix+"mask/mask.raw"]);meshsha=sha(files[prefix+"surface.glb"])
        if mask.get("ct_sha256")!=ct or mask.get("structure")!=k or mask.get("mask_sha256")!=masksha or mask.get("review")!="PENDING" or report.get("structure")!=k or report.get("source_sha256")!=ct or report.get("mask_sha256")!=masksha or report.get("mesh_sha256")!=meshsha or report.get("review")!="PENDING" or report.get("clinical_navigation_allowed") is not False or report.get("model_from_ras_mm")!=meta["model_from_ras_mm"] or meta["structure_inventory"].get(k)!=meshsha or topology.get("ct_sha256")!=ct or topology.get("mask_sha256")!=masksha or topology.get("structure")!=k:raise ValueError("Structure source chain mismatch")
    if not seen or set(seen)!=set(m["available_structures"]) or len(seen)!=len(m["available_structures"]) or set(seen)!=set(meta["structure_inventory"]):raise ValueError("Structure inventory mismatch")
    expected=[k for k in seen if k!="lung" or not all(l in seen for l in LOBES)]
    if set(meta["combined_structure_names"])!=set(expected) or len(meta["combined_structure_names"])!=len(expected):raise ValueError("Combined labels mismatch")
    graph="structures/airway_lumen/mask/airway-skeleton.json"
    if graph in files:
        if "airway_lumen" not in seen:raise ValueError("Orphan graph")
        q=json.loads(files[graph]);mask=json.loads(files["structures/airway_lumen/mask/mask.json"])
        if q.get("ct_sha256")!=ct or q.get("mask_sha256")!=mask["mask_sha256"] or q.get("navigation_allowed") is not False:raise ValueError("Graph identity mismatch")
    files["study-package.json"]=raw
    records=assets+[{"path":"study-package.json","kind":"manifest","structure":None,"version":1,"object_key":sid+"/v1/study-package.json","sha256":sha(raw),"bytes":len(raw)}]
    return sid,records,files

def check_package(directory):
    root=Path(directory).resolve(strict=True)
    manifest=root/"study-package.json"
    if manifest.is_symlink() or manifest.stat().st_size>262144: raise ValueError("Invalid package manifest")
    raw=manifest.read_bytes();m=json.loads(raw)
    if m.get("schema")=="pulmo.multi-structure-package/1":return check_multi(directory)
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
    rpc="publish_multi_reference_package" if "pulmonary.glb" in files else "publish_reference_package"
    code,_=api.call("POST","/rest/v1/rpc/"+rpc,body,"application/json")
    if code not in (200,204): raise RuntimeError("Catalogue commit failed: "+str(code))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("package");p.add_argument("--source-url",required=True);p.add_argument("--license",required=True);p.add_argument("--confirm-redistribution",action="store_true");p.add_argument("--publish",action="store_true")
    a=p.parse_args()
    if not a.confirm_redistribution or not a.source_url.startswith("https://") or not 0<len(a.license)<=200: p.error("Explicit approved source and redistribution confirmation required")
    sid,records,files=check_package(a.package)
    if a.publish:
        publish(API(os.environ["SUPABASE_SECRET_KEY"]),sid,records,files,{"source_url":a.source_url,"license":a.license,"redistribution_confirmed":True})
    print(json.dumps({"study_id":sid,"assets":len(records),"published":a.publish,"review":"PENDING"}))
