"""Trusted first-package publication adapter; retains INF06 transfer budgets."""
import json
import uuid
from publish_reference import BUCKET, check_multi, sha
from cloud_job_worker import source_record

def rpc(api,name,payload):
    code, raw = api.call("POST","/rest/v1/rpc/"+name,json.dumps(payload).encode(),"application/json")
    if code not in (200,204):
        raise RuntimeError("RPC failed: "+name+" ("+str(code)+")")
    return json.loads(raw) if raw else None

def publish_job(api, job_id, token, package, provenance):
    job_id, token = str(uuid.UUID(str(job_id))), str(uuid.UUID(str(token)))
    sid, records, files = check_multi(package)
    if not isinstance(provenance,dict) or provenance.get("redistribution_confirmed") is not True or not str(provenance.get("source_url","")).startswith("https://") or not 0<len(str(provenance.get("license","")))<=200:
        raise ValueError("Approved provenance required")
    code,raw=api.call("GET","/rest/v1/processing_jobs?id=eq."+job_id+"&select=study_id,status,lease_token")
    if code!=200 or json.loads(raw)!=[{"study_id":sid,"status":"running","lease_token":token}]:
        raise ValueError("Job does not own package")
    _,source=source_record(api,sid)
    original=next(a for a in records if a["path"]=="source.nii")
    if any(source[k]!=original[k] for k in ("object_key","bytes","sha256")):
        raise ValueError("Registered source differs from package")
    def heartbeat():
        if rpc(api,"inf07_heartbeat",{"p_job":job_id,"p_token":token,"p_stage":"publish","p_progress":99}) is not True:
            raise RuntimeError("Publishing lease lost")
    heartbeat()
    for a in records:
        heartbeat()
        url="/storage/v1/object/"+BUCKET+"/"+a["object_key"]
        data=files[a["path"]]
        if a["path"]!="source.nii":
            code,_=api.call("POST",url,data,"application/octet-stream",extra={"x-upsert":"false"})
            if code not in (200,201,400,409):
                raise RuntimeError("Asset upload failed")
        code,stored=api.call("GET","/storage/v1/object/authenticated/"+BUCKET+"/"+a["object_key"],max_bytes=len(data))
        if code!=200 or len(stored)!=len(data) or sha(stored)!=a["sha256"]:
            raise RuntimeError("Stored asset integrity mismatch")
    heartbeat()
    rpc(api,"inf07_publish_job",{"p_job":job_id,"p_token":token,"p_assets":records,"p_provenance":provenance})
    return sid
