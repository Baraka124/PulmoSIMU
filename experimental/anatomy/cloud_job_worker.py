"""INF07B-C: trusted worker preparation; no queue claim or publication yet."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from urllib.parse import quote

from publish_reference import HOST, BUCKET, API

MAX_SOURCE_BYTES = 750_000_000
CHUNK = 1024 * 1024

def source_record(api, study_id):
    sid = str(uuid.UUID(str(study_id)))
    code, raw = api.call("GET", "/rest/v1/studies?id=eq."+sid+"&select=id,source_kind,status")
    if code != 200 or json.loads(raw) != [{"id":sid,"source_kind":"public_reference","status":"ingested"}]:
        raise ValueError("Ingested public reference required")
    code, raw = api.call("GET", "/rest/v1/study_assets?study_id=eq."+sid+"&kind=eq.ct_nifti&select=object_key,bytes,sha256")
    if code != 200:
        raise RuntimeError("Source lookup failed")
    rows = json.loads(raw)
    if len(rows) != 1:
        raise ValueError("Exactly one registered CT source required")
    return sid, rows[0]

def download_source(key, sid, record, destination, connection_factory=http.client.HTTPSConnection):
    sid = str(uuid.UUID(str(sid)))
    path = record["object_key"]
    size = record["bytes"]
    digest = record["sha256"]
    if not key.startswith("sb_secret_"):
        raise ValueError("Server-only secret required")
    if not isinstance(path,str) or not re.fullmatch(re.escape(sid)+r"/[A-Za-z0-9_/-]+\.nii",path) or "//" in path:
        raise ValueError("Invalid source key")
    if type(size) is not int or not 0 < size <= MAX_SOURCE_BYTES or not re.fullmatch(r"[0-9a-f]{64}",digest):
        raise ValueError("Invalid source metadata")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    conn = connection_factory(HOST,timeout=120)
    temp = None
    try:
        conn.request("GET","/storage/v1/object/authenticated/"+BUCKET+"/"+quote(path,safe="/"),headers={"apikey":key,"Accept-Encoding":"identity"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError("Source transfer failed")
        length = response.getheader("Content-Length")
        if length is not None and int(length) != size:
            raise ValueError("Source length mismatch")
        h = hashlib.sha256()
        count = 0
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as f:
            temp = Path(f.name)
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                count += len(block)
                if count > size:
                    raise ValueError("Source exceeds registered size")
                h.update(block)
                f.write(block)
            if count != size or h.hexdigest() != digest:
                raise ValueError("Source integrity mismatch")
        os.link(temp,destination)
        return destination
    finally:
        conn.close()
        if temp is not None:
            temp.unlink(missing_ok=True)

def prepare_package(study_id, output, *, device="cpu"):
    """Run existing inference/reconstruction on a trusted registered source.
    Output stays local until a lease-aware publication contract is implemented.
    """
    from multistructure_worker import build
    key = os.environ["SUPABASE_SECRET_KEY"]
    sid, record = source_record(API(key),study_id)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pulmo-source-",dir=output.parent) as scratch:
        source = download_source(key,sid,record,Path(scratch)/"source.nii")
        return build(source,output,sid,device=device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",choices=("cpu","gpu"),default="cpu")
    args = parser.parse_args()
    result = prepare_package(args.study_id,args.output,device=args.device)
    print(json.dumps({"study_id":result["study_id"],"status":result["status"],"available_structures":result["available_structures"],"published":False}))
