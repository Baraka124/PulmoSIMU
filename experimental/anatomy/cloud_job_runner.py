"""Explicitly enabled trusted worker, one job per invocation.
Requires INF07 migrations, approved provenance and inference environment.
No HTTP upload or clinical data acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from publish_reference import API
from cloud_job_publication import rpc, publish_job

def run_once(api, provenance, device="cpu", interval=30, timeout=10800):
    if os.environ.get("PULMO_WORKER_ENABLED")!="1":
        raise RuntimeError("Worker not enabled")
    if not isinstance(provenance,dict) or provenance.get("redistribution_confirmed") is not True:
        raise ValueError("Approved study provenance required")
    # Only claim jobs for which this operator has configured explicit provenance.
    approved_study=str(uuid.UUID(provenance["study_id"]))
    rows=rpc(api,"inf07_claim_approved_job",{"p_study":approved_study})
    if not rows:
        return {"status":"idle"}
    if len(rows)!=1:
        raise RuntimeError("Unexpected claim response")
    job=rows[0]
    jid=str(uuid.UUID(job["id"]))
    token=str(uuid.UUID(job["lease_token"]))
    sid=str(uuid.UUID(job["study_id"]))
    if sid!=approved_study:
        raise RuntimeError("Unexpected study claim")
    def beat(stage,progress):
        if rpc(api,"inf07_heartbeat",{"p_job":jid,"p_token":token,"p_stage":stage,"p_progress":progress}) is not True:
            raise RuntimeError("Worker lease lost")
    with tempfile.TemporaryDirectory(prefix="pulmo-job-") as scratch:
        output=Path(scratch)/"package"
        beat("validate",5)
        cmd=[sys.executable,str(Path(__file__).with_name("cloud_job_worker.py")),"--study-id",sid,"--output",str(output),"--device",device]
        # Suppress subprocess output: do not put source identifiers in cloud logs.
        proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        start=time.monotonic()
        try:
            while True:
                beat("segment",10)
                try:
                    code=proc.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic()-start>=timeout:
                        raise TimeoutError("Computation deadline exceeded")
            if code!=0:
                raise RuntimeError("Computational subprocess failed")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        beat("qa",90)
        # Existing adapter verifies every object and completes the database transaction.
        publish_job(api,jid,token,output,provenance)
        return {"status":"review_pending","job_id":jid,"study_id":sid}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provenance",required=True,help="Operator-approved JSON file for exactly one public reference")
    p.add_argument("--device",choices=("cpu","gpu"),default="cpu")
    args=p.parse_args()
    provenance=json.loads(Path(args.provenance).read_text())
    if not str(provenance.get("source_url","")).startswith("https://") or not 0<len(str(provenance.get("license","")))<=200:
        p.error("Approved provenance required")
    print(json.dumps(run_once(API(os.environ["SUPABASE_SECRET_KEY"]),provenance,args.device)))
