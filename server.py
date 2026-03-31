"""
PNEUMO·VR — Pipeline Server
============================
FastAPI server that accepts a DICOM zip upload, runs the segmentation 
pipeline, and serves the resulting GLB + JSON back to the browser.

No Python required on the planning workstation — just upload and load.
 
Usage:
    pip install fastapi uvicorn python-multipart
    python server.py

Endpoints:
    POST /process           — upload DICOM zip → returns job_id
    GET  /status/{job_id}   — poll job status
    GET  /result/{job_id}/glb  — download airway GLB
    GET  /result/{job_id}/json — download meta JSON
    GET  /                  — health check
    GET  /docs              — auto Swagger UI

Frontend integration:
    Load XvR3.html and set PNEUMOVR_SERVER = 'http://localhost:8000'
    The LOAD FROM SERVER button handles the rest.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── Config ────────────────────────────────────────────────────────
WORK_DIR   = Path(os.getenv("PNEUMOVR_WORKDIR", "./pneumovr_jobs"))
PIPELINE   = Path(os.path.dirname(__file__)) / "pipeline.py"
PYTHON     = sys.executable
MAX_JOBS   = int(os.getenv("PNEUMOVR_MAX_JOBS", "10"))

WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────
# {job_id: {status, created, patient_id, glb_path, json_path, error, log}}
jobs: dict = {}


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="PNEUMO·VR Pipeline Server",
    description="DICOM → airway GLB + meta JSON for procedure planning",
    version="1.1",
)

# Origins: GitHub Pages URL + localhost for dev
# Set ALLOWED_ORIGIN env var on Railway to lock down to your Pages URL
_allowed = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed] if _allowed != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "PNEUMO·VR Pipeline Server",
        "version": "1.1",
        "jobs_active": sum(1 for j in jobs.values() if j["status"] == "running"),
        "jobs_total": len(jobs),
        "pipeline": str(PIPELINE),
        "pipeline_exists": PIPELINE.exists(),
    }


# ── Upload + process ──────────────────────────────────────────────
@app.post("/process")
async def process_dicom(
    file: UploadFile = File(..., description="ZIP archive of DICOM files"),
    patient_id: Optional[str] = Form(None, description="Patient ID prefix (auto-derived if omitted)"),
):
    """
    Accept a ZIP of DICOM slices, extract, run pipeline.py in background.
    Returns job_id immediately — poll /status/{job_id} for completion.
    """
    # Validate upload
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".zip",):
        raise HTTPException(400, f"Expected a .zip file, got '{ext}'. Zip your DICOM folder first.")

    # Prune old completed jobs to stay under MAX_JOBS
    _prune_jobs()

    # Create job
    job_id    = str(uuid.uuid4())[:8]
    job_dir   = WORK_DIR / job_id
    dicom_dir = job_dir / "dicom"
    out_dir   = job_dir / "output"
    dicom_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    pid = (patient_id or Path(file.filename).stem or job_id).replace(" ", "_")

    jobs[job_id] = {
        "status":     "uploading",
        "created":    time.time(),
        "patient_id": pid,
        "job_dir":    str(job_dir),
        "glb_path":   None,
        "json_path":  None,
        "error":      None,
        "log":        [],
    }

    # Save and extract zip
    zip_path = job_dir / "upload.zip"
    try:
        raw = await file.read()
        with open(zip_path, "wb") as f:
            f.write(raw)
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = f"Upload write failed: {e}"
        raise HTTPException(500, str(e))

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dicom_dir)
    except zipfile.BadZipFile:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = "File is not a valid ZIP archive"
        raise HTTPException(400, "Invalid ZIP file")

    # Find the actual DICOM folder (may be nested)
    dicom_actual = _find_dicom_root(dicom_dir)
    if not dicom_actual:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = "No DICOM files found inside ZIP"
        raise HTTPException(400, "No DICOM (.dcm) files found inside the ZIP")

    jobs[job_id]["status"] = "queued"

    # Launch pipeline in background
    asyncio.create_task(_run_pipeline(job_id, dicom_actual, out_dir, pid))

    return {
        "job_id":     job_id,
        "patient_id": pid,
        "status":     "queued",
        "poll_url":   f"/status/{job_id}",
    }


# ── Status ────────────────────────────────────────────────────────
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = _get_job(job_id)
    resp = {
        "job_id":     job_id,
        "patient_id": job["patient_id"],
        "status":     job["status"],         # queued | running | done | failed
        "elapsed_s":  round(time.time() - job["created"], 1),
        "log":        job["log"][-20:],      # last 20 log lines
    }
    if job["status"] == "done":
        resp["result_glb"]  = f"/result/{job_id}/glb"
        resp["result_json"] = f"/result/{job_id}/json"
    if job["status"] == "failed":
        resp["error"] = job["error"]
    return resp


# ── Download results ──────────────────────────────────────────────
@app.get("/result/{job_id}/glb")
async def get_glb(job_id: str):
    job = _get_job(job_id)
    if job["status"] != "done" or not job["glb_path"]:
        raise HTTPException(404, f"GLB not ready — job status: {job['status']}")
    return FileResponse(
        job["glb_path"],
        media_type="model/gltf-binary",
        filename=f"{job['patient_id']}_airway.glb",
    )


@app.get("/result/{job_id}/json")
async def get_json(job_id: str):
    job = _get_job(job_id)
    if job["status"] != "done" or not job["json_path"]:
        raise HTTPException(404, f"JSON not ready — job status: {job['status']}")
    return FileResponse(
        job["json_path"],
        media_type="application/json",
        filename=f"{job['patient_id']}_meta.json",
    )


# ── List jobs ─────────────────────────────────────────────────────
@app.get("/jobs")
async def list_jobs():
    return [
        {
            "job_id":     jid,
            "patient_id": j["patient_id"],
            "status":     j["status"],
            "elapsed_s":  round(time.time() - j["created"], 1),
        }
        for jid, j in sorted(jobs.items(), key=lambda x: -x[1]["created"])
    ]


# ── Synthetic test endpoint ───────────────────────────────────────
@app.post("/synthetic")
async def generate_synthetic(patient_id: str = Form("SYNTHETIC_001")):
    """
    Generate a synthetic test patient without any DICOM data.
    Useful for verifying the frontend integration without real CT.
    """
    synthetic_py = Path(os.path.dirname(__file__)) / "synthetic.py"
    if not synthetic_py.exists():
        raise HTTPException(404, "synthetic.py not found alongside server.py")

    job_id  = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True)

    pid = patient_id.replace(" ", "_")
    jobs[job_id] = {
        "status":     "running",
        "created":    time.time(),
        "patient_id": pid,
        "job_dir":    str(job_dir),
        "glb_path":   None,
        "json_path":  None,
        "error":      None,
        "log":        ["Generating synthetic patient..."],
    }

    asyncio.create_task(_run_synthetic(job_id, synthetic_py, out_dir, pid))
    return {"job_id": job_id, "patient_id": pid, "status": "running", "poll_url": f"/status/{job_id}"}


# ── Internal helpers ──────────────────────────────────────────────
def _get_job(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return jobs[job_id]


def _find_dicom_root(base: Path) -> Optional[Path]:
    """Find the deepest folder containing .dcm files."""
    for root, dirs, files in os.walk(base):
        dcm = [f for f in files if f.lower().endswith(".dcm") or
               (not "." in f and os.path.getsize(os.path.join(root, f)) > 128)]
        if dcm:
            return Path(root)
    return None


def _prune_jobs():
    """Remove oldest completed/failed jobs when over MAX_JOBS."""
    done = [(jid, j) for jid, j in jobs.items() if j["status"] in ("done", "failed")]
    if len(jobs) >= MAX_JOBS and done:
        oldest_id, oldest = sorted(done, key=lambda x: x[1]["created"])[0]
        try:
            shutil.rmtree(oldest["job_dir"], ignore_errors=True)
        except Exception:
            pass
        del jobs[oldest_id]


async def _run_pipeline(job_id: str, dicom_dir: Path, out_dir: Path, patient_id: str):
    """Run pipeline.py as a subprocess and stream its stdout to job log."""
    job = jobs[job_id]
    job["status"] = "running"

    cmd = [
        PYTHON, str(PIPELINE),
        "--dicom",      str(dicom_dir),
        "--out",        str(out_dir),
        "--patient-id", patient_id,
    ]
    job["log"].append(f"$ {' '.join(cmd)}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()
            job["log"].append(text)
            # Keep log bounded
            if len(job["log"]) > 500:
                job["log"] = job["log"][-400:]

        await proc.wait()

        if proc.returncode != 0:
            job["status"] = "failed"
            job["error"]  = f"pipeline.py exited with code {proc.returncode}"
            return

        # Find outputs
        glb  = out_dir / f"{patient_id}_airway.glb"
        meta = out_dir / f"{patient_id}_meta.json"

        if not glb.exists() or not meta.exists():
            job["status"] = "failed"
            job["error"]  = "Pipeline finished but output files not found"
            return

        job["glb_path"]  = str(glb)
        job["json_path"] = str(meta)
        job["status"]    = "done"
        job["log"].append(f"✓ Done — {glb.stat().st_size//1024}KB GLB, {meta.stat().st_size//1024}KB JSON")

    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)
        job["log"].append(f"ERROR: {e}")


async def _run_synthetic(job_id: str, synthetic_py: Path, out_dir: Path, patient_id: str):
    """Run synthetic.py and record outputs."""
    job = jobs[job_id]
    cmd = [PYTHON, str(synthetic_py), "--out", str(out_dir), "--patient-id", patient_id]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            job["log"].append(line.decode("utf-8", errors="replace").rstrip())
        await proc.wait()

        glb  = out_dir / f"{patient_id}_airway.glb"
        meta = out_dir / f"{patient_id}_meta.json"
        if glb.exists() and meta.exists():
            job["glb_path"]  = str(glb)
            job["json_path"] = str(meta)
            job["status"]    = "done"
        else:
            job["status"] = "failed"
            job["error"]  = "synthetic.py finished but output files missing"
    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import socket

    port = int(os.getenv("PORT", 8001))

    print(f"\n══ PNEUMO·VR Pipeline Server ══════════════════════")
    print(f"   http://localhost:{port}")
    print(f"   Docs: http://localhost:{port}/docs")
    print(f"   Work dir: {WORK_DIR.resolve()}")
    print(f"   Pipeline: {PIPELINE}")
    print(f"══════════════════════════════════════════════════")
    print(f"\n   Set the URL in XvR3.html sidebar to: http://localhost:{port}\n")

    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
