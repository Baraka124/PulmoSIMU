#!/usr/bin/env python3
"""INF04 local trusted worker. No network API, database writes or clinical authorization."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
import numpy as np
import nibabel as nib
from dicom_ingest import convert
from segmentation_pipeline import prepare, sha, write_json
from anatomy_fidelity import run as reconstruct

MAX_BYTES = 750_000_000
MAX_VOXELS = 150_000_000
MODEL_VERSION = "2.13.0"
MODEL_TASK = "lung_vessels"

def validate_ct(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("CT exceeds worker byte limit")
    ct = nib.load(str(path))
    if len(ct.shape) != 3 or min(ct.shape) < 2 or math.prod(ct.shape) > MAX_VOXELS:
        raise ValueError("Invalid CT dimensions or worker voxel budget")
    a, code = ct.header.get_sform(coded=True)
    if code <= 0 or ct.header.get_xyzt_units()[0] != "mm" or not np.isfinite(a).all():
        raise ValueError("Explicit mm sform required")
    sp = np.linalg.norm(a[:3,:3], axis=0)
    if not np.all((sp >= .001) & (sp <= 100)) or not np.allclose((a[:3,:3]/sp).T @ (a[:3,:3]/sp),np.eye(3),atol=1e-5):
        raise ValueError("Orthogonal physical CT grid required")
    if not np.isfinite([ct.dataobj.slope,ct.dataobj.inter]).all():
        raise ValueError("Invalid CT calibration")
    return ct

def infer(ct, scratch, device):
    if importlib.metadata.version("TotalSegmentator") != MODEL_VERSION:
        raise ValueError("Install the pinned TotalSegmentator provider")
    exe = shutil.which("TotalSegmentator")
    if not exe:
        raise ValueError("TotalSegmentator CLI unavailable")
    scratch.mkdir()
    # No threshold fallback. Do not silently merge wall, artery or vein labels.
    subprocess.run([exe,"-i",str(ct),"-o",str(scratch),"-ta",MODEL_TASK,"--device",device],
                   check=True,timeout=3600)
    mask=scratch/"lung_airways.nii.gz"
    if not mask.is_file():
        raise ValueError("Expected model airway-lumen output missing")
    return mask

def inventory(root, entries, study_id):
    assets=[]
    for rel,kind,structure in entries:
        path=root/rel
        if not path.is_file() or path.is_symlink() or path.stat().st_size<=0:
            raise ValueError("Missing generated asset: "+rel)
        assets.append({"kind":kind,"structure":structure,"version":1,
            "path":rel,"object_key":str(study_id)+"/v1/"+rel,
            "sha256":sha(path),"bytes":path.stat().st_size})
    return assets

def build(source,out,study_id,*,mask=None,engine=None,engine_version=None,device="cpu",progress=lambda stage:None):
    study_id=uuid.UUID(str(study_id))
    source=Path(source).resolve(strict=True)
    out=Path(out).resolve()
    if out.exists(): raise ValueError("Choose a new study output directory")
    if source.is_dir() and (source==out or source in out.parents):
        raise ValueError("Output must be outside DICOM input")
    if mask is not None:
        mask=Path(mask).resolve(strict=True)
        if not mask.is_file() or mask.stat().st_size>MAX_BYTES:
            raise ValueError("Invalid supplied mask")
        if not engine or not engine_version or len(engine)>120 or len(engine_version)>120:
            raise ValueError("Explicit mask provider and version required")
    elif engine or engine_version:
        raise ValueError("Engine claims only apply to supplied masks")
    if device not in ("cpu","gpu"): raise ValueError("Choose cpu or gpu")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pulmo-",dir=out.parent) as work:
        stage=Path(work)/"package";stage.mkdir()
        ct=stage/"source.nii"
        progress("ingest")
        entries=[("source.nii","ct_nifti",None)]
        if source.is_dir():
            # Trusted local input only; bounded directory walk before DICOM decoding.
            count=0;total=0
            for p in source.rglob("*"):
                if p.is_symlink(): raise ValueError("DICOM symlinks unsupported")
                if p.is_file():
                    count+=1;total+=p.stat().st_size
                    if count>4096 or total>2_000_000_000: raise ValueError("DICOM input budget exceeded")
            convert(source,ct)
            entries.append(("source.ingest.json","manifest",None))
        elif source.is_file() and source.suffix.lower()==".nii":
            validate_ct(source)
            before=sha(source);shutil.copyfile(source,ct)
            if sha(ct)!=before: raise ValueError("CT changed during copy")
        else: raise ValueError("Supply a classic DICOM directory or uncompressed NIfTI .nii")
        validate_ct(ct); ct_sha=sha(ct)
        progress("segment")
        if mask is None:
            mask=infer(ct,Path(work)/"inference",device)
            engine="TotalSegmentator:"+MODEL_TASK
            engine_version=MODEL_VERSION
            mode="model_inference"
        else: mode="supplied_binary_mask"
        mask_before=sha(mask)
        progress("topology")
        prepare(ct,mask,"airway_lumen",engine,engine_version,stage/"airway-mask")
        if sha(mask)!=mask_before or sha(ct)!=ct_sha:
            raise ValueError("Input changed during segmentation preparation")
        entries.extend([("airway-mask/mask.raw","mask","airway_lumen"),("airway-mask/mask.json","manifest","airway_lumen"),("airway-mask/topology.json","quality_report","airway_lumen")])
        if (stage/"airway-mask/airway-skeleton.json").is_file():
            entries.append(("airway-mask/airway-skeleton.json","airway_graph","airway_lumen"))
        else: raise ValueError("Airway topology missing; study not publishable")
        progress("reconstruct")
        reconstruct(ct,stage/"airway-mask",stage/"airway-surface")
        entries.extend([("airway-surface/airway.glb","glb","airway_lumen"),("airway-surface/reconstruction.json","manifest","airway_lumen"),("airway-surface/fidelity.json","quality_report","airway_lumen")])
        progress("package")
        assets=inventory(stage,entries,study_id)
        if assets[0]["sha256"]!=ct_sha: raise ValueError("CT identity changed")
        manifest={"schema":"pulmo.study-package/1","study_id":str(study_id),"status":"review_pending",
                  "source_sha256":ct_sha,"input_mask_sha256":mask_before,"segmentation":{"mode":mode,"engine":engine,"version":engine_version},
                  "clinical_navigation_allowed":False,"assets":assets}
        write_json(stage/"study-package.json",manifest)
        if out.exists(): raise ValueError("Output appeared during processing")
        stage.rename(out)
    progress("review_pending")
    return manifest

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source");parser.add_argument("out");parser.add_argument("--study-id",required=True)
    parser.add_argument("--mask");parser.add_argument("--engine");parser.add_argument("--engine-version")
    parser.add_argument("--device",choices=("cpu","gpu"),default="cpu")
    a=parser.parse_args()
    build(a.source,a.out,a.study_id,mask=a.mask,engine=a.engine,engine_version=a.engine_version,device=a.device,
          progress=lambda stage:print(json.dumps({"stage":stage}),flush=True))
