#!/usr/bin/env python3
"""K01–K02: offline paired CT registration and explicit RAS displacement export.
Python 3.11+, numpy, nibabel, SimpleITK. Only authorized local CTs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def load(path):
    img=nib.load(str(path))
    if len(img.shape)!=3 or min(img.shape)<2 or np.prod(img.shape)>100_000_000:
        raise ValueError("Invalid or oversized 3D CT")
    a,code=img.header.get_sform(coded=True)
    if code<=0 or img.header.get_xyzt_units()[0]!="mm" or not np.isfinite(a).all():
        raise ValueError("Explicit mm sform required")
    lps=np.diag([-1.,-1.,1.,1.])@a
    spacing=np.linalg.norm(lps[:3,:3],axis=0)
    if np.any(spacing<=0):
        raise ValueError("Singular grid")
    direction=lps[:3,:3]/spacing
    if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5,rtol=0):
        raise ValueError("Sheared grid not supported")
    pixels=img.get_fdata(dtype=np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError("Nonfinite CT values")
    image=sitk.GetImageFromArray(np.clip(pixels,-1000,400).transpose(2,1,0))
    image.SetSpacing(tuple(spacing))
    image.SetDirection(tuple(direction.ravel()))
    image.SetOrigin(tuple(lps[:3,3]))
    return img,a,image

def register(fixed_path,moving_path,out,stride=4,iterations=50):
    if not 1<=stride<=16 or not 1<=iterations<=200:
        raise ValueError("Invalid registration settings")
    out=Path(out)
    if out.exists():
        raise ValueError("Choose new output directory")
    fs,ms=sha(fixed_path),sha(moving_path)
    if fs==ms:
        raise ValueError("Distinct inspiration and expiration CT required")
    fixed,fa,fi=load(fixed_path)
    moving,ma,mi=load(moving_path)
    # Pull moving image onto fixed physical grid, then estimate displacement
    # in LPS mm: x_moving = x_fixed + u(x_fixed).
    mr=sitk.Resample(mi,fi,sitk.Transform(3,sitk.sitkIdentity),sitk.sitkLinear,-1000.0,sitk.sitkFloat32)
    # Coarse-to-fine is future work; single-resolution Demons is a baseline.
    dem=sitk.DiffeomorphicDemonsRegistrationFilter()
    dem.SetNumberOfIterations(iterations)
    dem.SetStandardDeviations(1.5)
    field=dem.Execute(fi,mr)
    # SimpleITK vector array z,y,x,component; convert to fixed-grid RAS mm.
    arr=sitk.GetArrayFromImage(field)
    jac=sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field))
    if not np.isfinite(arr).all() or not np.isfinite(jac).all():
        raise ValueError("Nonfinite registration output")
    sampled=arr[::stride,::stride,::stride,:].copy()
    sampled[...,:2]*=-1
    dims=list(sampled.shape[:3][::-1])
    if min(dims)<2 or np.prod(dims)>10_000_000 or np.max(np.abs(sampled))>200:
        raise ValueError("Sampled grid outside browser budget")
    raw=sampled.astype("<f4").tobytes(order="C")
    manifest={"schema":"pulmosimu.respiratory-field/1","fixed_ct_sha256":fs,"moving_ct_sha256":ms,
              "fixed_phase":"inspiration","moving_phase":"expiration",
              "direction":"fixed_RAS_to_moving_RAS_mm","encoding":"float32le-xyz-interleaved-x-fast",
              "fixed_voxel_to_ras_mm":fa.tolist(),"moving_voxel_to_ras_mm":ma.tolist(),
              "dimensions":dims,"stride":stride,"field_sha256":hashlib.sha256(raw).hexdigest(),
              "method":"single_resolution_diffeomorphic_demons","iterations":iterations,
              "registration_metric":float(dem.GetMetric()),"dense_jacobian_min":float(jac.min()),
              "dense_nonpositive_jacobian_fraction":float(np.mean(jac<=0)),
              "review":"PENDING","same_patient_verified":False,"clinical_navigation_allowed":False}
    if not np.isfinite(manifest["registration_metric"]):
        raise ValueError("Nonfinite metric")
    out.mkdir(parents=True)
    (out/"field.raw").write_bytes(raw)
    (out/"field.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inspiration");p.add_argument("expiration");p.add_argument("out")
    p.add_argument("--stride",type=int,default=4);p.add_argument("--iterations",type=int,default=50)
    a=p.parse_args()
    print(json.dumps(register(a.inspiration,a.expiration,a.out,a.stride,a.iterations),indent=2))
