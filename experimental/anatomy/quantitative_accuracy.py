"""Research-only comparison of independently annotated, co-registered NIfTI masks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def compare_masks(pred,ref,spacing):
    pred,ref=np.asarray(pred),np.asarray(ref)
    spacing=np.asarray(spacing,dtype=float)
    if pred.ndim!=3 or pred.shape!=ref.shape or min(pred.shape)<1:
        raise ValueError("Matching nonempty 3D grids required")
    if spacing.shape!=(3,) or not np.all(np.isfinite(spacing)) or np.any(spacing<=0):
        raise ValueError("Positive physical voxel spacing required")
    if not np.all((pred==0)|(pred==1)) or not np.all((ref==0)|(ref==1)):
        raise ValueError("Binary masks required")
    pred,ref=pred.astype(bool),ref.astype(bool)
    p,r=int(pred.sum()),int(ref.sum())
    tp=int(np.count_nonzero(pred&ref))
    voxel_ml=float(np.prod(spacing))/1000
    result={"predicted_voxels":p,"reference_voxels":r,"true_positive_voxels":tp,
      "predicted_volume_ml":p*voxel_ml,"reference_volume_ml":r*voxel_ml,
      "signed_volume_error_ml":(p-r)*voxel_ml,
      "absolute_volume_error_ml":abs(p-r)*voxel_ml,
      "relative_volume_error":(p-r)/r if r else None,
      "dice":2*tp/(p+r) if p+r else None,
      "jaccard":tp/(p+r-tp) if p+r else None,
      "sensitivity":tp/r if r else None,"precision":tp/p if p else None,
      "mean_symmetric_surface_distance_mm":None,"hd95_mm":None,
      "max_surface_distance_mm":None,
      "status":"both_empty" if not p and not r else "prediction_empty" if not p else "reference_empty" if not r else "compared"}
    if p and r:
        # Surface samples are centers of 6-connected boundary voxels;
        # pooled distances are voxel-sample weighted, not area weighted.
        structure=ndi.generate_binary_structure(3,1)
        ps=pred&~ndi.binary_erosion(pred,structure=structure,border_value=0)
        rs=ref&~ndi.binary_erosion(ref,structure=structure,border_value=0)
        d1=ndi.distance_transform_edt(~rs,sampling=spacing)[ps]
        d2=ndi.distance_transform_edt(~ps,sampling=spacing)[rs]
        distances=np.concatenate((d1,d2))
        result.update(mean_symmetric_surface_distance_mm=float(distances.mean()),
          hd95_mm=float(np.percentile(distances,95)),
          max_surface_distance_mm=float(distances.max()))
    return result

def evaluate(pred_path,ref_path):
    hashes=[sha256(p) for p in (pred_path,ref_path)]
    pred,ref=[nib.load(str(p)) for p in (pred_path,ref_path)]
    for image in (pred,ref):
        if len(image.shape)!=3 or np.prod(image.shape,dtype=np.int64)>32_000_000:
            raise ValueError("3D mask exceeds evaluation memory budget")
        if image.header.get_xyzt_units()[0]!="mm":
            raise ValueError("Explicit millimetre units required")
        if not np.all(np.isfinite(image.affine)):
            raise ValueError("Finite affine required")
        s,sc=image.get_sform(coded=True);q,qc=image.get_qform(coded=True)
        if not sc and not qc:
            raise ValueError("Explicit spatial transform required")
        if sc and qc and not np.allclose(s,q,rtol=0,atol=1e-4):
            raise ValueError("Conflicting spatial transforms")
    if pred.shape!=ref.shape or not np.allclose(pred.affine,ref.affine,rtol=0,atol=1e-4):
        raise ValueError("Masks must already share a registered grid; no implicit resampling")
    a=ref.affine[:3,:3];spacing=np.linalg.norm(a,axis=0)
    if np.any(spacing<=0) or not np.allclose((a/spacing).T@(a/spacing),np.eye(3),rtol=0,atol=1e-5):
        raise ValueError("Sheared or degenerate grid unsupported")
    result=compare_masks(np.asanyarray(pred.dataobj),np.asanyarray(ref.dataobj),spacing)
    if hashes!=[sha256(p) for p in (pred_path,ref_path)]:
        raise ValueError("Input changed during evaluation")
    return {"schema":"pulmo.mask-accuracy/1","prediction_sha256":hashes[0],"reference_sha256":hashes[1],
      "spacing_mm":spacing.tolist(),"shape":list(ref.shape),"affine_ras_mm":ref.affine.tolist(),
      "surface_convention":"6-connected boundary voxel centers; pooled sample-weighted distances",
      "clinical_navigation_allowed":False,"metrics":result}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction");parser.add_argument("reference");parser.add_argument("output")
    args=parser.parse_args()
    report=evaluate(args.prediction,args.reference)
    with Path(args.output).open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
