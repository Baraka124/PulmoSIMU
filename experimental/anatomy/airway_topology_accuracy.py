"""Research metrics for co-registered airway masks; not anatomical branch matching."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

def compare_airways(pred,ref,spacing,tolerance_mm=1.0):
    p,r=np.asarray(pred),np.asarray(ref)
    s=np.asarray(spacing,dtype=float)
    if p.ndim!=3 or p.shape!=r.shape or not p.size or p.size>32_000_000:
        raise ValueError("Matching bounded 3D grids required")
    if s.shape!=(3,) or not np.isfinite(s).all() or np.any(s<=0):
        raise ValueError("Positive physical spacing required")
    if not np.isfinite(tolerance_mm) or tolerance_mm<=0:
        raise ValueError("Positive matching tolerance required")
    if not np.all((p==0)|(p==1)) or not np.all((r==0)|(r==1)):
        raise ValueError("Binary masks required")
    p,r=p.astype(bool),r.astype(bool)
    def describe(mask):
        labels,n=ndi.label(mask,structure=ndi.generate_binary_structure(3,1))
        counts=np.bincount(labels.ravel())[1:]
        del labels
        sk=skeletonize(mask,method="lee").astype(bool)
        pts=np.argwhere(sk)
        if len(pts)>100000:
            raise ValueError("Centerline sample budget exceeded")
        # EDT radius to nearest background voxel center, not a fitted cross section.
        radii=ndi.distance_transform_edt(np.pad(mask,1),sampling=s)[tuple((pts+1).T)] if len(pts) else np.array([])
        return pts*s,radii,{"components":int(n),"largest_component_fraction":float(counts.max()/counts.sum()) if n else None,"centerline_samples":len(pts)}
    pp,pr,pinfo=describe(p)
    rp,rr,rinfo=describe(r)
    result={"prediction":pinfo,"reference":rinfo,"tolerance_mm":float(tolerance_mm),
      "reference_centerline_coverage":None,"predicted_centerline_precision":None,
      "matched_radius_mean_absolute_error_mm":None,"matched_radius_signed_bias_mm":None,
      "matched_reference_samples":0,"clinical_navigation_allowed":False,
      "convention":"Unweighted Lee skeleton voxel centers; nearest neighbor matching in physical mm; EDT radius proxy; not branch identification"}
    if len(rp):
        result["reference_centerline_coverage"]=0.0
    if len(pp):
        result["predicted_centerline_precision"]=0.0
    if len(pp) and len(rp):
        dr,idx=cKDTree(pp).query(rp)
        dp,_=cKDTree(rp).query(pp)
        matched=dr<=tolerance_mm
        result["reference_centerline_coverage"]=float(matched.mean())
        result["predicted_centerline_precision"]=float((dp<=tolerance_mm).mean())
        result["matched_reference_samples"]=int(matched.sum())
        if matched.any():
            delta=pr[idx[matched]]-rr[matched]
            result["matched_radius_mean_absolute_error_mm"]=float(np.abs(delta).mean())
            result["matched_radius_signed_bias_mm"]=float(delta.mean())
    return result
