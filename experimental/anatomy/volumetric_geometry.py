"""Bounded local implicit geometry; not an exact subvoxel distance or medial axis."""
import numpy as np
from scipy import ndimage as ndi

MAX_VOXELS=2_000_000

def fields(mask,affine,*,sigma_mm=1.,band_mm=1.5):
    m=np.asarray(mask);a=np.asarray(affine,dtype=float)
    if m.ndim!=3 or min(m.shape)<5 or m.size>MAX_VOXELS or not np.all((m==0)|(m==1)) or not m.any() or np.all(m):
        raise ValueError("Bounded nontrivial binary ROI required")
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1]):
        raise ValueError("Invalid affine")
    s=np.linalg.norm(a[:3,:3],axis=0)
    if np.any(s<.1) or np.any(s>10) or not np.allclose((a[:3,:3]/s).T@(a[:3,:3]/s),np.eye(3),rtol=0,atol=1e-5):
        raise ValueError("Orthogonal mm grid required")
    if not np.isfinite([sigma_mm,band_mm]).all() or not .1<=sigma_mm<=5 or not .1<=band_mm<=10 or sigma_mm/min(s)>30:
        raise ValueError("Invalid smoothing or band parameters")
    m=m.astype(bool)
    inside=ndi.distance_transform_edt(np.pad(m,1),sampling=s)[1:-1,1:-1,1:-1]
    outside=ndi.distance_transform_edt(~m,sampling=s)
    phi=(outside-inside).astype(np.float32)
    smooth=ndi.gaussian_filter(phi,sigma_mm/s,mode="nearest")
    # Derivatives in orthonormal voxel-axis physical coordinates. Curvature
    # invariants do not change under an orthogonal RAS basis transformation.
    grad=np.stack(np.gradient(smooth,*s,edge_order=2),axis=-1)
    norm=np.linalg.norm(grad,axis=-1)
    valid=(np.abs(smooth)<=band_mm)&(norm>=.25)
    # Exclude the artificial padded boundary and smoothing/derivative halo.
    for axis,h in enumerate(s):
        margin=int(np.ceil(4*sigma_mm/h))+2
        l=[slice(None)]*3;l[axis]=slice(0,margin);valid[tuple(l)]=False
        l[axis]=slice(-margin,None);valid[tuple(l)]=False
    hess=np.empty(m.shape+(3,3),dtype=np.float32)
    for i in range(3):
        for j in range(i,3):
            d=.5*(np.gradient(grad[...,i],s[j],axis=j,edge_order=2)+np.gradient(grad[...,j],s[i],axis=i,edge_order=2))
            hess[...,i,j]=d;hess[...,j,i]=d
    # For positive-outside implicit phi, shape operator projected on tangent
    # plane is P H P / |grad phi|. Compute eigenvalues in an explicit tangent basis.
    g=grad[valid];n=g/norm[valid,None]
    seed=np.eye(3)[np.argmin(np.abs(n),axis=1)]
    t=np.cross(n,seed);t/=np.linalg.norm(t,axis=1,keepdims=True)
    v=np.cross(n,t);basis=np.stack((t,v),axis=-1)
    op=np.swapaxes(basis,-1,-2)@hess[valid]@basis/norm[valid,None,None]
    k=np.linalg.eigvalsh(op)
    def scatter(x):
        q=np.zeros(m.shape,dtype=np.float32);q[valid]=x;return q
    result={"signed-distance-mm":phi,"smoothed-implicit-field-mm":smooth,
       "interior-clearance-proxy-mm":inside.astype(np.float32),"curvature-valid-mask":valid.astype(np.uint8),
       "principal-curvature-min":scatter(k[:,0]),"principal-curvature-max":scatter(k[:,1]),
       "mean-curvature":scatter(k.mean(axis=1)),"gaussian-curvature":scatter(k[:,0]*k[:,1])}
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError("Nonfinite geometry field")
    report={"schema":"pulmo.volumetric-geometry/1","sigma_mm":sigma_mm,"band_mm":band_mm,
       "valid_curvature_voxels":int(valid.sum()),"spacing_mm":s.tolist(),
       "curvature_units":{"principal":"1/mm","mean":"1/mm","gaussian":"1/mm^2"},
       "distance_convention":"negative inside, positive outside; voxel-center EDT; artificial padded exterior",
       "curvature_convention":"positive on convex outward-oriented surfaces; estimated in smoothed implicit narrow band",
       "zero_fill":"curvature invalid outside curvature-valid-mask; interior clearance zero outside foreground",
       "limitations":["Not exact subvoxel interface distance","Smoothed field changes geometry and curvature","No certified medial axis or collision clearance","ROI caps are not observed anatomy"],
       "review":"PENDING","clinical_navigation_allowed":False}
    return result,report

def skeleton_distance(mask,affine,graph):
    """Distance to the sampled Lee skeleton, not an exact medial axis."""
    m=np.asarray(mask);s=np.linalg.norm(np.asarray(affine)[:3,:3],axis=0)
    pts=np.asarray(graph["nodes_voxel_ijk"])
    if pts.ndim!=2 or pts.shape[1]!=3 or not len(pts) or not np.issubdtype(pts.dtype,np.integer) or np.any(pts<0) or np.any(pts>=np.asarray(m.shape)) or np.any(m[tuple(pts.T)]!=1):
        raise ValueError("Graph nodes not supported by mask")
    sk=np.zeros(m.shape,dtype=bool);sk[tuple(pts.T)]=True
    return ndi.distance_transform_edt(~sk,sampling=s).astype(np.float32),sk.astype(np.uint8)
