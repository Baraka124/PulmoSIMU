"""Evaluate independently estimated opposite-direction dense maps in physical RAS mm."""
import numpy as np
from scipy.ndimage import map_coordinates
from deformation_mechanics import MAX_VOXELS

def validate(u,a):
    u=np.asarray(u,dtype=np.float32);a=np.asarray(a,dtype=float)
    if u.ndim!=4 or u.shape[-1]!=3 or min(u.shape[:3])<3 or np.prod(u.shape[:3])>MAX_VOXELS or not np.isfinite(u).all():
        raise ValueError("Finite bounded displacement required")
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1]):
        raise ValueError("Invalid affine")
    s=np.linalg.norm(a[:3,:3],axis=0)
    if np.any(s<.1) or np.any(s>10) or not np.allclose((a[:3,:3]/s).T@(a[:3,:3]/s),np.eye(3),rtol=0,atol=1e-5):
        raise ValueError("Orthogonal mm grid required")
    return u,a

def sample(u,a,points):
    inv=np.linalg.inv(a)
    vox=points@inv[:3,:3].T+inv[:3,3]
    valid=np.all((vox>=0)&(vox<=np.asarray(u.shape[:3])-1),axis=-1)
    coords=np.moveaxis(vox,-1,0)
    out=np.stack([map_coordinates(u[...,i],coords,order=1,mode="constant",cval=0,prefilter=False) for i in range(3)],axis=-1)
    return out,valid

def summarize(v):
    return None if not v.size else {"mean_mm":float(v.mean(dtype=np.float64)),"rmse_mm":float(np.sqrt(np.mean(v.astype(np.float64)**2))),"p95_mm":float(np.percentile(v,95)),"max_mm":float(v.max())}

def roundtrip(u,a,v,b):
    u,a=validate(u,a);v,b=validate(v,b)
    result=np.zeros(u.shape[:3],dtype=np.float32);support=np.zeros_like(result,dtype=np.uint8)
    ij=np.indices(u.shape[:2],dtype=float)
    for k in range(u.shape[2]):
        p=np.stack([ij[0],ij[1],np.full(u.shape[:2],k)],axis=-1)
        p=p@a[:3,:3].T+a[:3,3]
        w,ok=sample(v,b,p+u[:,:,k])
        result[:,:,k]=np.where(ok,np.linalg.norm(u[:,:,k]+w,axis=-1),0)
        support[:,:,k]=ok
    return result,support,{"support_fraction":float(support.mean()),"evaluated_voxels":int(support.sum()),"roundtrip":summarize(result[support>0])}

def evaluate(forward,fa,reverse,ma,*,landmarks=None):
    forward,fa=validate(forward,fa);reverse,ma=validate(reverse,ma)
    e1,s1,r1=roundtrip(forward,fa,reverse,ma);e2,s2,r2=roundtrip(reverse,ma,forward,fa)
    report={"fixed_to_moving_to_fixed":r1,"moving_to_fixed_to_moving":r2,
      "interpretation":"round-trip consistency is necessary but not sufficient for anatomical correspondence",
      "zero_fill":"consult corresponding support mask","review":"PENDING"}
    if landmarks is not None:
        if not isinstance(landmarks,list) or not 1<=len(landmarks)<=1000:raise ValueError("Bounded paired landmark list required")
        if any(not isinstance(x,dict) or set(x)!={"fixed_RAS_mm","moving_RAS_mm"} for x in landmarks):raise ValueError("Explicit paired RAS landmarks required")
        x=np.array([p["fixed_RAS_mm"] for p in landmarks],dtype=float);y=np.array([p["moving_RAS_mm"] for p in landmarks],dtype=float)
        if x.shape!=(len(landmarks),3) or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("Invalid landmark coordinates")
        ux,okx=sample(forward,fa,x);uy,oky=sample(reverse,ma,y)
        if not (okx&oky).all():raise ValueError("Landmarks outside evaluated grids")
        report["landmarks"]={"count":len(x),"forward_TRE":summarize(np.linalg.norm(x+ux-y,axis=-1)),
            "reverse_TRE":summarize(np.linalg.norm(y+uy-x,axis=-1)),"reference_independence":"operator supplied; not independently verified"}
    return {"fixed-roundtrip-error":e1,"fixed-roundtrip-support":s1,"moving-roundtrip-error":e2,"moving-roundtrip-support":s2},report
