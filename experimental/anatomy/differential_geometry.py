"""Differential geometric estimates for unreviewed anatomical branch chains."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

def curve(points,*,step_mm=.5,smooth_mm=1.):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or not 2<=len(p)<=100000:
        raise ValueError("Finite physical polyline required")
    if not np.isfinite([step_mm,smooth_mm]).all() or not .1<=step_mm<=5 or not 0<=smooth_mm<=10:
        raise ValueError("Invalid physical sampling or smoothing scale")
    ds=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if np.any(ds<=1e-8):raise ValueError("Repeated consecutive points")
    arc=np.r_[0,np.cumsum(ds)]
    count=int(np.ceil(arc[-1]/step_mm))+1
    if count>100000:raise ValueError("Curve sample budget exceeded")
    if len(p)<4 or count<7:return {"status":"INSUFFICIENT_SAMPLES","samples":[]}
    t=np.linspace(0,arc[-1],count);h=float(t[1]-t[0])
    # The original polyline arc coordinate is retained as a parameter, not
    # assumed to be exact arc length of the smoothed curve.
    q=np.column_stack([np.interp(t,arc,p[:,i]) for i in range(3)])
    if smooth_mm>0:q=gaussian_filter1d(q,smooth_mm/h,axis=0,mode="nearest")
    spline=CubicSpline(t,q,axis=0)
    d1,d2,d3=[spline(t,k) for k in (1,2,3)]
    speed=np.linalg.norm(d1,axis=1);cross=np.cross(d1,d2);cn=np.linalg.norm(cross,axis=1)
    margin=max(2*h,4*smooth_mm)
    good=(t>=margin)&(t<=t[-1]-margin)&(speed>1e-5)
    rows=[]
    for i in np.flatnonzero(good):
        curvature=float(cn[i]/speed[i]**3)
        torsion=float(np.dot(cross[i],d3[i])/cn[i]**2) if curvature>1e-4 and cn[i]>1e-8 else None
        rows.append({"polyline_arc_mm":float(t[i]),"smoothed_RAS_mm":q[i].tolist(),
            "unit_tangent_RAS":(d1[i]/speed[i]).tolist(),"curvature_per_mm":curvature,"torsion_per_mm":torsion})
    return {"status":"ESTIMATED" if rows else "INSUFFICIENT_INTERIOR_SUPPORT","samples":rows,
            "sample_spacing_mm":h,"smoothing_sigma_mm":smooth_mm,"excluded_end_margin_mm":margin,
            "original_length_mm":float(arc[-1]),"torsion_undefined_near_straight":True}

def analyze(graph,*,step_mm=.5,smooth_mm=1.,max_samples=100000):
    if type(max_samples) is not int or not 1<=max_samples<=100000:raise ValueError("Invalid sample budget")
    p=np.asarray(graph["nodes_ras_mm"],dtype=float);rows=[];n=0
    for b in graph["branches"]:
        if b["closed"]:
            r={"status":"CLOSED_BRANCH_NOT_EVALUATED","samples":[]}
        else:
            r=curve(p[b["voxel_node_ids"]],step_mm=step_mm,smooth_mm=smooth_mm)
        n+=len(r["samples"])
        if n>max_samples:raise ValueError("Combined curve sample budget exceeded")
        rows.append({"branch_id":b["id"],**r})
    return {"schema":"pulmo.centerline-differential-geometry/1","branches":rows,"sample_count":n,
       "method":"uniform polyline arc resampling, optional Gaussian smoothing, cubic spline derivatives",
       "limitations":["Torsion is sensitive to voxelization and smoothing","No clinical branch names","Closed chains omitted rather than cut artificially"],
       "review":"PENDING","clinical_navigation_allowed":False}
