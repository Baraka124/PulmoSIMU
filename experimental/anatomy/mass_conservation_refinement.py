"""Conservation-aware low-dimensional refinement of an existing dense RAS map.
Not a full deformable mass-preserving registration or physiological validation.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from deformation_mechanics import kinematics

MAX_OPT_VOXELS=500_000

def refine(fixed,moving,mi,me,fa,ma,u,*,scale_bound=.05,translation_bound_mm=2.,regularization=.01,max_iterations=30):
    x=np.asarray(fixed,dtype=np.float32);y=np.asarray(moving,dtype=np.float32)
    mi=np.asarray(mi);me=np.asarray(me);u=np.asarray(u,dtype=np.float32)
    fa=np.asarray(fa,dtype=float);ma=np.asarray(ma,dtype=float)
    if x.ndim!=3 or y.ndim!=3 or min(x.shape+y.shape)<3 or max(x.size,y.size)>MAX_OPT_VOXELS or u.shape!=x.shape+(3,) or mi.shape!=x.shape or me.shape!=y.shape:
        raise ValueError("Bounded aligned phase arrays required")
    if not all(np.isfinite(v).all() for v in (x,y,u,fa,ma)) or not all(np.all((v==0)|(v==1)) for v in (mi,me)):
        raise ValueError("Finite images, transform and binary masks required")
    if ma.shape!=(4,4) or not np.allclose(ma[3],[0,0,0,1]) or abs(np.linalg.det(ma[:3,:3]))<1e-6:
        raise ValueError("Invalid moving affine")
    if not np.isfinite([scale_bound,translation_bound_mm,regularization]).all() or not 0<scale_bound<=.15 or not 0<translation_bound_mm<=5 or not 0<regularization<=10 or type(max_iterations) is not int or not 1<=max_iterations<=100:
        raise ValueError("Invalid bounded refinement parameters")
    _,j0,_,_=kinematics(u,fa)
    if np.any(j0<=0):raise ValueError("Baseline contains folding; refinement cannot repair it")
    idx=np.moveaxis(np.indices(x.shape,dtype=np.float64),0,-1)
    ras=idx@fa[:3,:3].T+fa[:3,3];mapped=ras+u
    inv=np.linalg.inv(ma)
    def coords(p):return (p@inv[:3,:3].T+inv[:3,3])
    def sample(points):
        v=coords(points)
        support=np.all((v>=0)&(v<=np.asarray(y.shape)-1),axis=-1)
        c=v.T
        h=map_coordinates(y,c,order=1,mode="constant",cval=1,prefilter=False)
        mask=map_coordinates(me.astype(np.uint8),c,order=0,mode="constant",cval=0,prefilter=False)>0
        return h,support&mask&(h>=-1000)&(h<=0)
    base=(mi==1)&(x>=-1000)&(x<=0)
    if not base.any():raise ValueError("Empty fixed tissue model")
    base_points=mapped[base]
    base_h,valid=sample(base_points)
    if int(valid.sum())<32:raise ValueError("Insufficient common tissue-model support")
    points=base_points[valid];t=(x[base][valid]+1000)/1000
    jac=j0[base][valid].astype(np.float64)
    center=points.mean(axis=0)
    # A fixed cohort prevents the optimizer improving the objective by dropping difficult voxels.
    bounds=[(-scale_bound,scale_bound)]*3+[(-translation_bound_mm,translation_bound_mm)]*3
    scales=np.array([scale_bound]*3+[translation_bound_mm]*3)
    def evaluate(p):
        stretch=np.exp(p[:3]);q=(points-center)*stretch+center+p[3:]
        h,ok=sample(q)
        if not ok.all():return None
        residual=jac*np.prod(stretch)*(h+1000)/1000-t
        mse=float(np.mean(residual**2))
        penalty=float(regularization*np.mean((p/scales)**2))
        return mse,penalty,residual
    zero=np.zeros(6);baseline=evaluate(zero)
    if baseline is None:raise ValueError("Invalid baseline cohort")
    best=zero.copy();best_value=baseline[0];history=[best_value]
    def objective(p):
        nonlocal best,best_value
        v=evaluate(p)
        if v is None:return 1e6+float(np.sum(p*p))
        total=v[0]+v[1]
        if total<best_value:
            best=p.copy();best_value=total;history.append(total)
        return total
    result=minimize(objective,zero,method="Powell",bounds=bounds,options={"maxiter":max_iterations,"maxfev":max_iterations*60,"xtol":1e-4,"ftol":1e-6})
    stretch=np.exp(best[:3])
    corrected=((mapped-center)*stretch+center+best[3:]-ras).astype(np.float32)
    _,j1,_,_=kinematics(corrected,fa)
    if np.any(j1<=0) or not np.allclose(j1,j0*np.prod(stretch),rtol=2e-4,atol=2e-4):
        raise ValueError("Corrected field failed positive Jacobian or composition identity")
    # Re-evaluate persisted float32 field, not only the optimizer's double precision map.
    cohort=np.zeros(x.shape,dtype=bool);cohort[base]=valid
    h,ok=sample((ras+corrected)[cohort])
    final_res=j1[cohort].astype(np.float64)*(h+1000)/1000-t
    final_mse=float(np.mean(final_res**2));penalty=float(regularization*np.mean((best/scales)**2))
    accepted=bool(np.any(best!=0) and ok.all() and final_mse+penalty<baseline[0])
    if not accepted:
        corrected=u.copy();j1=j0.copy();best=zero;final_res=baseline[2];final_mse=baseline[0];penalty=0.
    residual_before=np.zeros(x.shape,dtype=np.float32);residual_after=residual_before.copy()
    residual_before[cohort]=baseline[2];residual_after[cohort]=final_res
    report={"method":"bounded_diagonal_affine_postcomposition","accepted":accepted,
       "cohort_voxels":int(cohort.sum()),"cohort_fraction_of_fixed_model":float(cohort.sum()/base.sum()),
       "baseline_mass_mse":baseline[0],"refined_mass_mse":final_mse,"regularization_penalty":penalty,
       "baseline_mass_rmse":float(np.sqrt(baseline[0])),"refined_mass_rmse":float(np.sqrt(final_mse)),
       "log_scale_RAS":best[:3].tolist(),"translation_RAS_mm":best[3:].tolist(),"center_RAS_mm":center.tolist(),
       "scale_bound":scale_bound,"translation_bound_mm":translation_bound_mm,"regularization":regularization,
       "optimizer_success":bool(result.success),"function_evaluations":int(result.nfev),"best_objective_history":history,
       "minimum_jacobian_before":float(j0.min()),"minimum_jacobian_after":float(j1.min()),
       "limitations":["Only six global correction parameters; no local deformable optimization","Fixed cohort support enforced; no landmark validation","Water-equivalent tissue volume is a model assumption"],
       "review":"PENDING"}
    return corrected,j1,cohort.astype(np.uint8),residual_before,residual_after,report
