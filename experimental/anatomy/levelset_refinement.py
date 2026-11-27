"""Research prototype: explicit 3D level-set evolution on orthogonal mm grids.
Positive level set is inside. No automatic anatomical identity or clinical acceptance.
"""
import numpy as np
from scipy import ndimage as ndi

CONNECTIVITY=ndi.generate_binary_structure(3,1)

def evolve(x,s,allowed,fg,bg,*,seed_radius_mm,inside_hu,outside_hu,max_volume_ml,
           curvature_mm=.25,iterations=20,contrast_hu=100.,speed=1.):
    x=np.asarray(x,dtype=np.float32);s=np.asarray(s,dtype=float)
    if x.ndim!=3 or min(x.shape)<3 or x.size>2_000_000 or not np.isfinite(x).all():
        raise ValueError("Finite ROI of at most two million voxels required")
    if s.shape!=(3,) or not np.isfinite(s).all() or np.any(s<.1) or np.any(s>10):
        raise ValueError("Invalid physical spacing")
    allowed,fg,bg=[np.asarray(v,dtype=bool) for v in (allowed,fg,bg)]
    if any(v.shape!=x.shape for v in (allowed,fg,bg)) or not fg.any() or np.any(fg&(~allowed|bg)):
        raise ValueError("Invalid seed or domain")
    if not np.isfinite([seed_radius_mm,inside_hu,outside_hu,max_volume_ml,curvature_mm,contrast_hu,speed]).all():
        raise ValueError("Finite parameters required")
    if not 0<seed_radius_mm<=50 or inside_hu==outside_hu or not 0<max_volume_ml or not 0<=curvature_mm<=5 or not 1<=contrast_hu<=5000 or not 0<speed<=5 or type(iterations) is not int or not 1<=iterations<=100:
        raise ValueError("Invalid level-set parameters")
    domain=ndi.binary_propagation(fg,structure=CONNECTIVITY,mask=allowed&~bg)
    initial=(ndi.distance_transform_edt(~fg,sampling=s)<=seed_radius_mm)&domain
    # Physical voxel-center initialization, positive inside.
    phi=(ndi.distance_transform_edt(np.pad(initial,1),sampling=s)[1:-1,1:-1,1:-1]-ndi.distance_transform_edt(~initial,sampling=s)).astype(np.float32)
    initial_phi=phi.copy()
    # Bounded region force: positive for intensities closer to inside prototype.
    force=np.tanh(((x-outside_hu)**2-(x-inside_hu)**2)/(contrast_hu**2))*speed
    if not np.isfinite(force).all(): raise ValueError("Invalid region force")
    # Conservative explicit time-step heuristic, not proof of convergence.
    dt=.2/(speed*np.sum(1/s)+2*curvature_mm*np.sum(1/s**2))
    hmin=float(min(s));voxel_ml=float(np.prod(s))/1000
    def constrain(p):
        p[~domain]=np.minimum(p[~domain],-hmin/2)
        p[fg]=np.maximum(p[fg],hmin/2)
        if np.count_nonzero(p>0)*voxel_ml>max_volume_ml:
            raise ValueError("Level-set candidate exceeds volume ceiling")
    constrain(phi)
    history=[int(np.count_nonzero(phi>0))]
    for _ in range(iterations):
        gradients=np.gradient(phi,*s)
        norm=np.sqrt(sum(g*g for g in gradients)+1e-8)
        curvature=sum(np.gradient(g/norm,s[i],axis=i) for i,g in enumerate(gradients))
        # Positive-inside convention: divergence of inward normal is negative on convex surfaces.
        # Godunov upwind norm for the regional force.
        plus=np.zeros_like(phi);minus=np.zeros_like(phi)
        for i,h in enumerate(s):
            pad=[(0,0)]*3;pad[i]=(1,1)
            p=np.pad(phi,pad,mode="edge")
            l=[slice(None)]*3;r=l.copy();l[i]=slice(0,-2);r[i]=slice(2,None)
            back=(phi-p[tuple(l)])/h;forward=(p[tuple(r)]-phi)/h
            plus+=np.maximum(np.maximum(forward,0)**2,np.minimum(back,0)**2)
            minus+=np.maximum(np.minimum(forward,0)**2,np.maximum(back,0)**2)
        change=force*np.sqrt(np.where(force>=0,plus,minus))+curvature_mm*curvature*norm
        # Bound individual updates to reduce instability on under-resolved interfaces.
        phi+=np.clip(dt*change,-.25*hmin,.25*hmin)
        if not np.isfinite(phi).all(): raise ValueError("Nonfinite level-set evolution")
        constrain(phi)
        history.append(int(np.count_nonzero(phi>0)))
    before=phi>0
    mask=ndi.binary_propagation(fg,structure=CONNECTIVITY,mask=before)
    phi[before&~mask]=-hmin/2
    return mask,{"levelset-initial.nii":initial_phi,"levelset-final.nii":phi,"region-force.nii":force.astype(np.float32)},{
        "method":"region_levelset","positive_inside":True,"iterations_completed":iterations,
        "time_step_model_units":float(dt),"voxel_history":history,"removed_disconnected_voxels":int(before.sum()-mask.sum()),
        "max_volume_ml":float(max_volume_ml),"field_convention":"initial voxel-center distance in mm; evolved level-set field in mm, not reinitialized signed distance",
        "numerical_status":"prototype_explicit_scheme_no_convergence_certification"}
