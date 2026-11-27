"""Non-AI geodesic segmentation and edge-weighted binary contour optimisation.
Discrete graph energy, not a continuous level-set PDE or fast marching solver.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

CONNECTIVITY=ndi.generate_binary_structure(3,1)
def refine(x,s,allowed,fg,bg,*,max_cost_mm,gradient_scale=80.,edge_weight=2.,smooth_mm=.5,iterations=8,max_volume_ml=None):
    x=np.asarray(x,dtype=np.float32);s=np.asarray(s,dtype=float)
    if x.ndim!=3 or x.size>2_000_000 or min(x.shape)<3 or not np.isfinite(x).all():
        raise ValueError("Geodesic refinement requires finite ROI of at most 2 million voxels")
    if s.shape!=(3,) or not np.isfinite(s).all() or np.any(s<.1) or np.any(s>10):
        raise ValueError("Invalid spacing")
    allowed,fg,bg=[np.asarray(v,dtype=bool) for v in (allowed,fg,bg)]
    if any(v.shape!=x.shape for v in (allowed,fg,bg)) or not fg.any() or np.any(fg&(~allowed|bg)):
        raise ValueError("Invalid seed or admissible region")
    if not np.isfinite([max_cost_mm,gradient_scale,edge_weight,smooth_mm]).all() or not 0<max_cost_mm<=1000 or not 1<=gradient_scale<=10000 or not 0<=edge_weight<=100 or not 0<=smooth_mm<=10 or type(iterations) is not int or not 0<=iterations<=50:
        raise ValueError("Invalid contour parameters")
    if max_volume_ml is None or not np.isfinite(max_volume_ml) or max_volume_ml<=0:
        raise ValueError("Explicit positive volume ceiling required")
    domain=allowed&~bg
    gradient=np.zeros_like(x)
    for axis,h in enumerate(s):
        g=np.gradient(x,h,axis=axis);gradient+=g*g
    np.sqrt(gradient,out=gradient)
    edge=1/(1+(gradient/gradient_scale)**2)
    cost=np.where(domain,1+edge_weight*(1-edge),np.inf)
    solver=MCP_Geometric(cost,fully_connected=False,sampling=tuple(s))
    arrival,_=solver.find_costs([tuple(p) for p in np.argwhere(fg)])
    reachable=np.isfinite(arrival)&domain
    mask=reachable&(arrival<=max_cost_mm)
    initial=mask.copy()
    def ceiling(m):
        if m.sum()*np.prod(s)/1000>max_volume_ml:
            raise ValueError("Candidate exceeds explicit volume ceiling")
    ceiling(mask)
    # Energy = sum of unary arrival-derived terms plus weighted interface
    # areas. Dividing by voxel volume gives face penalty smooth_mm / spacing.
    unary=np.zeros_like(x)
    unary[reachable]=arrival[reachable]/max_cost_mm-1
    parity=np.zeros(x.shape,dtype=bool)
    for axis,n in enumerate(x.shape):
        shape=[1,1,1];shape[axis]=n
        parity^=(np.arange(n).reshape(shape)%2).astype(bool)
    weights=[]
    for axis,h in enumerate(s):
        l=[slice(None)]*3;r=l.copy()
        l[axis]=slice(0,-1);r[axis]=slice(1,None)
        l,r=tuple(l),tuple(r)
        weights.append((l,r,(smooth_mm/h)*.5*(edge[l]+edge[r])))
    def energy(m):
        return float(unary[m].sum(dtype=np.float64)+sum(w[m[l]!=m[r]].sum(dtype=np.float64) for l,r,w in weights))
    history=[energy(mask)]
    # Red-black coordinate descent: adjacent voxels never change in same half step.
    for _ in range(iterations):
        old=mask.copy()
        for color in (False,True):
            delta=unary.copy()
            for l,r,w in weights:
                delta[l]+=w*np.where(mask[r],-1,1)
                delta[r]+=w*np.where(mask[l],-1,1)
            change=reachable&~fg&(parity==color)
            mask[change&(delta<0)]=True
            mask[change&(delta>0)]=False
        ceiling(mask)
        history.append(energy(mask))
        if np.array_equal(old,mask): break
    # Remove unsupported islands; this final connectivity operation is separate
    # from the energy optimisation and may change its objective.
    mask=ndi.binary_propagation(fg,structure=CONNECTIVITY,mask=mask)
    ceiling(mask)
    return mask,{"geodesic-cost.nii":np.where(reachable,arrival,-1).astype(np.float32),
                 "edge-stopping.nii":edge.astype(np.float32),"initial-contour.nii":initial.astype(np.float32)},{
        "method":"geodesic_discrete_contour","energy_history_before_connectivity":history,
        "final_energy":energy(mask),"iterations_completed":len(history)-1,
        "initial_voxels":int(initial.sum()),"final_voxels":int(mask.sum()),
        "cost_convention":"weighted physical path length in mm; -1 unreachable",
        "boundary_convention":"six-neighbor internal interfaces, no exterior face penalty",
        "max_volume_ml":float(max_volume_ml)}
