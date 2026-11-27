"""Deterministic CT enhancement on orthogonal physical grids.
Responses are image features, not anatomical labels or measured perfusion.
"""
import numpy as np
from scipy import ndimage as ndi

MAX_VOXELS=8_000_000

def validate(data,spacing):
    x=np.asarray(data,dtype=np.float32)
    s=np.asarray(spacing,dtype=float)
    if x.ndim!=3 or min(x.shape)<3 or x.size>MAX_VOXELS or not np.isfinite(x).all():
        raise ValueError("Finite bounded 3D ROI required")
    if s.shape!=(3,) or not np.isfinite(s).all() or np.any(s<0.1) or np.any(s>10):
        raise ValueError("Physical spacing outside supported range")
    return x,s

def diffuse(data,spacing,iterations=8,kappa=40.0,step=None):
    """Explicit conservative Perona-Malik diffusion; zero flux boundary.
    kappa in HU/mm, step in mm squared. No change to source data.
    """
    x,s=validate(data,spacing)
    if type(iterations) is not int or not 1<=iterations<=100 or not np.isfinite(kappa) or kappa<=0:
        raise ValueError("Invalid diffusion parameters")
    limit=0.5/np.sum(1/s**2)
    step=float(0.8*limit if step is None else step)
    if not np.isfinite(step) or not 0<step<=limit:
        raise ValueError("Explicit diffusion stability bound exceeded")
    u=x.copy()
    for _ in range(iterations):
        delta=np.zeros_like(u)
        for axis,h in enumerate(s):
            left=[slice(None)]*3;right=left.copy()
            left[axis]=slice(0,-1);right[axis]=slice(1,None)
            left,right=tuple(left),tuple(right)
            g=(u[right]-u[left])/h
            flux=(g/(1+(g/kappa)**2))/h
            delta[left]+=flux;delta[right]-=flux
        u+=step*delta
    if not np.isfinite(u).all(): raise ValueError("Nonfinite diffusion output")
    return u,{"iterations":iterations,"kappa_HU_per_mm":float(kappa),"step_mm2":step,
              "mean_change_HU":float(np.mean(u,dtype=np.float64)-np.mean(x,dtype=np.float64)),
              "rmse_change_HU":float(np.sqrt(np.mean((u-x)**2,dtype=np.float64)))}

def vesselness(data,spacing,scales_mm=(1.0,2.0,3.0),alpha=.5,beta=.5,gamma=100.0,bright=True):
    """Scale-normalized 3D Frangi response using physical Hessian derivatives.
    gamma has HU units after scale normalization. Best scale is not lumen radius.
    """
    x,s=validate(data,spacing)
    scales=tuple(float(v) for v in scales_mm)
    if not 1<=len(scales)<=8 or sorted(set(scales))!=list(scales) or any(not np.isfinite(v) or not .2<=v<=10 or v/min(s)>30 for v in scales):
        raise ValueError("Invalid physical scales")
    if not all(np.isfinite(v) and v>0 for v in (alpha,beta,gamma)) or type(bright) is not bool:
        raise ValueError("Invalid vesselness parameters")
    best=np.zeros_like(x); chosen=np.zeros_like(x)
    for scale in scales:
        h=np.empty(x.shape+(3,3),dtype=np.float32)
        for i in range(3):
            for j in range(i,3):
                order=[0,0,0];order[i]+=1;order[j]+=1
                d=ndi.gaussian_filter(x,sigma=scale/s,order=order,mode="reflect")*(scale**2/(s[i]*s[j]))
                h[...,i,j]=d;h[...,j,i]=d
        eig=np.linalg.eigvalsh(h)
        del h
        eig=np.take_along_axis(eig,np.argsort(np.abs(eig),axis=-1),axis=-1)
        a,b,c=np.moveaxis(eig,-1,0)
        eps=np.finfo(np.float32).tiny
        ra=np.abs(b)/np.maximum(np.abs(c),eps)
        rb=np.abs(a)/np.maximum(np.sqrt(np.abs(b*c)),eps)
        strength=a*a+b*b+c*c
        response=(-np.expm1(-ra**2/(2*alpha**2)))*np.exp(-rb**2/(2*beta**2))*(-np.expm1(-strength/(2*gamma**2)))
        sign=(b<0)&(c<0) if bright else (b>0)&(c>0)
        response=np.where(sign,response,0)
        if not np.isfinite(response).all(): raise ValueError("Nonfinite Hessian response")
        update=response>best
        best[update]=response[update];chosen[update]=scale
    return best,chosen,{"scales_mm":list(scales),"alpha":alpha,"beta":beta,"gamma_HU":gamma,
                         "polarity":"bright" if bright else "dark","best_scale_zero":"no positive response",
                         "meaning":"tubular image feature, not segmentation or lumen diameter"}
