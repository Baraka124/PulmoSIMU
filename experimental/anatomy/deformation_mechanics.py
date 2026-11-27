"""Dense displacement kinematics, not a constitutive mechanics or stress solver.
u maps fixed RAS-mm positions to moving RAS-mm positions.
"""
import numpy as np

MAX_VOXELS=2_000_000
def kinematics(displacement,affine):
    u=np.asarray(displacement,dtype=np.float64);a=np.asarray(affine,dtype=float)
    if u.ndim!=4 or u.shape[-1]!=3 or min(u.shape[:3])<3 or np.prod(u.shape[:3])>MAX_VOXELS or not np.isfinite(u).all():
        raise ValueError("Finite bounded dense RAS displacement required")
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1]):
        raise ValueError("Invalid voxel-to-RAS affine")
    basis=a[:3,:3];s=np.linalg.norm(basis,axis=0)
    if np.any(s<.1) or np.any(s>10) or not np.allclose((basis/s).T@(basis/s),np.eye(3),atol=1e-5,rtol=0):
        raise ValueError("Orthogonal mm grid required")
    inv=np.linalg.inv(basis)
    f=np.empty(u.shape[:3]+(3,3),dtype=np.float64)
    for i in range(3):
        d=np.stack(np.gradient(u[...,i],edge_order=2),axis=-1)
        f[...,i,:]=d@inv
        f[...,i,i]+=1
    jac=np.linalg.det(f)
    # Green-Lagrange strain in fixed RAS basis: E=(F^T F-I)/2.
    e=np.empty(u.shape[:3]+(6,),dtype=np.float32)
    order=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))
    for k,(i,j) in enumerate(order):
        e[...,k]=.5*(np.sum(f[..., :,i]*f[..., :,j],axis=-1)-(i==j))
    f=f.astype(np.float32);jac=jac.astype(np.float32)
    if not all(np.isfinite(v).all() for v in (e,f,jac)):raise ValueError("Nonfinite deformation output")
    report={"direction":"fixed_to_moving","jacobian_min":float(jac.min()),"jacobian_max":float(jac.max()),
            "nonpositive_jacobian_fraction":float(np.mean(jac<=0)),"strain_components":["RR","AA","SS","RA","RS","AS"],
            "deformation_gradient_components":"row-major RAS output by RAS input",
            "derivative":"second-order centered interior and one-sided boundary in physical RAS coordinates",
            "boundary_derivatives":"one-sided; interpret separately","units":"dimensionless",
            "interpretation":"kinematic estimates from registration, not measured ventilation, stress or material properties",
            "review":"PENDING"}
    return f,jac,e,report
