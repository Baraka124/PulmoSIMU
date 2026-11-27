"""Trusted offline paired lung CT analysis using existing registration package.
Explicit expert masks and noncontrast acquisition attestation required.
"""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from computational_assets import checked_file,file_hash
from deformation_mechanics import MAX_VOXELS,kinematics
from quantitative_lung import density,paired

def grid(path,*,shape=None,affine=None):
    im=nib.load(str(path))
    a,sc=im.get_sform(coded=True);q,qc=im.get_qform(coded=True)
    if len(im.shape)!=3 or min(im.shape)<3 or np.prod(im.shape)>MAX_VOXELS or not sc or im.header.get_xyzt_units()[0]!="mm" or not np.isfinite(a).all() or (qc and not np.allclose(a,q,rtol=0,atol=1e-4)):
        raise ValueError("Unambiguous bounded 3D mm grid required")
    s=np.linalg.norm(a[:3,:3],axis=0)
    if np.any(s<.1) or np.any(s>10) or not np.allclose((a[:3,:3]/s).T@(a[:3,:3]/s),np.eye(3),atol=1e-5,rtol=0):
        raise ValueError("Orthogonal supported mm grid required")
    if (shape is not None and tuple(shape)!=im.shape) or (affine is not None and not np.allclose(a,affine,rtol=0,atol=1e-4)):
        raise ValueError("Grid mismatch")
    x=im.get_fdata(dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError("Nonfinite image")
    return x,a

def pull(moving,mask,ma,fa,u):
    """Pull unfiltered moving HU and categorical mask using RAS displacement."""
    shape=u.shape[:3]
    inv=np.linalg.inv(ma)
    out=np.zeros(shape,dtype=np.float32);m=np.zeros(shape,dtype=np.uint8);support=np.zeros(shape,dtype=np.uint8)
    # Slice-wise physical mapping; do not allocate full coordinate tensors.
    ij=np.indices(shape[:2],dtype=np.float64)
    for k in range(shape[2]):
        p=np.stack((ij[0],ij[1],np.full(shape[:2],k)),axis=-1)
        ras=nib.affines.apply_affine(fa,p)+u[:,:,k,:]
        v=nib.affines.apply_affine(inv,ras)
        valid=np.all((v>=0)&(v<=np.asarray(moving.shape)-1),axis=-1)
        coords=np.moveaxis(v,-1,0)
        out[:,:,k]=np.where(valid,map_coordinates(moving,coords,order=1,mode="constant",cval=0,prefilter=False),0)
        m[:,:,k]=np.where(valid,map_coordinates(mask,coords,order=0,mode="constant",cval=0,prefilter=False),0)
        support[:,:,k]=valid
    return out,m,support

def run(fixed,moving,fixed_mask,moving_mask,registration,output,*,noncontrast_HU_attested=False,parenchymal_masks_attested=False):
    if noncontrast_HU_attested is not True or parenchymal_masks_attested is not True:
        raise ValueError("Explicit noncontrast calibrated HU and expert parenchymal mask attestations required")
    root=Path(registration).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or out==root or root in out.parents:raise ValueError("New separate output required")
    index=checked_file(root,"registration.json")
    if index.stat().st_size>262144:raise ValueError("Registration manifest exceeds budget")
    paths=[Path(p).resolve(strict=True) for p in (fixed,moving,fixed_mask,moving_mask)]
    if any(out==p or out in p.parents for p in paths):raise ValueError("Output cannot contain input")
    before=[file_hash(p) for p in paths];reg_hash=file_hash(index)
    r=json.loads(index.read_text())
    if r.get("schema")!="pulmo.dense-respiratory-registration/1" or r.get("direction")!="fixed_RAS_to_moving_RAS_mm" or r.get("fixed_phase")!="inspiration" or r.get("moving_phase")!="expiration" or r.get("same_patient_operator_attested") is not True or before[:2]!=[r["fixed_ct_sha256"],r["moving_ct_sha256"]]:
        raise ValueError("Registration identity or direction mismatch")
    x,fa=grid(paths[0],shape=r["shape"],affine=r["fixed_voxel_to_ras_mm"])
    y,ma=grid(paths[1],affine=r["moving_voxel_to_ras_mm"])
    mi,_=grid(paths[2],shape=x.shape,affine=fa);me,_=grid(paths[3],shape=y.shape,affine=ma)
    if not np.all((mi==0)|(mi==1)) or not np.all((me==0)|(me==1)) or not mi.any() or not me.any():
        raise ValueError("Nonempty binary parenchymal masks required")
    def asset(name,shape):
        p=checked_file(root,name);meta=r["assets"][name]
        if p.stat().st_size!=meta["bytes"] or file_hash(p)!=meta["sha256"]:raise ValueError("Registration asset changed")
        im=nib.load(str(p));a,sc=im.get_sform(coded=True);q,qc=im.get_qform(coded=True)
        if im.shape!=shape or not sc or im.header.get_xyzt_units()[0]!="mm" or not np.allclose(a,fa,rtol=0,atol=1e-4) or (qc and not np.allclose(a,q,rtol=0,atol=1e-4)):
            raise ValueError("Registration asset geometry mismatch")
        v=im.get_fdata(dtype=np.float32)
        if not np.isfinite(v).all():raise ValueError("Nonfinite registration asset")
        return v
    u=asset("displacement-ras-mm.nii",x.shape+(3,))
    j=asset("jacobian.nii",x.shape)
    _,actual_j,_,_=kinematics(u,fa)
    if not np.allclose(j,actual_j,rtol=1e-4,atol=1e-4):raise ValueError("Jacobian inconsistent with displacement")
    yw,mw,support=pull(y,me.astype(np.uint8),ma,fa,u)
    fields,report=paired(x,yw,j,mi,mw,support,abs(float(np.linalg.det(fa[:3,:3])))/1000)
    report["native_inspiration"]=density(x,mi,abs(float(np.linalg.det(fa[:3,:3])))/1000,phase="inspiration")
    report["native_expiration"]=density(y,me,abs(float(np.linalg.det(ma[:3,:3])))/1000,phase="expiration")
    if before!=[file_hash(p) for p in paths] or reg_hash!=file_hash(index):raise ValueError("Input changed during analysis")
    for name in ("displacement-ras-mm.nii","jacobian.nii"):
        if file_hash(root/name)!=r["assets"][name]["sha256"]:raise ValueError("Registration asset changed during analysis")
    fields.update({"warped-expiration-unfiltered-HU":yw,"warped-expiration-mask":mw,"strict-moving-support":support})
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qim-",dir=out.parent) as tmp:
        stage=Path(tmp)/"package";stage.mkdir();inventory={}
        for name,value in fields.items():
            p=stage/(name+".nii")
            im=nib.Nifti1Image(value,fa);im.header.set_xyzt_units("mm");im.set_sform(fa,code=1);im.set_qform(None,code=0)
            nib.save(im,str(p));inventory[p.name]={"sha256":file_hash(p),"bytes":p.stat().st_size}
        manifest={"schema":"pulmo.paired-lung-quantification/1","input_sha256":dict(zip(("inspiration","expiration","inspiration_mask","expiration_mask"),before)),
          "registration_manifest_sha256":reg_hash,"registration_asset_sha256":{n:r["assets"][n]["sha256"] for n in ("displacement-ras-mm.nii","jacobian.nii")},
          "noncontrast_HU_operator_attested":True,"parenchymal_masks_operator_attested":True,"reference_voxel_to_ras_mm":fa.tolist(),
          "interpolation":"unfiltered expiration HU trilinear; mask nearest; strict in-grid support",
          "quantification":report,"assets":inventory,"clinical_navigation_allowed":False,"review":"PENDING"}
        (stage/"quantification.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
        if out.exists():raise ValueError("Output appeared during analysis")
        stage.rename(out)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("inspiration","expiration","inspiration_mask","expiration_mask","registration","output"):p.add_argument(name)
    p.add_argument("--noncontrast-HU-attested",action="store_true");p.add_argument("--parenchymal-masks-attested",action="store_true")
    a=p.parse_args()
    print(json.dumps(run(a.inspiration,a.expiration,a.inspiration_mask,a.expiration_mask,a.registration,a.output,noncontrast_HU_attested=a.noncontrast_HU_attested,parenchymal_masks_attested=a.parenchymal_masks_attested),indent=2))
