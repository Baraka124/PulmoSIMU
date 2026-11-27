"""Offline paired CT registration prototype. Fixed inspiration -> moving expiration.
Dense outputs are ROI-scale research estimates, not measured physiological quantities.
"""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from respiratory_registration import load,sha
from deformation_mechanics import kinematics,MAX_VOXELS

def save(path,values,affine):
    im=nib.Nifti1Image(np.asarray(values,dtype=np.float32),affine)
    im.header.set_xyzt_units("mm");im.set_sform(affine,code=1);im.set_qform(None,code=0)
    nib.save(im,str(path))

def register(fixed_path,moving_path,out,*,iterations=(25,15,10),factors=(4,2,1),same_patient_attested=False):
    if same_patient_attested is not True:raise ValueError("Explicit operator same-patient attestation required")
    if not isinstance(iterations,(tuple,list)) or not isinstance(factors,(tuple,list)) or len(iterations)!=len(factors) or not 1<=len(factors)<=4 or any(type(v) is not int or not 1<=v<=100 for v in iterations) or any(type(v) is not int or not 1<=v<=8 for v in factors) or list(factors)!=sorted(set(factors),reverse=True) or factors[-1]!=1:
        raise ValueError("Descending pyramid factors ending at 1 and bounded iterations required")
    out=Path(out).resolve()
    if out.exists():raise ValueError("New output required")
    fs,ms=sha(fixed_path),sha(moving_path)
    if fs==ms:raise ValueError("Distinct phase inputs required")
    for p in (fixed_path,moving_path):
        im=nib.load(str(p))
        if len(im.shape)!=3 or min(im.shape)<3 or np.prod(im.shape)>MAX_VOXELS:
            raise ValueError("Dense prototype requires bounded phase ROIs of at most two million voxels each")
        sf,sc=im.get_sform(coded=True);q,qc=im.get_qform(coded=True)
        if not sc or (qc and not np.allclose(sf,q,rtol=0,atol=1e-4)):
            raise ValueError("Explicit unambiguous phase geometry required")
    fixed,fa,fi=load(fixed_path);moving,ma,mi=load(moving_path)
    if min(fi.GetSize())//max(factors)<3:raise ValueError("Pyramid too small for ROI")
    identity=sitk.Transform(3,sitk.sitkIdentity)
    mr=sitk.Resample(mi,fi,identity,sitk.sitkLinear,-1000.,sitk.sitkFloat32)
    # Restrict initial registration to grids with substantial physical overlap.
    ones=sitk.Image(mi.GetSize(),sitk.sitkFloat32)+1
    ones.CopyInformation(mi)
    initial_support=sitk.Resample(ones,fi,identity,sitk.sitkNearestNeighbor,0.,sitk.sitkFloat32)
    initial_overlap=float(np.mean(sitk.GetArrayViewFromImage(initial_support)>0))
    if initial_overlap<.5:raise ValueError("Insufficient initial physical overlap; separate alignment required")
    field=None;levels=[]
    for factor,count in zip(factors,iterations):
        def level(image):
            if factor==1:return image
            sigma=.5*factor*min(image.GetSpacing())
            return sitk.Shrink(sitk.SmoothingRecursiveGaussian(image,sigma),[factor]*3)
        f,m=level(fi),level(mr)
        if field is None:
            field=sitk.Image(f.GetSize(),sitk.sitkVectorFloat64,3);field.CopyInformation(f)
        else:
            field=sitk.Resample(field,f,identity,sitk.sitkLinear,0.,sitk.sitkVectorFloat64)
        dem=sitk.DiffeomorphicDemonsRegistrationFilter()
        dem.SetNumberOfIterations(count);dem.SetStandardDeviations(1.0)
        dem.SetNumberOfThreads(1)
        field=dem.Execute(f,m,field)
        metric=float(dem.GetMetric())
        if not np.isfinite(metric):raise ValueError("Nonfinite registration metric")
        levels.append({"shrink_factor":factor,"requested_iterations":count,"metric":metric,"spacing_mm":list(f.GetSpacing())})
    u=sitk.GetArrayFromImage(field).transpose(2,1,0,3).copy()
    u[...,:2]*=-1
    if not np.isfinite(u).all() or np.max(np.linalg.norm(u,axis=-1))>200:
        raise ValueError("Displacement outside prototype budget")
    f,j,e,quality=kinematics(u,fa)
    transform=sitk.DisplacementFieldTransform(sitk.Image(field))
    warped=sitk.GetArrayFromImage(sitk.Resample(mi,fi,transform,sitk.sitkLinear,-1000.,sitk.sitkFloat32)).transpose(2,1,0)
    support=sitk.GetArrayFromImage(sitk.Resample(ones,fi,transform,sitk.sitkNearestNeighbor,0.,sitk.sitkFloat32)).transpose(2,1,0)>0
    if not support.any():raise ValueError("No moving-image support after registration")
    fixed_hu=sitk.GetArrayFromImage(fi).transpose(2,1,0)
    residual=np.where(support,fixed_hu-warped,0)
    initial_residual=fixed_hu-sitk.GetArrayFromImage(mr).transpose(2,1,0)
    common=support&(sitk.GetArrayFromImage(initial_support).transpose(2,1,0)>0)
    if not common.any():raise ValueError("No common support for residual comparison")
    quality.update({"initial_overlap_fraction":initial_overlap,"warped_support_fraction":float(support.mean()),
       "common_support_fraction":float(common.mean()),"initial_rmse_HU":float(np.sqrt(np.mean(initial_residual[common]**2))),
       "warped_rmse_HU":float(np.sqrt(np.mean(residual[common]**2)))})
    if fs!=sha(fixed_path) or ms!=sha(moving_path):raise ValueError("Source changed during registration")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".respiratory-",dir=out.parent) as tmp:
        stage=Path(tmp)/"package";stage.mkdir()
        arrays={"displacement-ras-mm.nii":u,"deformation-gradient.nii":f.reshape(f.shape[:3]+(9,)),
            "jacobian.nii":j,"green-lagrange-strain.nii":e,"warped-moving-clipped-HU.nii":warped,
            "residual-clipped-HU.nii":residual,"moving-support.nii":support}
        for name,value in arrays.items():save(stage/name,value,fa)
        manifest={"schema":"pulmo.dense-respiratory-registration/1","fixed_ct_sha256":fs,"moving_ct_sha256":ms,
          "fixed_phase":"inspiration","moving_phase":"expiration","same_patient_operator_attested":True,"same_patient_independently_verified":False,
          "direction":"fixed_RAS_to_moving_RAS_mm","fixed_voxel_to_ras_mm":fa.tolist(),"moving_voxel_to_ras_mm":ma.tolist(),
          "shape":list(fixed.shape),"method":"multiresolution_diffeomorphic_demons","levels":levels,
          "intensity_clip_HU":[-1000,400],"quality":quality,
          "field_layout":"displacement RAS components; gradient row-major 9 components; strain RR AA SS RA RS AS",
          "support_semantics":"moving image geometric support only, not registration confidence; outside support residual set to zero",
          "limitations":["No independent landmark or inverse consistency validation","No mass preservation or stress solver","ROI edge and missing anatomy can bias registration"],
          "review":"PENDING","clinical_navigation_allowed":False,
          "assets":{name:{"sha256":sha(stage/name),"bytes":(stage/name).stat().st_size} for name in arrays}}
        (stage/"registration.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
        if out.exists():raise ValueError("Output appeared during registration")
        stage.rename(out)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inspiration");p.add_argument("expiration");p.add_argument("output")
    p.add_argument("--same-patient-attested",action="store_true")
    p.add_argument("--factors",nargs="+",type=int,default=[4,2,1]);p.add_argument("--iterations",nargs="+",type=int,default=[25,15,10])
    a=p.parse_args()
    print(json.dumps(register(a.inspiration,a.expiration,a.output,factors=a.factors,iterations=a.iterations,same_patient_attested=a.same_patient_attested),indent=2))
