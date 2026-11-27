"""Offline registration QA and optional conservation-aware refinement.
Original fields remain immutable. Outputs are research estimates, not clinical clearance.
"""
import argparse
import json
from pathlib import Path
import tempfile
import nibabel as nib
import numpy as np
from computational_assets import checked_file,file_hash
from paired_lung_workflow import grid
from deformation_mechanics import kinematics
from inverse_consistency import evaluate
from mass_conservation_refinement import refine

def load_package(root,first,second,first_hash,second_hash,a,b,shape,phase):
    root=Path(root).resolve(strict=True)
    index=checked_file(root,"registration.json")
    if index.stat().st_size>262144:raise ValueError("Registration index too large")
    index_hash=file_hash(index)
    m=json.loads(index.read_text())
    other="expiration" if phase=="inspiration" else "inspiration"
    if m.get("schema")!="pulmo.dense-respiratory-registration/1" or m.get("direction")!="fixed_RAS_to_moving_RAS_mm" or m.get("fixed_phase")!=phase or m.get("moving_phase")!=other or m.get("same_patient_operator_attested") is not True or m.get("fixed_ct_sha256")!=first_hash or m.get("moving_ct_sha256")!=second_hash:
        raise ValueError("Registration source or respiratory phase mismatch")
    if tuple(m["shape"])!=tuple(shape) or not np.allclose(m["fixed_voxel_to_ras_mm"],a,rtol=0,atol=1e-4) or not np.allclose(m["moving_voxel_to_ras_mm"],b,rtol=0,atol=1e-4):
        raise ValueError("Registration source grid mismatch")
    p=checked_file(root,"displacement-ras-mm.nii");meta=m["assets"][p.name]
    if file_hash(p)!=meta["sha256"] or p.stat().st_size!=meta["bytes"]:raise ValueError("Displacement integrity mismatch")
    im=nib.load(str(p));s,sc=im.get_sform(coded=True);q,qc=im.get_qform(coded=True)
    if im.shape!=tuple(shape)+(3,) or not sc or im.header.get_xyzt_units()[0]!="mm" or not np.allclose(s,a,rtol=0,atol=1e-4) or (qc and not np.allclose(q,s,rtol=0,atol=1e-4)):
        raise ValueError("Displacement grid mismatch")
    u=im.get_fdata(dtype=np.float32)
    if not np.isfinite(u).all() or np.max(np.linalg.norm(u,axis=-1))>200:raise ValueError("Invalid displacement")
    return u,{index:index_hash,p:meta["sha256"]}

def run(inspiration,expiration,forward,reverse,output,*,landmarks=None,inspiration_mask=None,expiration_mask=None,noncontrast_HU_attested=False,parenchymal_masks_attested=False,refinement_parameters=None):
    out=Path(output).resolve()
    if out.exists():raise ValueError("New output required")
    paths=[Path(p).resolve(strict=True) for p in (inspiration,expiration)]
    if any(out==p or out in p.parents for p in paths) or any(out==Path(p).resolve() or Path(p).resolve() in out.parents for p in (forward,reverse)):
        raise ValueError("Separate output required")
    hashes={p:file_hash(p) for p in paths}
    if len(set(hashes.values()))!=2:raise ValueError("Distinct phase images required")
    x,a=grid(paths[0]);y,b=grid(paths[1])
    u,h=load_package(forward,*paths,*hashes.values(),a,b,x.shape,"inspiration");hashes.update(h)
    v,h=load_package(reverse,paths[1],paths[0],hashes[paths[1]],hashes[paths[0]],b,a,y.shape,"expiration");hashes.update(h)
    fields,report=evaluate(u,a,v,b,landmarks=landmarks)
    arrays={name:(value,b if name.startswith("moving-") else a) for name,value in fields.items()}
    refinement=None
    if inspiration_mask is not None or expiration_mask is not None or refinement_parameters is not None:
        if inspiration_mask is None or expiration_mask is None or noncontrast_HU_attested is not True or parenchymal_masks_attested is not True:
            raise ValueError("Refinement requires both expert masks and noncontrast HU attestation")
        p,q=[Path(p).resolve(strict=True) for p in (inspiration_mask,expiration_mask)]
        hashes.update({p:file_hash(p),q:file_hash(q)})
        mi,_=grid(p,shape=x.shape,affine=a);me,_=grid(q,shape=y.shape,affine=b)
        refined,j,cohort,r0,r1,refinement=refine(x,y,mi,me,a,b,u,**(refinement_parameters or {}))
        f,_,e,_=kinematics(refined,a)
        post,post_report=evaluate(refined,a,v,b,landmarks=landmarks)
        refinement["post_refinement_consistency"]=post_report
        refinement["note"]="Conservation acceptance alone does not establish improved correspondence; review roundtrip and landmark evidence separately"
        arrays.update({"refined-displacement-ras-mm":(refined,a),"refined-jacobian":(j,a),
          "refined-deformation-gradient":(f.reshape(x.shape+(9,)),a),"refined-strain":(e,a),
          "mass-cohort":(cohort,a),"mass-residual-before":(r0,a),"mass-residual-after":(r1,a)})
        arrays.update({"refined-"+name:(value,b if name.startswith("moving-") else a) for name,value in post.items()})
    if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError("Input changed during evaluation")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".registration-qa-",dir=out.parent) as tmp:
        stage=Path(tmp)/"package";stage.mkdir();inventory={}
        for name,(data,affine) in arrays.items():
            p=stage/(name+".nii")
            im=nib.Nifti1Image(data,affine);im.header.set_xyzt_units("mm");im.set_sform(affine,code=1);im.set_qform(None,code=0)
            nib.save(im,str(p))
            inventory[p.name]={"sha256":file_hash(p),"bytes":p.stat().st_size}
        manifest={"schema":"pulmo.registration-quality/1","inspiration_sha256":hashes[paths[0]],"expiration_sha256":hashes[paths[1]],
          "forward_manifest_sha256":hashes[Path(forward).resolve()/"registration.json"],"reverse_manifest_sha256":hashes[Path(reverse).resolve()/"registration.json"],
          "landmarks_RAS_mm":landmarks,"baseline_consistency":report,"conservation_refinement":refinement,
          "assets":inventory,"review":"PENDING","clinical_navigation_allowed":False}
        if refinement is not None:
            manifest["mask_sha256"]={"inspiration":hashes[Path(inspiration_mask).resolve()],"expiration":hashes[Path(expiration_mask).resolve()]}
        (stage/"registration-quality.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
        if out.exists():raise ValueError("Output appeared during evaluation")
        stage.rename(out)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("inspiration","expiration","forward","reverse","output"):p.add_argument(name)
    p.add_argument("--landmarks-json");p.add_argument("--inspiration-mask");p.add_argument("--expiration-mask")
    p.add_argument("--noncontrast-HU-attested",action="store_true");p.add_argument("--parenchymal-masks-attested",action="store_true")
    p.add_argument("--refinement-json")
    a=p.parse_args()
    print(json.dumps(run(a.inspiration,a.expiration,a.forward,a.reverse,a.output,
      landmarks=json.loads(Path(a.landmarks_json).read_text()) if a.landmarks_json else None,
      inspiration_mask=a.inspiration_mask,expiration_mask=a.expiration_mask,
      noncontrast_HU_attested=a.noncontrast_HU_attested,parenchymal_masks_attested=a.parenchymal_masks_attested,
      refinement_parameters=json.loads(Path(a.refinement_json).read_text()) if a.refinement_json else None),indent=2))
