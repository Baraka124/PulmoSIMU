"""Local deterministic CT feature pipeline. Trusted workspace; no clinical upload."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import uuid
import numpy as np
import nibabel as nib
from computational_assets import Registry, canonical, file_hash
from ct_enhancement import diffuse, vesselness, MAX_VOXELS
from study_worker import validate_ct

VERSION="1"
def save_volume(path,data,affine):
    if not np.isfinite(data).all(): raise ValueError("Nonfinite feature")
    im=nib.Nifti1Image(np.asarray(data,dtype=np.float32),affine)
    im.header.set_xyzt_units("mm")
    im.set_sform(affine,code=1);im.set_qform(None,code=0)
    nib.save(im,str(path))

def run_stage(cache,identity,expected,generate):
    key=hashlib.sha256(canonical(identity)).hexdigest()
    target=cache/key
    def verify():
        if target.is_symlink() or not target.is_dir(): raise ValueError("Invalid cache directory")
        index=target/"stage.json"
        if index.is_symlink() or not index.is_file() or index.stat().st_size>262144: raise ValueError("Invalid cache index")
        manifest=json.loads(index.read_text())
        if manifest["identity"]!=identity or set(manifest["files"])!=set(expected): raise ValueError("Cache identity mismatch")
        for name,digest in manifest["files"].items():
            p=target/name
            if p.is_symlink() or not p.is_file() or file_hash(p)!=digest: raise ValueError("Cached output changed")
        return target
    if target.exists(): return verify(),True
    with tempfile.TemporaryDirectory(prefix=".stage-",dir=cache) as tmp:
        stage=Path(tmp)/"result";stage.mkdir()
        generate(stage)
        files={name:file_hash(stage/name) for name in expected}
        (stage/"stage.json").write_text(json.dumps({"identity":identity,"files":files},allow_nan=False))
        if target.exists(): return verify(),True
        stage.rename(target)
    return verify(),False

def execute(intake,output,cache,*,roi,iterations=8,kappa=40.,scales_mm=(1.,2.,3.),bright=True):
    intake=Path(intake).resolve(strict=True);output=Path(output).resolve();cache=Path(cache).resolve()
    if output.exists() or intake==cache or intake in cache.parents or intake in output.parents or cache==output or cache in output.parents or output in cache.parents:
        raise ValueError("Choose separate intake, cache and new output directories")
    report=json.loads((intake/"intake.json").read_text())
    sid=str(uuid.UUID(report["study_id"]))
    source=intake/"source.nii"
    if report.get("schema")!="pulmo.imaging-intake/1" or report.get("modality")!="CT" or report.get("quantity")!="HU":
        raise ValueError("Calibrated CT intake required")
    digest=file_hash(source)
    if digest!=report["normalized"]["sha256"]: raise ValueError("Intake source changed")
    image=validate_ct(source)
    if list(image.shape)!=report["normalized"]["shape"] or not np.allclose(image.affine,report["normalized"]["voxel_to_ras_mm"],rtol=0,atol=1e-4):
        raise ValueError("Intake geometry mismatch")
    if len(roi)!=6 or any(type(x) is not int for x in roi): raise ValueError("Explicit integer ROI required")
    lo=np.asarray(roi[::2]);hi=np.asarray(roi[1::2])
    if np.any(lo<0) or np.any(hi>image.shape) or np.any(hi-lo<3) or int(np.prod(hi-lo))>MAX_VOXELS:
        raise ValueError("ROI outside bounds or processing budget")
    data=np.asarray(image.dataobj[tuple(slice(a,b) for a,b in zip(lo,hi))],dtype=np.float32)
    affine=image.affine.copy();affine[:3,3]=(image.affine@np.r_[lo,1])[:3]
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    if file_hash(source)!=digest: raise ValueError("Source changed while reading")
    versions={k:importlib.metadata.version(k) for k in ("numpy","scipy","nibabel")}
    code_hash=file_hash(Path(__file__).with_name("ct_enhancement.py"))
    base={"study_id":sid,"source_sha256":digest,"roi":list(roi),"version":VERSION,"dependencies":versions,"code_sha256":code_hash}
    cache.mkdir(parents=True,exist_ok=True);output.parent.mkdir(parents=True,exist_ok=True)
    def do_diff(folder):
        u,stats=diffuse(data,spacing,iterations,kappa)
        save_volume(folder/"enhanced.nii",u,affine)
        save_volume(folder/"residual.nii",data-u,affine)
        (folder/"diffusion.json").write_text(json.dumps(stats,allow_nan=False))
    d,hit_d=run_stage(cache,{**base,"operation":"diffusion","iterations":iterations,"kappa":kappa},("enhanced.nii","residual.nii","diffusion.json"),do_diff)
    enhanced=nib.load(str(d/"enhanced.nii")).get_fdata(dtype=np.float32)
    def do_vessel(folder):
        v,s,stats=vesselness(enhanced,spacing,scales_mm,bright=bright)
        save_volume(folder/"vesselness.nii",v,affine)
        save_volume(folder/"best-scale.nii",s,affine)
        (folder/"vesselness.json").write_text(json.dumps(stats,allow_nan=False))
    v,hit_v=run_stage(cache,{**base,"operation":"vesselness","parent_sha256":file_hash(d/"enhanced.nii"),"scales_mm":list(scales_mm),"bright":bright},("vesselness.nii","best-scale.nii","vesselness.json"),do_vessel)
    with tempfile.TemporaryDirectory(prefix=".features-",dir=output.parent) as tmp:
        stage=Path(tmp)/"package";stage.mkdir()
        shutil.copyfile(source,stage/"source.nii")
        if file_hash(stage/"source.nii")!=digest: raise ValueError("Source snapshot mismatch")
        save_volume(stage/"roi.nii",data,affine)
        for folder,names in ((d,("enhanced.nii","residual.nii","diffusion.json")),(v,("vesselness.nii","best-scale.nii","vesselness.json"))):
            for name in names:
                shutil.copyfile(folder/name,stage/name)
                if file_hash(stage/name)!=json.loads((folder/"stage.json").read_text())["files"][name]: raise ValueError("Cache copy mismatch")
        reg=Registry(stage,sid)
        frame="RAS-mm:"+digest
        def add(path,kind,quantity,units,op,parents=(),params=None):
            return reg.add(path,kind,quantity=quantity,units=units,frame_id=frame,operation=op,version=VERSION,parents=parents,parameters=params)
        src=add("source.nii","source_volume","CT intensity","HU","intake",params={"source_sha256":digest})
        crop=add("roi.nii","processed_volume","CT intensity","HU","crop",[src],{"roi":list(roi),"affine_ras_mm":affine.tolist()})
        enh=add("enhanced.nii","processed_volume","filtered CT intensity","HU","diffusion",[crop],{"iterations":iterations,"kappa_HU_per_mm":kappa})
        add("residual.nii","scalar_field","original minus filtered intensity","HU","subtract",[crop,enh])
        add("diffusion.json","quality_report","diffusion statistics","1","report",[enh])
        ven=add("vesselness.nii","scalar_field","tubular feature response","1","frangi",[enh],{"scales_mm":list(scales_mm),"bright":bright})
        add("best-scale.nii","scalar_field","maximum response scale","mm","frangi_scale",[enh,ven])
        add("vesselness.json","quality_report","feature parameters","1","report",[ven])
        manifest=reg.export()
        (stage/"execution.json").write_text(json.dumps({"diffusion_cache_hit":hit_d,"vesselness_cache_hit":hit_v,"code_sha256":code_hash,"dependencies":versions},indent=2))
        if output.exists(): raise ValueError("Output appeared during execution")
        stage.rename(output)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("intake");p.add_argument("output");p.add_argument("cache")
    p.add_argument("--roi",nargs=6,type=int,required=True,metavar=("X0","X1","Y0","Y1","Z0","Z1"))
    p.add_argument("--iterations",type=int,default=8);p.add_argument("--kappa",type=float,default=40)
    p.add_argument("--scales",nargs="+",type=float,default=[1,2,3]);p.add_argument("--dark",action="store_true")
    a=p.parse_args();print(json.dumps(execute(a.intake,a.output,a.cache,roi=a.roi,iterations=a.iterations,kappa=a.kappa,scales_mm=a.scales,bright=not a.dark),indent=2))
