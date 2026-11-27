"""Versioned local candidate mask generation from a verified CT feature package."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import nibabel as nib
from computational_assets import Registry, checked_file, file_hash
from deterministic_segmentation import segment, MAX_VOXELS
from ct_feature_workflow import save_volume
from study_worker import validate_ct
from segmentation_pipeline import KINDS

def execute(features,output,*,structure,foreground,lower_hu,upper_hu,method="region_growing",background=None,min_feature=None,refinement=None):
    if structure not in KINDS: raise ValueError("Explicit supported candidate structure required")
    root=Path(features).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or root==out or root in out.parents: raise ValueError("New separate output required")
    index=checked_file(root,"computational-assets.json")
    if index.stat().st_size>262144: raise ValueError("Asset manifest too large")
    m=json.loads(index.read_text())
    if m.get("schema")!="pulmo.computational-assets/1" or not 1<=len(m["assets"])<=32:
        raise ValueError("Unsupported asset registry")
    original=Registry(root,m["study_id"])
    by_path={}
    for row in m["assets"]:
        r=dict(row);key=r.pop("asset_id")
        if key in original.records or r["path"] in by_path: raise ValueError("Duplicate asset identity")
        original.records[key]=r;by_path[r["path"]]=(key,r)
    original.verify()
    source_id,source_record=by_path["source.nii"]
    roi_id,roi_record=by_path["roi.nii"]
    enhanced_id,enhanced_record=by_path["enhanced.nii"]
    if source_record["kind"]!="source_volume" or source_record["quantity"]!="CT intensity" or source_record["units"]!="HU":
        raise ValueError("Calibrated source required")
    if roi_record["parents"]!=[source_id] or enhanced_record["parents"]!=[roi_id] or enhanced_record["units"]!="HU":
        raise ValueError("Feature lineage mismatch")
    source=validate_ct(root/"source.nii")
    im=nib.load(str(root/"enhanced.nii"))
    roi=nib.load(str(root/"roi.nii"))
    if len(im.shape)!=3 or min(im.shape)<3 or np.prod(im.shape)>MAX_VOXELS or im.shape!=roi.shape or not np.allclose(im.affine,roi.affine,rtol=0,atol=1e-4):
        raise ValueError("Invalid enhanced ROI grid")
    bounds=roi_record["parameters"]["roi"]
    if len(bounds)!=6 or any(type(v) is not int for v in bounds): raise ValueError("Invalid ROI bounds")
    lo=np.array(bounds[::2]);hi=np.array(bounds[1::2])
    a=source.affine.copy();a[:3,3]=(source.affine@np.r_[lo,1])[:3]
    if np.any(lo<0) or np.any(hi>source.shape) or tuple(hi-lo)!=im.shape or not np.allclose(a,im.affine,rtol=0,atol=1e-4):
        raise ValueError("ROI not aligned to original CT")
    parents=[enhanced_id];feature=None
    if min_feature is not None:
        vid,vr=by_path["vesselness.nii"]
        vi=nib.load(str(root/"vesselness.nii"))
        if vr["parents"]!=[enhanced_id] or vr["units"]!="1" or vi.shape!=im.shape or not np.allclose(vi.affine,a,rtol=0,atol=1e-4):
            raise ValueError("Vesselness alignment mismatch")
        feature=vi.get_fdata(dtype=np.float32);parents.append(vid)
    intermediates={}
    mask,distance,report=segment(im.get_fdata(dtype=np.float32),np.linalg.norm(a[:3,:3],axis=0),foreground=foreground,
        background=background,lower_hu=lower_hu,upper_hu=upper_hu,method=method,feature=feature,min_feature=min_feature,refinement=refinement,intermediates=intermediates)
    report.update({"schema":"pulmo.candidate-segmentation/1","structure_claim":structure,"roi":bounds,
        "outside_roi":"not_evaluated","source_sha256":source_record["sha256"],"input_asset_ids":parents})
    params={"structure_claim":structure,"foreground_roi_ijk":foreground,"background_roi_ijk":background,
        "lower_hu":lower_hu,"upper_hu":upper_hu,"min_feature":min_feature,
        "refinement":refinement,"levelset_sha256":file_hash(Path(__file__).with_name("levelset_refinement.py")),"refinement_sha256":file_hash(Path(__file__).with_name("geodesic_refinement.py")),
        "algorithm_sha256":file_hash(Path(__file__).with_name("deterministic_segmentation.py")),
        "versions":{k:importlib.metadata.version(k) for k in ("numpy","scipy","scikit-image","nibabel")}}
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".segmentation-",dir=out.parent) as tmp:
        stage=Path(tmp)/"package";stage.mkdir()
        reg=Registry(stage,m["study_id"])
        for key,r in original.records.items():
            target=stage/r["path"];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/r["path"],target)
            reg.records[key]=r
        reg.verify()
        original.verify()
        frame=source_record["frame_id"]
        def add(path,kind,quantity,units,operation,parents,parameters=None):
            return reg.add(path,kind,quantity=quantity,units=units,frame_id=frame,parents=parents,
                           operation=operation,version="1",parameters=parameters)
        def write_mask(path,data,affine):
            nii=nib.Nifti1Image(data.astype(np.uint8,copy=False),affine)
            nii.header.set_xyzt_units("mm");nii.set_sform(affine,code=1);nii.set_qform(None,code=0)
            nib.save(nii,str(path))
        write_mask(stage/"candidate-roi.nii.gz",mask,a)
        if intermediates:
            definitions={"geodesic-cost.nii":("weighted geodesic path length","mm"),"edge-stopping.nii":("edge stopping coefficient","1"),"initial-contour.nii":("initial contour occupancy","1"),"levelset-initial.nii":("initial level-set field","mm"),"levelset-final.nii":("evolved level-set field","mm"),"region-force.nii":("bounded regional model force","1")}
            field_ids=[]
            for name,data in intermediates.items():
                save_volume(stage/name,data,a)
                q,u=definitions[name]
                field_ids.append(add(name,"scalar_field",q,u,method+"_intermediate",parents,params))
            parents=parents+field_ids
        mid=add("candidate-roi.nii.gz","label_mask",structure+" candidate","1",method,parents,params)
        save_volume(stage/"signed-distance.nii",distance,a)
        add("signed-distance.nii","scalar_field","signed voxel-center distance","mm","edt",[mid],{"negative_inside":True})
        full=np.zeros(source.shape,dtype=np.uint8)
        full[tuple(slice(i,j) for i,j in zip(lo,hi))]=mask
        write_mask(stage/"candidate-source-grid.nii.gz",full,source.affine)
        fid=add("candidate-source-grid.nii.gz","label_mask",structure+" ROI-limited candidate","1","embed",[mid,source_id],
                {"outside_roi":"not_evaluated_zero_fill","roi":bounds})
        (stage/"segmentation-review.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        add("segmentation-review.json","quality_report","candidate segmentation statistics","1","report",[mid,fid])
        manifest=reg.export()
        if out.exists(): raise ValueError("Output appeared during processing")
        stage.rename(out)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("features");p.add_argument("output");p.add_argument("parameters_json")
    a=p.parse_args();params=json.loads(Path(a.parameters_json).read_text())
    print(json.dumps(execute(a.features,a.output,**params),indent=2))
