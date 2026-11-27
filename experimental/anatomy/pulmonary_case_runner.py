"""Local multi-structure candidate generation. No clinical publication or inference."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import nibabel as nib
from computational_assets import canonical, file_hash
from pulmonary_protocols import compile_protocol
from ct_feature_workflow import execute as generate_features
from segmentation_workflow import execute as generate_candidate
from multistructure_worker import build as reconstruct
from study_worker import validate_ct

LOBES={"RUL","RML","RLL","LUL","LLL"}

def intersection_report(a,b,voxel_ml):
    """Only compare mutually evaluated ROIs; outside either ROI is unknown."""
    lo=[max(a["roi"][2*i],b["roi"][2*i]) for i in range(3)]
    hi=[min(a["roi"][2*i+1],b["roi"][2*i+1]) for i in range(3)]
    pair=[a["structure"],b["structure"]]
    policy="expected_disjoint_review" if set(pair)<=LOBES or set(pair)=={"pulmonary_artery","pulmonary_vein"} else "descriptive_overlap_only"
    result={"structures":pair,"policy":policy,"common_evaluated_voxels":0,"overlap_voxels":None,
            "overlap_ml":None,"dice_on_common_roi":None,"coverage":"no_common_roi","review":"PENDING"}
    if any(i>=j for i,j in zip(lo,hi)):return result
    ia=nib.load(str(a["roi_mask"]));ib=nib.load(str(b["roi_mask"]))
    def crop(im,item):
        return np.asarray(im.dataobj[tuple(slice(i-item["roi"][2*k],j-item["roi"][2*k]) for k,(i,j) in enumerate(zip(lo,hi)))],dtype=bool)
    x=crop(ia,a);y=crop(ib,b)
    overlap=int(np.count_nonzero(x&y));den=int(x.sum()+y.sum())
    result.update(common_evaluated_voxels=int(x.size),overlap_voxels=overlap,overlap_ml=overlap*voxel_ml,
                  dice_on_common_roi=2*overlap/den if den else None,coverage="common_roi_only")
    if overlap and policy=="expected_disjoint_review":result["review"]="OVERLAP_REVIEW_REQUIRED"
    return result

def run(intake,output,cache,requests,*,geometry=False):
    root=Path(intake).resolve(strict=True);out=Path(output).resolve();cache=Path(cache).resolve()
    if out.exists() or any(a==b or a in b.parents or b in a.parents for a,b in ((root,out),(root,cache),(out,cache))):
        raise ValueError("Separate intake, cache and new output required")
    plan=compile_protocol(root,requests)
    source=root/"source.nii";im=validate_ct(source)
    if file_hash(source)!=plan["source_sha256"]:raise ValueError("Source changed")
    voxel_ml=abs(float(np.linalg.det(im.affine[:3,:3])))/1000
    out.parent.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".case-",dir=out.parent) as tmp:
        temp=Path(tmp);stage=temp/"case";stage.mkdir()
        masks=temp/"masks";masks.mkdir()
        rows=[];feature_cache={}
        for task in plan["tasks"]:
            name=task["structure"];roi=task["roi"];enh=task["enhancement"]
            key=hashlib.sha256(canonical({"roi":roi,"enhancement":enh})).hexdigest()
            if key not in feature_cache:
                f=temp/("features-"+key)
                generate_features(root,f,cache,roi=roi,**enh)
                feature_cache[key]=f
            f=feature_cache[key]
            dest=stage/"structures"/name
            manifest=generate_candidate(f,dest,structure=name,**task["segmentation"])
            review=json.loads((dest/"segmentation-review.json").read_text())
            if review["source_sha256"]!=plan["source_sha256"] or review["roi"]!=roi:
                raise ValueError("Candidate source mismatch")
            path=dest/"candidate-source-grid.nii.gz"
            shutil.copyfile(path,masks/(name+".nii.gz"))
            if file_hash(path)!=file_hash(masks/(name+".nii.gz")):raise ValueError("Mask copy mismatch")
            rows.append({"structure":name,"roi":roi,"roi_mask":dest/"candidate-roi.nii.gz",
                         "mask_sha256":file_hash(path),"asset_manifest_sha256":file_hash(dest/"computational-assets.json"),
                         "volume_ml":review["volume_ml"],"components":review["components"],
                         "touches_roi_boundary":review["touches_roi_boundary"],
                         "feature_group":key,"outside_roi":"not_evaluated",
                         "asset_count":len(manifest["assets"])})
        pairs=[intersection_report(a,b,voxel_ml) for a,b in itertools.combinations(rows,2)]
        if file_hash(source)!=plan["source_sha256"]:raise ValueError("Source changed during case")
        geom=None
        if geometry:
            # Existing reconstruction creates closed ROI-capped surfaces. Keep
            # this separate from any ordinary publication-ready package.
            geom=reconstruct(source,stage/"research-geometry",plan["study_id"],mask_dir=masks,
                             provider="PulmoSIMU deterministic ROI candidates",version="AMIC06E/1")
            if geom["source_sha256"]!=plan["source_sha256"]:raise ValueError("Geometry source mismatch")
        report={"schema":"pulmo.deterministic-case/1","study_id":plan["study_id"],
                "source_sha256":plan["source_sha256"],"plan_sha256":plan["plan_sha256"],
                "status":"review_pending","clinical_navigation_allowed":False,
                "structures":[{k:v for k,v in a.items() if k!="roi_mask"} for a in rows],
                "feature_groups":len(feature_cache),"pairwise_overlap":pairs,
                "geometry":None if geom is None else {"path":"research-geometry/pulmonary.glb",
                   "manifest_sha256":file_hash(stage/"research-geometry"/"study-package.json"),
                   "derived_structures":geom["available_structures"],
                   "warning":"ROI-limited candidate geometry; artificial caps possible; not approved for publication or navigation"},
                "limitations":["Expert seeds and case-specific parameters required","Zero fill outside ROI is unassessed, not anatomical absence",
                               "Overlap checks are descriptive, not anatomical ground truth"]}
        (stage/"protocol.json").write_text(json.dumps(plan,indent=2,allow_nan=False))
        (stage/"case-review.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        if out.exists():raise ValueError("Output appeared during case execution")
        stage.rename(out)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("intake");p.add_argument("output");p.add_argument("cache");p.add_argument("requests_json")
    p.add_argument("--geometry",action="store_true")
    a=p.parse_args()
    print(json.dumps(run(a.intake,a.output,a.cache,json.loads(Path(a.requests_json).read_text()),geometry=a.geometry),indent=2))
