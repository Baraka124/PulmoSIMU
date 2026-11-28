"""Offline differential/implicit geometry derived from a source-linked mask."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
from computational_assets import checked_file,file_hash
from differential_geometry import analyze
from volumetric_geometry import fields,skeleton_distance,MAX_VOXELS

def run(ct_path,mask_path,branch_package,output,*,curve_parameters=None,field_parameters=None):
    ct,mask=[Path(p).resolve(strict=True) for p in (ct_path,mask_path)]
    root=Path(branch_package).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or root==out or root in out.parents:raise ValueError("Separate new output required")
    index=checked_file(root,"branch-analysis.json")
    if index.stat().st_size>262144:raise ValueError("Oversized branch manifest")
    hashes={p:file_hash(p) for p in (ct,mask,index)}
    meta=json.loads(index.read_text())
    if meta.get("schema")!="pulmo.branch-analysis/1" or meta["ct_sha256"]!=hashes[ct] or meta["mask_sha256"]!=hashes[mask]:
        raise ValueError("Branch source identity mismatch")
    graph_path=checked_file(root,"branch-graph.json")
    info=meta["assets"]["branch-graph.json"]
    if graph_path.stat().st_size!=info["bytes"] or graph_path.stat().st_size>100_000_000 or file_hash(graph_path)!=info["sha256"]:
        raise ValueError("Graph integrity mismatch")
    hashes[graph_path]=info["sha256"];graph=json.loads(graph_path.read_text())
    if graph.get("schema")!="pulmo.anatomical-branch-graph/1" or any(graph[k]!=meta[k] for k in ("ct_sha256","mask_sha256","structure","shape","voxel_to_ras_mm")):
        raise ValueError("Graph lineage mismatch")
    ci,mi=nib.load(str(ct)),nib.load(str(mask))
    if len(mi.shape)!=3 or ci.shape!=mi.shape or list(mi.shape)!=meta["shape"] or np.prod(mi.shape)>MAX_VOXELS:
        raise ValueError("Bounded matching grid required")
    a,ac=ci.get_sform(coded=True);b,bc=mi.get_sform(coded=True)
    if not ac or not bc or not np.allclose(a,b,rtol=0,atol=1e-4) or not np.allclose(a,meta["voxel_to_ras_mm"],rtol=0,atol=1e-4):
        raise ValueError("Source geometry mismatch")
    for im in (ci,mi):
        q,qc=im.get_qform(coded=True);s,_=im.get_sform(coded=True)
        if im.header.get_xyzt_units()[0]!="mm" or (qc and not np.allclose(q,s,rtol=0,atol=1e-4)):
            raise ValueError("Unambiguous mm grid required")
    m=np.asarray(mi.dataobj)
    arrays,report=fields(m,a,**(field_parameters or {}))
    d,sk=skeleton_distance(m,a,graph)
    p=np.asarray(graph["nodes_voxel_ijk"])
    if not np.allclose(p@a[:3,:3].T+a[:3,3],graph["nodes_ras_mm"],rtol=0,atol=1e-4):
        raise ValueError("Graph physical coordinates inconsistent with mask")
    arrays.update({"sampled-skeleton-distance-mm":d,"sampled-skeleton-mask":sk})
    curves=analyze(graph,**(curve_parameters or {}))
    if any(file_hash(p)!=h for p,h in hashes.items()):raise ValueError("Input changed during analysis")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".differential-",dir=out.parent) as tmp:
        stage=Path(tmp)/"package";stage.mkdir()
        for name,value in arrays.items():
            im=nib.Nifti1Image(value,a);im.header.set_xyzt_units("mm");im.set_sform(a,code=1);im.set_qform(None,code=0)
            nib.save(im,str(stage/(name+".nii")))
        (stage/"curve-geometry.json").write_text(json.dumps(curves,allow_nan=False))
        inventory={p.name:{"sha256":file_hash(p),"bytes":p.stat().st_size} for p in stage.iterdir()}
        manifest={"schema":"pulmo.differential-geometry-package/1","ct_sha256":hashes[ct],"mask_sha256":hashes[mask],
          "branch_manifest_sha256":hashes[index],"branch_graph_sha256":hashes[graph_path],"structure":meta["structure"],
          "voxel_to_ras_mm":a.tolist(),"shape":list(m.shape),"curve_parameters":curve_parameters or {},
          "field_parameters":field_parameters or {},"curve_samples":curves["sample_count"],"field_quality":report,
          "assets":inventory,"review":"PENDING","clinical_navigation_allowed":False}
        (stage/"geometry.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
        if out.exists():raise ValueError("Output appeared during analysis")
        stage.rename(out)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("ct","mask","branch_package","output"):p.add_argument(name)
    p.add_argument("--parameters-json")
    a=p.parse_args()
    print(json.dumps(run(a.ct,a.mask,a.branch_package,a.output,**(json.loads(Path(a.parameters_json).read_text()) if a.parameters_json else {})),indent=2))
