"""Trusted local anatomical geometry analysis, no inferred clinical labels."""
import argparse
import json
from pathlib import Path
import tempfile
import nibabel as nib
import numpy as np
from computational_assets import file_hash
from anatomical_branch_graph import build,MAX_VOXELS
from branch_morphometry import measure

def run(ct_path,mask_path,output,*,structure,section_parameters=None):
    if structure not in ("airway_lumen","pulmonary_artery","pulmonary_vein"):
        raise ValueError("Explicit tubular structure claim required")
    ct_path=Path(ct_path).resolve(strict=True);mask_path=Path(mask_path).resolve(strict=True);out=Path(output).resolve()
    if out.exists():raise ValueError("New output required")
    before=[file_hash(p) for p in (ct_path,mask_path)]
    ct=nib.load(str(ct_path));mi=nib.load(str(mask_path))
    a,ac=ct.get_sform(coded=True);b,bc=mi.get_sform(coded=True)
    if len(ct.shape)!=3 or min(ct.shape)<3 or np.prod(ct.shape)>MAX_VOXELS or ct.shape!=mi.shape or not ac or not bc or not np.allclose(a,b,rtol=0,atol=1e-4):
        raise ValueError("Identical bounded CT and mask grids required")
    for im in (ct,mi):
        q,qc=im.get_qform(coded=True);s,_=im.get_sform(coded=True)
        if im.header.get_xyzt_units()[0]!="mm" or (qc and not np.allclose(q,s,rtol=0,atol=1e-4)):raise ValueError("Unambiguous mm coordinates required")
    m=np.asarray(mi.dataobj)
    g=build(m,a)
    r=measure(m,a,g,**(section_parameters or {}))
    provenance={"ct_sha256":before[0],"mask_sha256":before[1],"structure":structure,"voxel_to_ras_mm":a.tolist(),
                "shape":list(m.shape),"scope":"supplied regional mask; no assertion of complete anatomical coverage"}
    g.update(provenance);r.update(provenance)
    if before!=[file_hash(p) for p in (ct_path,mask_path)]:raise ValueError("Input changed during computation")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".branches-",dir=out.parent) as tmp:
        stage=Path(tmp)/"package";stage.mkdir()
        for name,data in (("branch-graph.json",g),("branch-morphometry.json",r)):
            (stage/name).write_text(json.dumps(data,allow_nan=False))
        manifest={"schema":"pulmo.branch-analysis/1",**provenance,"branch_count":len(g["branches"]),
          "graph_component_count":len(g["graph_components"]),"cycle_rank":g["cycle_rank"],
          "section_count":len(r["sections"]),"estimated_section_count":r["estimated_sections"],
          "section_parameters":section_parameters or {},
          "assets":{p.name:{"sha256":file_hash(p),"bytes":p.stat().st_size} for p in stage.iterdir()},
          "review":"PENDING","clinical_navigation_allowed":False}
        (stage/"branch-analysis.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
        if out.exists():raise ValueError("Output appeared during analysis")
        stage.rename(out)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("ct");p.add_argument("mask");p.add_argument("output")
    p.add_argument("--structure",required=True,choices=("airway_lumen","pulmonary_artery","pulmonary_vein"))
    p.add_argument("--section-json")
    a=p.parse_args()
    print(json.dumps(run(a.ct,a.mask,a.output,structure=a.structure,section_parameters=json.loads(Path(a.section_json).read_text()) if a.section_json else None),indent=2))
