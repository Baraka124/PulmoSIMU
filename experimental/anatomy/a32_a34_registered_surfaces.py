#!/usr/bin/env python3
"""A3.2-A3.4: make registered GLB surfaces ONLY from separately supplied masks.

Requires numpy vtk trimesh scipy scikit-image; mask files are .npz with
mask[k,j,i] and affine_ras[4,4]. No lungs, lobes, airways or vessels are inferred.
Study assets remain local. This is not clinical or procedural validation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from a31_disjoint_airspace import source_mesh,sha256

NAMES=("RUL","RML","RLL","LUL","LLL",
       "airway_lumen","pulmonary_artery","pulmonary_vein")
COLORS={
"RUL":[169,183,191,220],"RML":[148,170,179,220],"RLL":[177,185,188,220],
"LUL":[179,176,184,220],"LLL":[172,156,168,220],
"airway_lumen":[225,206,176,240],
"pulmonary_artery":[160,96,100,240],
"pulmonary_vein":[107,149,184,240]
}
def build(mask_dir,label_file,manifest_file,outdir,face_budget=65000):
    mask_dir,label_file,manifest_file,outdir=map(Path,(mask_dir,label_file,manifest_file,outdir))
    outdir.mkdir(parents=True,exist_ok=True)
    meta=json.loads(manifest_file.read_text())
    if sha256(label_file)!=meta["source_labelmap_sha256"]:
        raise ValueError("Source A3.1 labelmap SHA-256 mismatch")
    with np.load(label_file,allow_pickle=False) as z:
        labels=np.asarray(z["labels"],dtype=np.uint8)
        affine=np.asarray(z["affine_ras"],dtype=float)
    if not np.allclose(affine,meta["sform_RAS_mm"],atol=1e-4):
        raise ValueError("Source NIfTI sform mismatch")
    model_from_ijk=np.asarray(meta["model_from_ras_mm"])@affine
    scene=trimesh.Scene()
    results={}
    for name in NAMES:
        path=mask_dir/(name+".npz")
        if not path.is_file():
            results[name]={"status":"MISSING_SOURCE_MASK"}
            continue
        with np.load(path,allow_pickle=False) as z:
            if "mask" not in z or "affine_ras" not in z:
                raise ValueError(name+": missing source mask or RAS affine")
            mask=np.asarray(z["mask"],dtype=bool)
            mask_affine=np.asarray(z["affine_ras"],dtype=float)
        if mask.shape!=labels.shape or not np.allclose(affine,mask_affine,atol=1e-4):
            raise ValueError(name+": mask voxel grid does not match CT source")
        if not mask.any():
            raise ValueError(name+": empty source mask")
        mesh,raw_faces=source_mesh(mask.astype(np.uint8),1,model_from_ijk,face_budget)
        mesh.visual.vertex_colors=np.tile(COLORS[name],(len(mesh.vertices),1))
        node="A3_"+name+"_source_mask_UNREVIEWED"
        scene.add_geometry(mesh,geom_name=node,node_name=node)
        results[name]={
           "status":"MESH_FROM_SUPPLIED_MASK_UNREVIEWED",
           "source_mask_sha256":sha256(path),
           "node":node,"voxels":int(mask.sum()),"faces":len(mesh.faces),
           "source_faces_before_decimation":raw_faces,"watertight":bool(mesh.is_watertight)
        }
    asset=None
    if scene.graph.nodes_geometry:
        asset=outdir/"PulmoSIMU_A3_Source_Structures_UNREVIEWED.glb"
        scene.export(str(asset))
    report={
      "protocol":"PulmoSIMU-A3.2-3.4-source-registration-v1",
      "source_CT_SHA256":meta["source_sha256"],
      "source_A31_labels_SHA256":meta["source_labelmap_sha256"],
      "source_geometry":"NIfTI sform RAS + model_from_ras_mm",
      "structure_results":results,
      "output_glb_sha256":sha256(asset) if asset else None,
      "five_lobes_mask_available":all(results[n]["status"]=="MESH_FROM_SUPPLIED_MASK_UNREVIEWED" for n in NAMES[:5]),
      "airway_source_mask_available":results["airway_lumen"]["status"]=="MESH_FROM_SUPPLIED_MASK_UNREVIEWED",
      "vessel_masks_available":all(results[n]["status"]=="MESH_FROM_SUPPLIED_MASK_UNREVIEWED" for n in NAMES[-2:]),
      "independent_expert_review":"NOT_VERIFIED",
      "CT_linked_bronchoscopy":"BLOCKED_REQUIRES_REVIEWED_LUMEN_CENTRELINE_AND_DEVICE",
      "clinically_validated":False
    }
    (outdir/"PulmoSIMU_A3_Source_Structures_QA.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("masks","labels","manifest","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    p.add_argument("--face-budget",type=int,default=65000)
    a=p.parse_args()
    print(json.dumps(build(a.masks,a.labels,a.manifest,a.out,a.face_budget),indent=2))
