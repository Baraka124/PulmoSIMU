#!/usr/bin/env python3
"""I01–I03 offline source-linked binary mask preparation and topology.
Python 3.11+, numpy, nibabel, scipy, scikit-image. Local authorized data only.
No automatic inference or clinical branch identification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.morphology import skeletonize

KINDS = ("airway_lumen", "RUL", "RML", "RLL", "LUL", "LLL",
         "pulmonary_artery", "pulmonary_vein", "lung")

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")

def load_source(ct_path, mask_path):
    ct, mask = nib.load(str(ct_path)), nib.load(str(mask_path))
    if len(ct.shape)!=3 or min(ct.shape)<2 or ct.shape!=mask.shape:
        raise ValueError("CT/mask require identical 3D shape")
    if np.prod(ct.shape)>250_000_000:
        raise ValueError("Source exceeds local viewer limit")
    a, ac = ct.header.get_sform(coded=True)
    b, bc = mask.header.get_sform(coded=True)
    if ac<=0 or bc<=0 or ct.header.get_xyzt_units()[0]!="mm" or mask.header.get_xyzt_units()[0]!="mm":
        raise ValueError("Both sources require coded sform and mm units")
    if not np.isfinite(a).all() or abs(np.linalg.det(a[:3,:3]))<1e-12 or not np.allclose(a,b,atol=.001,rtol=0):
        raise ValueError("Mask frame differs; no implicit resampling")
    data=np.asarray(mask.dataobj)
    if not np.all((data==0)|(data==1)):
        raise ValueError("Provide one explicitly binary structure mask (0/1)")
    return data.astype(np.uint8), a

def apply_patch(data, patch, ct_sha, mask_sha, structure):
    if patch.get("schema")!="pulmosimu.mask-patch/1" or patch.get("ct_sha256")!=ct_sha or patch.get("mask_sha256")!=mask_sha or patch.get("structure")!=structure or patch.get("review")!="PENDING":
        raise ValueError("Patch identity mismatch")
    edits=patch.get("edits")
    if not isinstance(edits,list) or len(edits)>10000:
        raise ValueError("Invalid patch size")
    out=data.copy(order="F")
    flat=out.ravel(order="F")
    seen=set()
    for edit in edits:
        i,v=edit["index"],edit["value"]
        if type(i) is not int or not 0<=i<flat.size or i in seen or type(v) is not int or v not in (0,1) or flat[i]==v:
            raise ValueError("Invalid, duplicate or ineffective edit")
        seen.add(i);flat[i]=v
    return out

def topology(data, affine, graph=False):
    # Face-connected lumen is deliberately stricter than diagonal contact.
    labels,n=ndimage.label(data, structure=ndimage.generate_binary_structure(3,1))
    counts=np.bincount(labels.ravel())[1:]
    order=np.argsort(counts)[::-1]
    voxels=int(counts.sum())
    report={"connectivity":6,"component_count":int(n),"foreground_voxels":voxels,
            "volume_ml":voxels*abs(float(np.linalg.det(affine[:3,:3])))/1000,
            "largest_component_fraction":float(counts.max()/voxels) if voxels else None,
            "largest_components_voxels":[int(counts[i]) for i in order[:20]],
            "fragment_voxels":int(voxels-counts.max()) if voxels else 0}
    if not graph or not voxels:
        return report,None
    # Skeletonize only the largest face-connected component. Crop before thinning.
    selected=int(order[0])+1
    region=ndimage.find_objects(labels)[selected-1]
    cropped=(labels[region]==selected)
    sk=skeletonize(cropped,method="lee").astype(bool)
    pts=np.argwhere(sk)+np.array([r.start for r in region])
    if len(pts)>100000:
        raise ValueError("Skeleton exceeds graph export limit; inspect ROI offline")
    lookup={tuple(int(x) for x in p):i for i,p in enumerate(pts)}
    neighbors=[(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if (i,j,k)!=(0,0,0)]
    xyz=nib.affines.apply_affine(affine,pts)
    edges=[];degrees=np.zeros(len(pts),dtype=int)
    for p,i in lookup.items():
        for delta in neighbors:
            j=lookup.get(tuple(x+y for x,y in zip(p,delta)))
            if j is not None and j>i:
                edges.append([i,j,float(np.linalg.norm(xyz[i]-xyz[j]))])
                degrees[i]+=1;degrees[j]+=1
    report.update({"skeleton_voxels":len(pts),"skeleton_adjacency":26,
                   "endpoint_candidates":int(np.count_nonzero(degrees==1)),
                   "junction_voxel_candidates":int(np.count_nonzero(degrees>=3)),
                   "skeleton_scope":"largest_6_connected_component",
                   "graph_status":"UNREVIEWED_VOXEL_SKELETON_NOT_BRANCH_TREE"})
    return report,{"schema":"pulmosimu.airway-skeleton/1","nodes_ras_mm":xyz.tolist(),"edges_index_length_mm":edges,
                   "review":"PENDING","navigation_allowed":False}

def prepare(ct_path, mask_path, structure, engine, version, out_dir, patch_path=None):
    if structure not in KINDS or not engine.strip() or not version.strip():
        raise ValueError("Explicit structure and claimed engine/version required")
    out=Path(out_dir)
    if out.exists():
        raise ValueError("Choose a new output directory")
    data,affine=load_source(ct_path,mask_path)
    original=data.tobytes(order="F")
    ct_sha=sha(ct_path); original_sha=hashlib.sha256(original).hexdigest()
    if patch_path:
        data=apply_patch(data,json.loads(Path(patch_path).read_text()),ct_sha,original_sha,structure)
    raw=data.tobytes(order="F")
    digest=hashlib.sha256(raw).hexdigest()
    report,graph=topology(data,affine,structure=="airway_lumen")
    manifest={"schema":"pulmosimu.mask/1","ct_sha256":ct_sha,"mask_sha256":digest,
              "input_nifti_sha256":sha(mask_path),"dimensions":list(data.shape),"voxel_to_ras_mm":affine.tolist(),
              "structure":structure,"encoding":"uint8-x-fast-binary","engine_claim":engine,"engine_version_claim":version,
              "review":"PENDING","parent_mask_sha256":original_sha if patch_path else None,
              "patch_sha256":sha(patch_path) if patch_path else None}
    report.update({"ct_sha256":ct_sha,"mask_sha256":digest,"structure":structure,"review":"PENDING"})
    if graph is not None:
        graph.update({"ct_sha256":ct_sha,"mask_sha256":digest,"structure":structure})
    out.mkdir(parents=True)
    (out/"mask.raw").write_bytes(raw)
    write_json(out/"mask.json",manifest)
    write_json(out/"topology.json",report)
    if graph is not None: write_json(out/"airway-skeleton.json",graph)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("ct");p.add_argument("mask");p.add_argument("out")
    p.add_argument("--structure",choices=KINDS,required=True)
    p.add_argument("--engine",required=True);p.add_argument("--version",required=True);p.add_argument("--patch")
    a=p.parse_args()
    print(json.dumps(prepare(a.ct,a.mask,a.structure,a.engine,a.version,a.out,a.patch),indent=2))
