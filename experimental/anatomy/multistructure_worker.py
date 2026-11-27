#!/usr/bin/env python3
"""INF05D: independent pulmonary masks in one physical frame. Trusted offline execution."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import uuid
import numpy as np
import nibabel as nib
from segmentation_pipeline import prepare, load_source, sha, write_json
from anatomy_fidelity import load_bound, reconstruct
from study_worker import validate_ct, MODEL_VERSION, inventory

LOBES={"RUL":"lung_upper_lobe_right","RML":"lung_middle_lobe_right","RLL":"lung_lower_lobe_right","LUL":"lung_upper_lobe_left","LLL":"lung_lower_lobe_left"}
VESSELS={"airway_lumen":"lung_airways","pulmonary_artery":"lung_arteries","pulmonary_vein":"lung_veins"}
KINDS=tuple(VESSELS)+tuple(LOBES)+("lung",)
COLORS={"airway_lumen":[.68,.78,.81,1],"pulmonary_artery":[.38,.55,.75,1],"pulmonary_vein":[.75,.40,.42,1],"RUL":[.79,.70,.63,1],"RML":[.75,.67,.56,1],"RLL":[.68,.64,.59,1],"LUL":[.74,.72,.67,1],"LLL":[.64,.67,.65,1],"lung":[.72,.72,.70,1]}
MAX_TOTAL_FACES=3_000_000
MAX_GLB=150*1024*1024

def shared_frame(ct):
    center=nib.affines.apply_affine(ct.affine,(np.asarray(ct.shape)-1)/2)
    rotation=np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=float)
    t=np.eye(4);t[:3,:3]=rotation/1000;t[:3,3]=-rotation@center/1000
    return t

def write_scene(path,meshes):
    """Self-contained binary glTF; structure names and geometry remain separate."""
    doc={"asset":{"version":"2.0","generator":"PulmoSIMU INF05D"},"scene":0,"scenes":[{"nodes":[]}],"nodes":[],"meshes":[],"materials":[],"buffers":[{"byteLength":0}],"bufferViews":[],"accessors":[]}
    blocks=[];offset=0;total_faces=0
    def accessor(data,target,component,typ,bounds=False):
        nonlocal offset
        raw=data.tobytes()
        if offset+len(raw)>MAX_GLB: raise ValueError("Combined GLB exceeds byte budget")
        view=len(doc["bufferViews"]);doc["bufferViews"].append({"buffer":0,"byteOffset":offset,"byteLength":len(raw),"target":target})
        item={"bufferView":view,"componentType":component,"count":len(data),"type":typ}
        if bounds:item.update(min=data.min(axis=0).tolist(),max=data.max(axis=0).tolist())
        index=len(doc["accessors"]);doc["accessors"].append(item);blocks.append(raw);offset+=len(raw)
        return index
    for name,pos,norm,faces in meshes:
        if name not in KINDS:raise ValueError("Unknown structure")
        total_faces+=len(faces)
        if total_faces>MAX_TOTAL_FACES:raise ValueError("Combined surface exceeds triangle budget")
        i=len(doc["meshes"])
        p=accessor(pos.astype("<f4"),34962,5126,"VEC3",True)
        n=accessor(norm.astype("<f4"),34962,5126,"VEC3")
        f=accessor(faces.astype("<u4").ravel(),34963,5125,"SCALAR")
        doc["nodes"].append({"mesh":i,"name":name});doc["scenes"][0]["nodes"].append(i)
        doc["meshes"].append({"name":name,"primitives":[{"attributes":{"POSITION":p,"NORMAL":n},"indices":f,"material":i}]})
        doc["materials"].append({"name":name,"pbrMetallicRoughness":{"baseColorFactor":COLORS[name],"metallicFactor":0,"roughnessFactor":.86},"doubleSided":True})
    if not doc["meshes"]:raise ValueError("No surfaces")
    doc["buffers"][0]["byteLength"]=offset
    j=json.dumps(doc,separators=(",",":"),allow_nan=False).encode();j+=b" "*((-len(j))%4)
    size=28+len(j)+offset
    if size>MAX_GLB:raise ValueError("Combined GLB exceeds byte budget")
    with Path(path).open("xb") as f:
        f.write(struct.pack("<4sII",b"glTF",2,size));f.write(struct.pack("<I4s",len(j),b"JSON"));f.write(j)
        f.write(struct.pack("<I4s",offset,b"BIN\0"))
        for block in blocks:f.write(block)

def model_masks(ct,scratch,device):
    if importlib.metadata.version("TotalSegmentator")!=MODEL_VERSION:raise ValueError("Pinned inference provider required")
    exe=shutil.which("TotalSegmentator")
    if not exe:raise ValueError("TotalSegmentator executable unavailable")
    for task,mapping in (("lung_vessels",VESSELS),("total",LOBES)):
        out=scratch/task;out.mkdir(parents=True)
        cmd=[exe,"-i",str(ct),"-o",str(out),"-ta",task,"--device",device]
        if task=="total":cmd+=["--roi_subset",*mapping.values()]
        subprocess.run(cmd,check=True,timeout=3600)
        for kind,label in mapping.items():
            p=out/(label+".nii.gz")
            if not p.is_file():raise ValueError("Provider omitted "+kind)
            yield kind,p,"TotalSegmentator:"+task,MODEL_VERSION

def supplied_masks(folder,provider,version):
    if not provider or not version or len(provider)>120 or len(version)>120:raise ValueError("Explicit provider/version required")
    folder=Path(folder).resolve(strict=True)
    for kind in KINDS:
        paths=[p for p in (folder/(kind+".nii"),folder/(kind+".nii.gz")) if p.is_file()]
        if len(paths)>1:raise ValueError("Duplicate mask: "+kind)
        if paths:
            if paths[0].is_symlink() or paths[0].stat().st_size>750_000_000:raise ValueError("Invalid mask file")
            yield kind,paths[0],provider,version

def build(source,out,study_id,*,mask_dir=None,provider=None,version=None,device="cpu"):
    sid=str(uuid.UUID(str(study_id)));source=Path(source).resolve(strict=True);out=Path(out).resolve()
    if source.suffix.lower()!=".nii" or out.exists():raise ValueError("Uncompressed NIfTI and new output required")
    if device not in ("cpu","gpu"):raise ValueError("Invalid device")
    if mask_dir is None and (provider or version):raise ValueError("Provider claims apply to supplied masks")
    validate_ct(source);original_sha=sha(source)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pulmo-multi-",dir=out.parent) as tmp:
        root=Path(tmp)/"package";root.mkdir()
        ct=root/"source.nii";shutil.copyfile(source,ct)
        if sha(ct)!=original_sha:raise ValueError("Source changed")
        image=validate_ct(ct);transform=shared_frame(image)
        inputs=list(supplied_masks(mask_dir,provider,version) if mask_dir is not None else model_masks(ct,Path(tmp)/"inference",device))
        if not inputs:raise ValueError("No structure masks supplied")
        entries=[("source.nii","ct_nifti",None)];reports={};meshes=[];faces_total=0
        lobe_union=None;lobe_overlap=0;lobe_parents={}
        def add(kind,path,engine,ver,derived=None):
            nonlocal faces_total,lobe_union,lobe_overlap
            before=sha(path);folder=root/"structures"/kind;maskdir=folder/"mask"
            prepare(ct,path,kind,engine,ver,maskdir)
            if sha(path)!=before or sha(ct)!=original_sha:raise ValueError("Source or mask changed")
            _,m,data,a,spacing,crop,origin,count,touches=load_bound(ct,maskdir,structure=kind,max_foreground=5_000_000 if kind=="airway_lumen" else 80_000_000,max_crop=25_000_000 if kind=="airway_lumen" else 150_000_000)
            if kind in LOBES:
                if lobe_union is None:lobe_union=np.zeros(image.shape,dtype=bool)
                lobe_overlap+=int(np.count_nonzero(lobe_union & (data!=0)))
                lobe_union|=data!=0;lobe_parents[kind]=m["mask_sha256"]
            pos,norm,faces,t,q=reconstruct(crop,origin,a,common_transform=transform)
            faces_total+=len(faces)
            if faces_total>MAX_TOTAL_FACES:raise ValueError("Study surface triangle budget exceeded")
            mesh=(kind,pos,norm,faces);meshes.append(mesh)
            write_scene(folder/"surface.glb",[mesh])
            report={"structure":kind,"source_sha256":original_sha,"mask_sha256":m["mask_sha256"],"mesh_sha256":sha(folder/"surface.glb"),"model_from_ras_mm":t.tolist(),"volume_ml":count*abs(float(np.linalg.det(a[:3,:3])))/1000,"foreground_voxels":count,"surface":q,"fov_boundary_faces":touches,"provider":engine,"provider_version":ver,"derived_from":derived,"review":"PENDING","clinical_navigation_allowed":False}
            write_json(folder/"surface.json",report);reports[kind]=report
            for name,role in (("mask/mask.raw","mask"),("mask/mask.json","manifest"),("mask/topology.json","quality_report"),("surface.glb","glb"),("surface.json","manifest")):entries.append((f"structures/{kind}/{name}",role,kind))
            if (maskdir/"airway-skeleton.json").is_file():entries.append((f"structures/{kind}/mask/airway-skeleton.json","airway_graph",kind))
        for kind,path,engine,ver in inputs:add(kind,path,engine,ver)
        if "lung" not in reports and len(lobe_parents)==5:
            p=Path(tmp)/"lung.nii";im=nib.Nifti1Image(lobe_union.astype(np.uint8),image.affine);im.header.set_xyzt_units("mm");im.set_sform(image.affine,code=1);nib.save(im,str(p))
            add("lung",p,"union_of_five_supplied_or_inferred_lobes","1",lobe_parents)
        # The combined scene includes lobes rather than an overlapping whole-lung shell when all five exist.
        combined=[m for m in meshes if m[0]!="lung" or not all(k in reports for k in LOBES)]
        write_scene(root/"pulmonary.glb",combined)
        meta={"source_sha256":original_sha,"mesh_sha256":sha(root/"pulmonary.glb"),"source_shape_vox":list(image.shape),"sform_RAS_mm":image.affine.tolist(),"spacing_mm":np.linalg.norm(image.affine[:3,:3],axis=0).tolist(),"HU_stored_slope":float(image.dataobj.slope),"HU_stored_intercept":float(image.dataobj.inter),"model_from_ras_mm":transform.tolist(),"validation":"UNREVIEWED_MULTI_STRUCTURE_SURFACES","research_flags":["Independent provider masks; no gap filling or smoothing.","Whole-lung union is derived, not independent segmentation.","Boundary padding may cap observed FOV."],"structures":{},"structure_inventory":{k:reports[k]["mesh_sha256"] for k in reports},"combined_structure_names":[m[0] for m in combined]}
        write_json(root/"reconstruction.json",meta)
        entries.extend([("pulmonary.glb","glb",None),("reconstruction.json","manifest",None)])
        manifest={"schema":"pulmo.multi-structure-package/1","study_id":sid,"status":"review_pending","clinical_navigation_allowed":False,"source_sha256":original_sha,"model_from_ras_mm":transform.tolist(),"available_structures":list(reports),"missing_structures":[k for k in KINDS if k not in reports],"lobe_overlap_excess_memberships":lobe_overlap,"assets":inventory(root,entries,sid)}
        write_json(root/"study-package.json",manifest)
        if out.exists():raise ValueError("Output appeared during processing")
        root.rename(out)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source");p.add_argument("out");p.add_argument("--study-id",required=True);p.add_argument("--mask-dir");p.add_argument("--provider");p.add_argument("--version");p.add_argument("--device",choices=("cpu","gpu"),default="cpu")
    a=p.parse_args();print(json.dumps(build(a.source,a.out,a.study_id,mask_dir=a.mask_dir,provider=a.provider,version=a.version,device=a.device),indent=2))
