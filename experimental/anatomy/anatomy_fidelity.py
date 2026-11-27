#!/usr/bin/env python3
"""V01–V03: local source-mask surface reconstruction and geometry evidence.
No segmentation inference, gap filling, clinical approval or upload.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.measure import marching_cubes
from segmentation_pipeline import sha, write_json

LANDMARKS = ("oral_cavity", "pharynx", "larynx", "trachea", "distal_airway")
MAX_CROP = 25_000_000
MAX_FACES = 1_500_000

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def load_bound(ct_path, mask_dir):
    ct = nib.load(str(ct_path))
    manifest = json.loads((Path(mask_dir)/"mask.json").read_text())
    raw_path = Path(mask_dir)/"mask.raw"
    ct_sha = sha(ct_path)
    require(manifest.get("schema")=="pulmosimu.mask/1" and manifest.get("structure")=="airway_lumen"
            and manifest.get("encoding")=="uint8-x-fast-binary" and manifest.get("review")=="PENDING"
            and manifest.get("ct_sha256")==ct_sha, "Source mask identity mismatch")
    dims = tuple(ct.shape)
    require(len(dims)==3 and min(dims)>=2 and int(np.prod(dims))<=250_000_000 and list(dims)==manifest.get("dimensions"),
            "Invalid source dimensions")
    a, code = ct.header.get_sform(coded=True)
    require(code>0 and ct.header.get_xyzt_units()[0]=="mm" and np.isfinite(a).all(), "Explicit mm sform required")
    b = np.asarray(manifest.get("voxel_to_ras_mm"), dtype=float)
    require(b.shape==(4,4) and np.allclose(a,b,atol=.001,rtol=0), "Mask affine mismatch")
    spacing = np.linalg.norm(a[:3,:3],axis=0)
    require(np.all((spacing>=.001)&(spacing<=100)) and np.allclose((a[:3,:3]/spacing).T@(a[:3,:3]/spacing),np.eye(3),atol=1e-5),
            "Orthogonal physical voxel frame required")
    require(raw_path.stat().st_size==int(np.prod(dims)) and sha(raw_path)==manifest.get("mask_sha256"), "Mask bytes/hash mismatch")
    data = np.memmap(raw_path,dtype=np.uint8,mode="r",shape=dims,order="F")
    require(int(data.max())<=1, "Nonbinary mask")
    count = int(np.count_nonzero(data))
    require(0<count<=5_000_000, "Empty or oversized foreground; prepare a reviewed ROI")
    # Axis-wise projections avoid allocating three full foreground coordinate arrays.
    bounds=[]
    for axis in range(3):
        occupied=np.flatnonzero(np.any(data,axis=tuple(i for i in range(3) if i!=axis)))
        bounds.append((int(occupied[0]),int(occupied[-1])+1))
    crop_shape=tuple(hi-lo+2 for lo,hi in bounds)
    require(int(np.prod(crop_shape))<=MAX_CROP, "Cropped mask exceeds local reconstruction budget")
    crop=np.pad(np.asarray(data[tuple(slice(lo,hi) for lo,hi in bounds)]),1)
    origin=np.array([lo-1 for lo,_ in bounds])
    touches=[name for i,(lo,hi) in enumerate(bounds) for name,yes in [(f"axis{i}_min",lo==0),(f"axis{i}_max",hi==dims[i])] if yes]
    return ct,manifest,data,a,spacing,crop,origin,count,touches

def reconstruct(crop, origin, affine):
    v,f,_,_=marching_cubes(crop,level=.5,allow_degenerate=False,method="lewiner",gradient_direction="descent")
    require(0<len(f)<=MAX_FACES, "Surface exceeds face budget; use a smaller verified ROI")
    ras=nib.affines.apply_affine(affine,v+origin)
    # Work in a centered right-handed metric glTF frame: X=R, Y=S, Z=-A.
    center=(ras.min(axis=0)+ras.max(axis=0))/2
    rotation=np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=float)
    transform=np.eye(4); transform[:3,:3]=rotation/1000;transform[:3,3]=-rotation@center/1000
    pos=nib.affines.apply_affine(transform,ras).astype("<f4")
    f=f.astype("<u4")
    # Orient the entire closed boundary consistently. Negative determinant affines
    # and marching-cubes winding conventions are covered by signed-volume check.
    p=pos.astype(float)
    tri=p[f]
    signed=float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
    if signed<0:
        f=f[:,[0,2,1]].copy()
        signed=-signed
    tri=p[f]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=float(np.linalg.norm(cross,axis=1).sum()/2*1e6)
    normals=np.zeros_like(p)
    for i in range(3):
        np.add.at(normals,f[:,i],cross)
    lens=np.linalg.norm(normals,axis=1)
    require(np.all(lens>1e-15),"Undefined surface normal")
    normals=(normals/lens[:,None]).astype("<f4")
    # Undirected incidence; no automatic remeshing or topological repair.
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    keys=edges[:,0].astype(np.uint64)*np.uint64(len(pos))+edges[:,1]
    _,inc=np.unique(keys,return_counts=True)
    quality={"vertices":len(pos),"triangles":len(f),"boundary_edges":int(np.sum(inc==1)),
             "nonmanifold_edges":int(np.sum(inc>2)),"surface_area_mm2":area,
             "enclosed_mesh_volume_ml":signed*1e6,"bounds_ras_mm":[ras.min(axis=0).tolist(),ras.max(axis=0).tolist()],
             "surface_method":"LEWINER_0_5_NO_SMOOTHING","boundary_padding":"ONE_ZERO_VOXEL"}
    return pos,normals,f,transform,quality

def make_glb(pos,normals,faces):
    blocks=[pos.astype("<f4").tobytes(),normals.astype("<f4").tobytes(),faces.astype("<u4").tobytes()]
    offsets=np.cumsum([0]+[len(b) for b in blocks[:-1]]).tolist()
    blob=b"".join(blocks)
    doc={"asset":{"version":"2.0","generator":"PulmoSIMU V01"},
         "scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"airway_lumen_unreviewed"}],
         "meshes":[{"name":"airway_lumen_unreviewed","primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"indices":2,"material":0}]}],
         "materials":[{"pbrMetallicRoughness":{"baseColorFactor":[.68,.78,.81,1],"metallicFactor":0,"roughnessFactor":.86},"doubleSided":True}],
         "buffers":[{"byteLength":len(blob)}],
         "bufferViews":[{"buffer":0,"byteOffset":int(o),"byteLength":len(b),"target":34963 if i==2 else 34962} for i,(o,b) in enumerate(zip(offsets,blocks))],
         "accessors":[{"bufferView":0,"componentType":5126,"count":len(pos),"type":"VEC3","min":pos.min(axis=0).tolist(),"max":pos.max(axis=0).tolist()},
                      {"bufferView":1,"componentType":5126,"count":len(normals),"type":"VEC3"},
                      {"bufferView":2,"componentType":5125,"count":int(faces.size),"type":"SCALAR"}]}
    j=json.dumps(doc,separators=(",",":"),allow_nan=False).encode()
    j+=b" "*((-len(j))%4);blob+=b"\0"*((-len(blob))%4)
    return struct.pack("<4sII",b"glTF",2,12+8+len(j)+8+len(blob))+struct.pack("<I4s",len(j),b"JSON")+j+struct.pack("<I4s",len(blob),b"BIN\0")+blob

def landmark_evidence(spec, labels, origin, data, affine):
    if spec is None:
        return {"state":"NOT_ASSESSED","landmarks":[],"same_component":False}
    points=spec.get("landmarks",{})
    require(isinstance(points,dict) and set(points)<=set(LANDMARKS),"Unknown landmark")
    rows=[]
    for name in LANDMARKS:
        if name not in points: continue
        p=points[name]
        require(isinstance(p,list) and len(p)==3 and all(type(x) is int and 0<=x<data.shape[i] for i,x in enumerate(p)), "Invalid landmark voxel")
        q=np.asarray(p)-origin
        component=int(labels[tuple(q)]) if np.all(q>=0) and np.all(q<np.array(labels.shape)) else 0
        rows.append({"name":name,"voxel":p,"ras_mm":nib.affines.apply_affine(affine,p).tolist(),"component":component})
    require(len({tuple(r["voxel"]) for r in rows})==len(rows),"Duplicate landmarks")
    connected=len(rows)==len(LANDMARKS) and len({r["component"] for r in rows})==1 and rows[0]["component"]>0
    return {"state":"ALL_MARKS_SAME_6_CONNECTED_COMPONENT" if connected else "INCOMPLETE_OR_DISCONNECTED",
            "landmarks":rows,"same_component":connected,"anatomical_labels_verified":False,
            "ordered_path_verified":False,"missing_anatomy_inferred":False}

def section(data, affine, item):
    p=item.get("center_voxel"); n=np.asarray(item.get("normal_ras"),dtype=float)
    require(isinstance(p,list) and len(p)==3 and all(type(x) is int and 0<=x<data.shape[i] for i,x in enumerate(p)) and data[tuple(p)]==1,"Section center must be foreground")
    require(n.shape==(3,) and np.isfinite(n).all() and np.linalg.norm(n)>1e-6,"Invalid section normal")
    n/=np.linalg.norm(n)
    radius=float(item.get("radius_mm",15));step=float(item.get("step_mm",.5))
    require(2<=radius<=40 and .25<=step<=2,"Section radius or raster step out of range")
    # Sample a finite physical plane. Only the center-connected 2D region is counted.
    seed=np.eye(3)[int(np.argmin(np.abs(n)))]; u=np.cross(n,seed);u/=np.linalg.norm(u);v=np.cross(n,u)
    half=int(np.ceil(radius/step));coord=np.arange(-half,half+1)*step
    x,y=np.meshgrid(coord,coord,indexing="ij")
    disk=x*x+y*y<=radius*radius
    ras=nib.affines.apply_affine(affine,p)+x[...,None]*u+y[...,None]*v
    vox=nib.affines.apply_affine(np.linalg.inv(affine),ras.reshape(-1,3)).reshape(ras.shape)
    nearest=np.floor(vox+.5).astype(int)
    valid=np.all((nearest>=0)&(nearest<np.array(data.shape)),axis=-1)
    hit=np.zeros(disk.shape,dtype=bool);hit[valid]=data[tuple(nearest[valid].T)]==1;hit&=disk
    labels,_=ndimage.label(hit,structure=ndimage.generate_binary_structure(2,1));component=int(labels[half,half])
    require(component>0,"Center sample lost")
    region=labels==component
    rim=disk&~ndimage.binary_erosion(disk,structure=ndimage.generate_binary_structure(2,1),border_value=0)
    truncated=bool(np.any(region&rim) or np.any(disk&~valid))
    area=float(region.sum()*step*step)
    return {"center_voxel":p,"normal_ras":n.tolist(),"radius_mm":radius,"step_mm":step,
            "center_component_pixels":int(region.sum()),"truncated":truncated,
            "sampled_area_mm2":None if truncated else area,
            "area_equivalent_diameter_mm":None if truncated else float(2*np.sqrt(area/np.pi)),
            "method":"NEAREST_VOXEL_4_CONNECTED_PLANE","minimum_diameter_certified":False}

def run(ct_path,mask_dir,out_dir,landmark_path=None):
    out=Path(out_dir);require(not out.exists(),"Choose new output directory")
    ct,m,data,a,sp,crop,origin,count,touches=load_bound(ct_path,mask_dir)
    spec=None
    if landmark_path:
        p=Path(landmark_path);require(p.stat().st_size<=65536,"Landmark input exceeds limit")
        spec=json.loads(p.read_text())
        require(spec.get("schema")=="pulmosimu.anatomy-landmarks/1" and spec.get("ct_sha256")==m["ct_sha256"] and spec.get("mask_sha256")==m["mask_sha256"],"Landmark identity mismatch")
    labels,num=ndimage.label(crop,structure=ndimage.generate_binary_structure(3,1))
    counts=np.bincount(labels.ravel())[1:]
    upper=landmark_evidence(spec,labels,origin,data,a)
    sections=[] if spec is None else spec.get("sections",[])
    require(isinstance(sections,list) and len(sections)<=32,"Maximum 32 requested sections")
    measurements=[section(data,a,item) for item in sections]
    pos,normals,faces,transform,quality=reconstruct(crop,origin,a)
    glb=make_glb(pos,normals,faces);mesh_sha=digest(glb)
    flags=["Unreviewed airway lumen mask surface; no inferred anatomical labels.",
           "Zero padding closes observed mask boundaries; caps are not measured tissue.",
           "Landmark connectivity is not anatomical continuity validation."]
    if touches: flags.append("Foreground contacts source FOV boundary; surface may contain artificial caps.")
    slope=float(getattr(ct.dataobj,"slope",1));intercept=float(getattr(ct.dataobj,"inter",0))
    require(np.isfinite(slope) and np.isfinite(intercept),"Invalid CT calibration")
    report={"schema":"pulmosimu.anatomy-fidelity/1","ct_sha256":m["ct_sha256"],"mask_sha256":m["mask_sha256"],"mesh_sha256":mesh_sha,
            "structure":"airway_lumen","review":"PENDING","clinical_navigation_allowed":False,
            "component_count":int(num),"foreground_voxels":count,"largest_component_fraction":float(counts.max()/count),
            "voxel_volume_ml":count*abs(float(np.linalg.det(a[:3,:3])))/1000,"fov_boundary_faces":touches,
            "surface":quality,"upper_airway":upper,"sections":measurements,
            "landmark_input_sha256":sha(landmark_path) if landmark_path else None}
    manifest={"source_sha256":m["ct_sha256"],"mesh_sha256":mesh_sha,"mask_sha256":m["mask_sha256"],
              "source_shape_vox":list(ct.shape),"sform_RAS_mm":a.tolist(),"spacing_mm":sp.tolist(),
              "HU_stored_slope":slope,"HU_stored_intercept":intercept,"model_from_ras_mm":transform.tolist(),
              "validation":"UNREVIEWED_SOURCE_MASK_SURFACE","research_flags":flags,"structures":{}}
    out.mkdir(parents=True)
    (out/"airway.glb").write_bytes(glb);write_json(out/"reconstruction.json",manifest);write_json(out/"fidelity.json",report)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("ct");p.add_argument("mask_dir");p.add_argument("out");p.add_argument("--landmarks")
    args=p.parse_args();print(json.dumps(run(args.ct,args.mask_dir,args.out,args.landmarks),indent=2,allow_nan=False))
