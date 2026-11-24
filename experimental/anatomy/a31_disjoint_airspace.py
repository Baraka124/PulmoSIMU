#!/usr/bin/env python3
"""A3.1 CT source disjoint bilateral research masks, no clinical or airway validation.
Install numpy scipy scikit-image vtk trimesh. Keep CT and binaries local.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import (
    numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def voxelize(mesh, mesh_to_ijk, dims):
    vox = trimesh.transform_points(mesh.vertices, mesh_to_ijk)
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.asarray(vox, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    n = len(mesh.faces)
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * n + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(mesh.faces, dtype=np.int64).ravel(), deep=True)
    )
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    st = vtk.vtkPolyDataToImageStencil()
    st.SetInputData(poly)
    st.SetOutputSpacing(1, 1, 1)
    st.SetOutputOrigin(0, 0, 0)
    st.SetOutputWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    st.Update()
    image = vtk.vtkImageStencilToImage()
    image.SetInputConnection(st.GetOutputPort())
    image.SetInsideValue(1)
    image.SetOutsideValue(0)
    image.SetOutputScalarTypeToUnsignedChar()
    image.Update()
    return vtk_to_numpy(image.GetOutput().GetPointData().GetScalars()).reshape(
        dims[2], dims[1], dims[0]
    ).astype(bool)



from scipy import ndimage as ndi
from skimage import measure

def disjoin(left,right,spacing_kji):
    ambiguous=left&right
    out=np.zeros(left.shape,dtype=np.uint8)
    out[left&~right]=1
    out[right&~left]=2
    count=int(ambiguous.sum())
    if count:
        coords=np.where(ambiguous)
        lo=np.maximum(0,np.asarray([a.min() for a in coords])-32)
        hi=np.minimum(left.shape,np.asarray([a.max()+1 for a in coords])+32)
        sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        only_l=left[sl]&~right[sl]
        only_r=right[sl]&~left[sl]
        if not only_l.any() or not only_r.any():
            raise ValueError("Overlap lacks side-exclusive reference seeds")
        dl=ndi.distance_transform_edt(~only_l,sampling=spacing_kji)
        dr=ndi.distance_transform_edt(~only_r,sampling=spacing_kji)
        target=out[sl];hit=ambiguous[sl]
        target[hit]=np.where(dl[hit]<=dr[hit],1,2)
        out[sl]=target
    if not np.array_equal(out>0,left|right):
        raise ValueError("Refusing an output that changed the bilateral union")
    return out,count

def source_mesh(labels,code,ijk_to_model,budget=45000):
    arr=np.pad((labels==code).astype(np.uint8),1)
    xyz,tri,_,_=measure.marching_cubes(arr,.5,allow_degenerate=False)
    ijk=np.column_stack((xyz[:,2]-1,xyz[:,1]-1,xyz[:,0]-1))
    positions=trimesh.transform_points(ijk,ijk_to_model)
    points=vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(positions,dtype=np.float64),deep=True))
    faces=np.asarray(tri,dtype=np.int64)
    cells=vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces)+1,dtype=np.int64)*3,deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces).ravel(),deep=True))
    poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    dec=vtk.vtkDecimatePro();dec.SetInputData(poly)
    dec.SetTargetReduction(max(0,1-budget/max(1,len(faces))))
    dec.PreserveTopologyOn();dec.BoundaryVertexDeletionOff();dec.SplittingOff()
    dec.Update()
    output=dec.GetOutput()
    mesh=trimesh.Trimesh(
        vertices=vtk_to_numpy(output.GetPoints().GetData()).copy(),
        faces=vtk_to_numpy(output.GetPolys().GetConnectivityArray()).reshape(-1,3).copy(),
        process=False
    )
    if not mesh.is_watertight:
        raise ValueError("Non-watertight research surface; refusing to export")
    return mesh,len(faces)

def run(ct,source_glb,manifest_file,outdir):
    ct,source_glb,manifest_file,outdir=map(Path,(ct,source_glb,manifest_file,outdir))
    outdir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(manifest_file.read_text())
    if sha256(ct)!=manifest["source_sha256"] or sha256(source_glb)!=manifest["mesh_sha256"]:
        raise ValueError("Source CT/GLB checksum mismatch")
    with open(ct,"rb") as f:h=f.read(352)
    if struct.unpack_from("<I",h,0)[0]!=348 or struct.unpack_from("<h",h,254)[0]<1:
        raise ValueError("Source NIfTI header/sform missing")
    dims=list(struct.unpack_from("<3h",h,42))
    affine=np.asarray([struct.unpack_from("<4f",h,i) for i in (280,296,312)]+[(0,0,0,1)])
    if dims!=manifest["source_shape_vox"] or not np.allclose(affine,manifest["sform_RAS_mm"],atol=1e-4):
        raise ValueError("Source CT RAS coordinate mismatch")
    model_from_ijk=np.asarray(manifest["model_from_ras_mm"])@affine
    ijk_from_model=np.linalg.inv(model_from_ijk)
    scene=trimesh.load(str(source_glb),force="scene")
    sides={}
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node]
        mesh=scene.geometry[name].copy();mesh.apply_transform(transform)
        side="left" if "patient_left" in node else "right" if "patient_right" in node else None
        if side is None or side in sides:
            raise ValueError("Expected one patient-left and patient-right source surface")
        sides[side]=voxelize(mesh,ijk_from_model,dims)
    if set(sides)!={"left","right"}:raise ValueError("Both source surfaces required")
    spacing=np.linalg.norm(affine[:3,:3],axis=0)[::-1]
    labels,overlap=disjoin(sides["left"],sides["right"],spacing)
    label_file=outdir/"PulmoSIMU_A31_Disjoint_Source_Masks.npz"
    np.savez_compressed(label_file,labels=labels,affine_ras=affine,
        label_names=np.array(["background","candidate_patient_left","candidate_patient_right"]))
    study=trimesh.Scene();statistics={}
    for side,code in (("left",1),("right",2)):
        mesh,initial=source_mesh(labels,code,model_from_ijk)
        name=f"A31_candidate_patient_{side}_airspace_UNVALIDATED"
        mesh.visual.vertex_colors=np.tile(([153,177,187,220] if code==1 else [202,166,165,220]),(len(mesh.vertices),1))
        study.add_geometry(mesh,geom_name=name,node_name=name)
        statistics[side]={"source_voxels":int(sides[side].sum()),
            "candidate_voxels":int((labels==code).sum()),
            "candidate_faces":len(mesh.faces),"faces_before_decimation":int(initial),
            "mesh_watertight":bool(mesh.is_watertight)}
    glb_file=outdir/"PulmoSIMU_A31_Disjoint_Airspace_Candidate.glb"
    study.export(str(glb_file))
    voxel_l=abs(float(np.linalg.det(affine[:3,:3])))/1e6
    report=dict(protocol="PulmoSIMU-A3.1-Voxel-Disjoint-Research-v1",
        source_sha256=manifest["source_sha256"],source_glb_sha256=manifest["mesh_sha256"],
        candidate_mesh_sha256=sha256(glb_file),candidate_labelmap_sha256=sha256(label_file),
        source_overlap_voxels=overlap,source_overlap_ml=overlap*voxel_l*1000,
        output_overlap_voxels=0,source_bilateral_union_preserved=True,
        surfaces=statistics,independent_anatomy_review="PENDING",
        residual_midline_leakage="UNRESOLVED",five_lobes="MISSING",
        airway_lumen="MISSING",pulmonary_vessels="MISSING",
        CT_linked_bronchoscopy="BLOCKED",clinical_validation=False)
    (outdir/"PulmoSIMU_A31_Disjoint_QA.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for field in ("ct","mesh","manifest","out"):
        ap.add_argument("--"+field,type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(run(args.ct,args.mesh,args.manifest,args.out),indent=2))
