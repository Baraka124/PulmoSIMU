#!/usr/bin/env python3
"""PulmoSIMU A2 — source NIfTI/GLB voxel-space audit (research only).

Requires numpy, trimesh, vtk. Run only with the candidate manifest and matching
local CT/GLB files. Neither model is edited; patient-specific clinical validation
and airway-procedure registration are explicitly blocked.

python a2_ct_surface_voxel_qa.py --source image.nii \
 --mesh A2_candidate.glb --manifest a2-candidate-manifest.json --out qa.json
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


def run(source_path, mesh_path, manifest_path, out_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sha256(source_path) != manifest["source_sha256"]:
        raise ValueError("Source CT SHA-256 mismatch")
    if sha256(mesh_path) != manifest["mesh_sha256"]:
        raise ValueError("Candidate GLB SHA-256 mismatch")
    with open(source_path, "rb") as f:
        header = f.read(352)
    if struct.unpack_from("<I", header, 0)[0] != 348:
        raise ValueError("NIfTI-1 header required")
    if struct.unpack_from("<h", header, 254)[0] < 1:
        raise ValueError("No explicit NIfTI sform")
    dims = list(struct.unpack_from("<3h", header, 42))
    if dims != manifest["source_shape_vox"]:
        raise ValueError("CT shape mismatch")
    if struct.unpack_from("<h", header, 70)[0] != 512:
        raise ValueError("This source audit expects uint16 NIfTI-1")
    offset = int(struct.unpack_from("<f", header, 108)[0])
    slope, intercept = struct.unpack_from("<2f", header, 112)
    if (abs(slope - manifest["HU_stored_slope"]) > 1e-3 or
        abs(intercept - manifest["HU_stored_intercept"]) > 1e-3):
        raise ValueError("CT scalar calibration mismatch")
    affine = np.asarray([
        struct.unpack_from("<4f", header, i) for i in (280, 296, 312)
    ] + [(0, 0, 0, 1)], dtype=float)
    if not np.allclose(affine, manifest["sform_RAS_mm"], atol=1e-4):
        raise ValueError("CT sform transform mismatch")

    voxel_size_l = abs(float(np.linalg.det(affine[:3, :3]))) / 1e6
    ct = np.memmap(source_path, dtype="<u2", mode="r", offset=offset,
                   shape=(dims[2], dims[1], dims[0]))
    transform = np.linalg.inv(affine) @ np.linalg.inv(
        np.asarray(manifest["model_from_ras_mm"], dtype=float)
    )
    scene = trimesh.load(str(mesh_path), force="scene")
    measurements, masks = [], {}
    for node in sorted(scene.graph.nodes_geometry):
        M, geom = scene.graph[node]
        surface = scene.geometry[geom].copy()
        surface.apply_transform(M)
        side = "left" if "patient_left" in node else (
            "right" if "patient_right" in node else None
        )
        if side is None:
            raise ValueError("Candidate source mesh has no explicit patient RAS side")
        mask = voxelize(surface, transform, dims)
        masks[side] = mask
        values = np.asarray(ct[mask], dtype=np.int32) * slope + intercept
        x_ras = affine[0, 0] * np.arange(dims[0]) + affine[0, 3]
        # NIfTI RAS+: +X is patient RIGHT; opposite of left is +X.\n        opposite_indices = np.flatnonzero(x_ras > 0 if side == "left" else x_ras < 0)
        opposite_voxels = int(mask[:, :, opposite_indices].sum())
        n_voxels = int(mask.sum())
        if n_voxels == 0:
            raise ValueError("Candidate mesh produced no CT voxels")
        measurements.append(dict(
            node=node, side=side, voxels=n_voxels,
            voxelized_airspace_volume_l=n_voxels * voxel_size_l,
            mesh_volume_l=abs(float(surface.volume)) * 1000,
            inside_mesh_HU_threshold_fraction=float(((values > -1010) & (values < -400)).mean()),
            inside_mesh_HU_mean=float(values.mean()),
            HU_quantiles_5_50_95=[float(x) for x in np.quantile(values, [.05, .5, .95])],
            voxels_on_opposite_RAS_side=opposite_voxels,
            volume_fraction_on_opposite_side=opposite_voxels / n_voxels,
            independent_anatomy_review="REQUIRED", clinically_validated=False
        ))
    if set(masks) != {"left", "right"}:
        raise ValueError("Both patient left/right candidate surfaces are required")
    overlap = int(np.count_nonzero(masks["left"] & masks["right"]))
    union = int(np.count_nonzero(masks["left"] | masks["right"]))
    result = dict(
        protocol="PulmoSIMU-A2-candidate-voxel-audit-v1",
        source_sha256=manifest["source_sha256"],
        candidate_mesh_sha256=manifest["mesh_sha256"],
        voxel_grid=dims, voxel_spacing_mm=manifest["spacing_mm"],
        voxelization_method="CT IJK stencil with NIfTI sform and model_from_ras_mm",
        measurements=measurements, overlap_voxels=overlap,
        overlap_volume_ml=overlap * voxel_size_l * 1000,
        overlap_fraction_of_union=overlap / union,
        threshold_method="inside HU agreement from a threshold-derived mesh; NOT independent ground truth",
        mask_approval="FAILED_PENDING_CLINICAL_REVIEW",
        bronchoscope_to_patient_CT="BLOCKED",
        vascular_and_five_lobe_anatomy="NOT_AVAILABLE",
        patient_specific_physics="BLOCKED"
    )
    Path(out_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("source", "mesh", "manifest", "out"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    outcome = run(args.source, args.mesh, args.manifest, args.out)
    print(json.dumps(outcome, indent=2))
