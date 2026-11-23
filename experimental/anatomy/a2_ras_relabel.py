#!/usr/bin/env python3
"""PulmoSIMU A2 — versioned anatomical laterality relabelling.

Source CT, original GLB and A1 study-manifest.json must match SHA-256.
Corrects only the source GLB's misleading surface names using RAS x coordinates.
Never changes original vertex positions, faces, image voxels or material colors.
The derived model remains unvalidated for imaging/clinical use.

pip install numpy trimesh
python a2_ras_relabel.py --source image.nii --mesh original.glb \
 --manifest study-manifest.json --output candidate.glb --qa candidate-qa.json
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import trimesh


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nifti_affine(path):
    with open(path, "rb") as f:
        header = f.read(352)
    if struct.unpack_from("<I", header, 0)[0] != 348:
        raise ValueError("NIfTI-1 header expected")
    if struct.unpack_from("<h", header, 254)[0] < 1:
        raise ValueError("NIfTI sform is missing; cannot establish RAS laterality")
    return np.asarray([struct.unpack_from("<4f", header, i) for i in (280, 296, 312)]
                      + [(0, 0, 0, 1)], dtype=float)


def run(source_path, mesh_path, manifest_path, output_path, qa_path):
    source_path, mesh_path = Path(source_path), Path(mesh_path)
    qa = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sha256(source_path) != qa["source_sha256"]:
        raise ValueError("Source CT SHA-256 does not match signed study manifest")
    if sha256(mesh_path) != qa["mesh_sha256"]:
        raise ValueError("Original GLB SHA-256 does not match study manifest")
    if not np.allclose(nifti_affine(source_path),
                       np.asarray(qa["sform_RAS_mm"]), atol=0.0001):
        raise ValueError("CT NIfTI sform disagrees with manifest")

    inverse = np.linalg.inv(np.asarray(qa["model_from_ras_mm"], dtype=float))
    source = trimesh.load(str(mesh_path), force="scene")
    if len(source.graph.nodes_geometry) != 2:
        raise ValueError("Exactly two source surfaces required")
    output = trimesh.Scene()
    findings = []
    for node in sorted(source.graph.nodes_geometry):
        transform, name = source.graph[node]
        mesh = source.geometry[name].copy()
        mesh.apply_transform(transform)
        ras_points = trimesh.transform_points(mesh.vertices, inverse)
        face_ras_x = ras_points[mesh.faces, 0].mean(axis=1)
        areas = mesh.area_faces
        patient_left_fraction = float(areas[face_ras_x > 0].sum() / areas.sum())
        if max(patient_left_fraction, 1 - patient_left_fraction) < 0.8:
            raise ValueError("Ambiguous anatomical side; refusing automatic name change")
        side = "left" if patient_left_fraction > .5 else "right"
        label = f"A2_candidate_patient_{side}_lung_airspace_UNVALIDATED"
        output.add_geometry(mesh, geom_name=label, node_name=label)
        opposite = (1 - patient_left_fraction) if side == "left" else patient_left_fraction
        findings.append(dict(original_node=node, candidate_node=label,
                             candidate_patient_side=side,
                             RAS_area_fraction_patient_left=patient_left_fraction,
                             opposite_side_surface_area_fraction=opposite,
                             cross_midline_leakage_review_required=opposite > .05,
                             faces=len(mesh.faces), vertices=len(mesh.vertices)))
    if sorted(item["candidate_patient_side"] for item in findings) != ["left", "right"]:
        raise ValueError("Detected duplicate anatomical sides")

    output.export(str(output_path))
    reloaded = trimesh.load(str(output_path), force="scene")
    for item in findings:
        matrix, name = reloaded.graph[item["candidate_node"]]
        mesh = reloaded.geometry[name].copy()
        mesh.apply_transform(matrix)
        matrix0, name0 = source.graph[item["original_node"]]
        original = source.geometry[name0].copy()
        original.apply_transform(matrix0)
        if (not np.array_equal(mesh.vertices, original.vertices)
                or not np.array_equal(mesh.faces, original.faces)):
            raise ValueError("Source topology or vertex locations changed in candidate")

    report = dict(protocol="PulmoSIMU-A2-source-RAS-relabel-v2",
                  source_sha256=qa["source_sha256"],
                  original_mesh_sha256=qa["mesh_sha256"],
                  candidate_mesh_sha256=sha256(output_path),
                  source_geometry_original_unchanged=True, findings=findings,
                  surface_names_corrected_candidate=True,
                  segmentation_anatomical_validation="NOT_VALIDATED",
                  lobe_boundaries="NOT_AVAILABLE",
                  bronchial_lumen="NOT_AVAILABLE",
                  vascular_structures="NOT_AVAILABLE",
                  cross_midline_leakage="REVIEW_REQUIRED",
                  CT_linked_bronchoscopy="BLOCKED",
                  independent_clinician_review="PENDING",
                  clinical_use=False)
    Path(qa_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--qa", required=True)
    args = p.parse_args()
    result = run(args.source, args.mesh, args.manifest, args.output, args.qa)
    print("Candidate GLB SHA-256:", result["candidate_mesh_sha256"])
    for finding in result["findings"]:
        print(finding["candidate_node"],
              f'{finding["opposite_side_surface_area_fraction"]:.1%} area opposite side')
