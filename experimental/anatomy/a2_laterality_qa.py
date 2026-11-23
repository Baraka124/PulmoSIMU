#!/usr/bin/env python3
"""A2 source-RAS/mesh laterality quality gate for a provisional CT GLB.

Offline research CLI, not patient-specific diagnosis. Requires pip install
numpy trimesh. Never modifies source anatomy or automatically swaps names.

Example:
  python experimental/anatomy/a2_laterality_qa.py \
    --source study.nii --mesh surface.glb \
    --manifest experimental/data/nlst-a1/study-manifest.json \
    --out a2-qa-report.json --report-only

Without --report-only this exits status 2 on an anatomical laterality conflict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import trimesh


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source_sform(path: Path):
    with path.open("rb") as source:
        header = source.read(352)
    if struct.unpack_from("<i", header, 0)[0] != 348 or header[344:347] != b"n+1":
        raise ValueError("A1 supports uncompressed little-endian NIfTI-1 only")
    shape = list(struct.unpack_from("<8h", header, 40)[1:4])
    sform_code = struct.unpack_from("<h", header, 254)[0]
    if sform_code < 1:
        raise ValueError("Cannot verify anatomical laterality without NIfTI RAS sform")
    affine = np.array([
        struct.unpack_from("<4f", header, offset) for offset in (280, 296, 312)
    ] + [(0, 0, 0, 1)], dtype=float)
    return shape, affine


def audit(ct: Path, glb: Path, manifest: Path) -> dict:
    catalog = json.loads(manifest.read_text(encoding="utf-8"))
    for role, path in (("source", ct), ("mesh", glb)):
        expected = catalog.get(f"{role}_sha256", "").lower()
        if not expected or digest(path) != expected:
            raise ValueError(f"{role} digest does not match the source manifest")
    dims, affine = source_sform(ct)
    if dims != catalog["source_shape_vox"] or not np.allclose(
        affine, np.asarray(catalog["sform_RAS_mm"], dtype=float), atol=1e-4
    ):
        raise ValueError("CT header and manifest disagree")
    transform = np.asarray(catalog["model_from_ras_mm"], dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Missing 4x4 RAS-to-model transform")
    ras_from_scene = np.linalg.inv(transform)
    scene = trimesh.load(str(glb), force="scene")
    surfaces = []
    for node in scene.graph.nodes_geometry:
        matrix, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(matrix)
        ras = trimesh.transform_points(np.asarray(mesh.vertices), ras_from_scene)
        areas = np.asarray(mesh.area_faces)
        signed = ras[np.asarray(mesh.faces), 0].mean(axis=1)
        # NIfTI RAS+: +X is PATIENT RIGHT, -X is PATIENT LEFT.
        # See the NIfTI-1 coordinate standard and 3D Slicer RAS documentation.
        pos = float(areas[signed > 0].sum() / areas.sum())
        name = "right" if "right" in node.lower() else (
            "left" if "left" in node.lower() else "unknown"
        )
        wrong = (1 - pos) if name == "right" else (pos if name == "left" else None)
        conflict = wrong is not None and wrong > 0.8
        leakage_review = wrong is not None and wrong > 0.10
        surfaces.append(dict(
            source_node=node, unverified_label=name,
            face_count=len(mesh.faces), watertight=bool(mesh.is_watertight),
            area_fraction_RAS_patient_left=1 - pos,
            area_fraction_RAS_patient_right=pos,
            area_fraction_RAS_x_positive=pos,
            area_fraction_opposite_named_side=wrong,
            area_weighted_center_RAS_x_mm=float(
                (signed * areas).sum() / areas.sum()
            ),
            source_laterality_conflict=bool(conflict),
            midline_crossing_requires_review=bool(leakage_review),
        ))
    if sorted([x["unverified_label"] for x in surfaces]) != ["left", "right"]:
        raise ValueError("Expected exactly one right-labelled and left-labelled mesh")
    failures = [x["source_node"] for x in surfaces if x["source_laterality_conflict"]]
    return dict(
        protocol="PulmoSIMU-A2-source-geometry-v1",
        sha256_verified=True,
        source=ct.name,
        mesh=glb.name,
        laterality="NIfTI sform RAS: +X PATIENT RIGHT, -X PATIENT LEFT",
        anatomy_review="FAIL" if failures else
            ("REVIEW_REQUIRED" if any(x["midline_crossing_requires_review"] for x in surfaces)
             else "NOT_INDEPENDENTLY_VALIDATED"),
        critical_laterality_conflicts=failures,
        supersedes="earlier A2 RAS sign-inverted laterality swap audit",
        original_node_names_preserved=True,
        surfaces=surfaces,
        bronchoscopy_linkage="BLOCKED",
        validated_lobes=False, validated_airways=False,
        clinically_validated=False,
        limitations="A watertight GLB with matching file hashes is not a validated lung/airway segmentation.",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    report = audit(args.source, args.mesh, args.manifest)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Anatomy review: {report['anatomy_review']}")
    for item in report["surfaces"]:
        print(f"{item['unverified_label']} mesh: "
              f"{item['area_fraction_opposite_named_side']:.1%} "
              "surface area on opposite RAS patient side")
    if report["critical_laterality_conflicts"] and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
