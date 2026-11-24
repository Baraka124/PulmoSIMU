#!/usr/bin/env python3
"""PulmoSIMU A2.1 — source-orientated NLST pulmonary airspace segmentation.

Produces a research-only, bilateral CT-derived airspace candidate, NIfTI-affine
model surfaces, slice overlays, masks and JSON quality checks. It does NOT
segment clinically validated lungs/lobes/airways/vessels and must never be
attached to the procedural bronchoscope without independent registration.

Dependencies: numpy, scipy, scikit-image, trimesh, matplotlib.
Run:
 python experimental/anatomy/a2_1_correct_nlst_airspace.py \
   --nifti "NLST_Chest_CT_150_Slices_R21(1).nii" --out a2_review
No medical images are uploaded; all processing is local.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.measure import marching_cubes
import trimesh

SOURCE_SHA256 = "5b234ac92a9a5d58d15f3f43144d31612d10ff6e60777d8c273cd52b36b13087"
ALGORITHM = "A2.1-conservative-hemithorax-3d-connected-airspace-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def load_nifti(path, downsample=2):
    with Path(path).open("rb") as f:
        header = f.read(352)
    if len(header) != 352 or struct.unpack_from("<I", header)[0] != 348:
        raise ValueError("Expected uncompressed NIfTI-1 (little endian)")
    dim = struct.unpack_from("<8h", header, 40)
    if dim[:4] != (3, 512, 512, 150):
        raise ValueError("Unexpected source dimensions; this code is study-specific")
    if struct.unpack_from("<h", header, 70)[0] != 512:
        raise ValueError("Expected unsigned 16-bit source; no implicit type conversion")
    slope, intercept = struct.unpack_from("<2f", header, 112)
    if (slope, intercept) != (1., -1024.):
        raise ValueError("HU scale/intercept mismatch")
    if struct.unpack_from("<h", header, 254)[0] <= 0:
        raise ValueError("Missing source sform: coordinate mapping cannot be inferred")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.array([struct.unpack_from("<4f", header, pos)
                              for pos in (280, 296, 312)])
    if abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError("Singular source affine")
    if np.any(np.abs(affine[0, 1:3]) > 1e-5):
        raise ValueError("Oblique patient L/R requires verified image reorientation")
    offset = int(struct.unpack_from("<f", header, 108)[0])
    raw = np.memmap(path, dtype="<u2", mode="r", offset=offset,
                    shape=(512, 512, 150), order="F")
    hu = np.asarray(raw[::downsample, ::downsample, :], dtype=np.int16) - 1024
    return hu, affine


def masks_from_hu(hu, affine, downsample=2):
    """Research *airspace*, not full lung parenchyma.

    2D body enclosure; HU airspace candidate; one mid-thorax seeded 6-connected
    3D component per physical RAS hemisphere. Explicit midline partition is
    conservative and may remove genuine medial lung: expert review required.
    """
    candidate = np.zeros(hu.shape, bool)
    for z in range(hu.shape[2]):
        plane = hu[:, :, z]
        torso = ndi.binary_fill_holes(
            ndi.binary_closing(plane > -450,
                               structure=disk(max(2, round(10 / downsample)))))
        candidate[:, :, z] = (plane > -1015) & (plane < -400) & torso
    ras_x = affine[0, 0] * (np.arange(hu.shape[0]) * downsample) + affine[0, 3]
    labels = np.zeros(hu.shape, np.uint8)
    stats = {}
    for name, code, selected_rows in (
        ("right", 1, ras_x >= 0), ("left", 2, ras_x < 0)
    ):
        air = candidate[selected_rows, :, :]
        components, n = ndi.label(air, ndi.generate_binary_structure(3, 1))
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        on_seed, seed_counts = np.unique(
            components[:, :, hu.shape[2] // 2], return_counts=True
        )
        eligible = [(int(sizes[k]), int(k))
                    for k, count in zip(on_seed, seed_counts)
                    if k > 0 and count >= 50 and sizes[k] >= 10000]
        if not eligible:
            raise ValueError("No seeded lung airspace: " + name)
        volume, component_id = max(eligible)
        labels[selected_rows, :, :] = np.where(
            components == component_id, code, 0
        ).astype(np.uint8)
        zs = np.flatnonzero((components == component_id).any(axis=(0, 1)))
        stats[name] = {
            "selected_voxels": volume,
            "all_components": int(n),
            "excluded_candidates_voxels": int(air.sum() - volume),
            "selected_label_id": component_id,
            "slice_range": [int(zs[0]), int(zs[-1])]
        }
    assert np.all(labels[ras_x < 0] != 1)
    assert np.all(labels[ras_x >= 0] != 2)
    assert labels.sum() == stats["right"]["selected_voxels"] + 2 * stats["left"]["selected_voxels"]
    return labels, ras_x, stats


def source_mesh(binary, affine, xy_step=2):
    """Source-affine, closed high-detail mesh. No fabricated bronchial walls."""
    pad = np.pad(binary.astype(np.float32), 1)
    smooth = ndi.gaussian_filter(pad, sigma=(.9, .9, .5))
    v, f, _, _ = marching_cubes(smooth, level=.5)
    vox = v - 1
    vox[:, :2] *= xy_step
    ras = np.c_[vox, np.ones(len(vox))] @ affine.T
    local = np.c_[ras[:, 0] * .001,
                  ras[:, 2] * .001 + .390,
                 -ras[:, 1] * .001 + .122]
    raw = trimesh.Trimesh(vertices=local, faces=f, process=False)
    shells = raw.split(only_watertight=True)
    if not shells:
        raise RuntimeError("Surface is not watertight: fail source geometry gate")
    mesh = max(shells, key=lambda x: len(x.faces))
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_watertight or mesh.volume <= 0:
        raise RuntimeError("Failed closed/oriented 3D surface QA")
    # Deliberately NO blind quadric decimation. Earlier low-poly attempts
    # corrupted surface topology, so Quest export remains BLOCKED.
    return mesh


def plot_review(hu, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs = (25, 35, 45, 60, 75, 95, 115, 135)
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.5))
    for ax, z in zip(axes.flat, zs):
        ax.imshow(hu[:, :, z].T, cmap="gray", vmin=-950, vmax=250)
        for code, color in ((1, "#f1b263"), (2, "#70d2d6")):
            ax.contour((labels[:, :, z] == code).T, levels=[.5],
                       colors=[color], linewidths=.8)
        ax.set_title(f"Axial NIfTI k={z}")
        ax.axis("off")
    fig.suptitle("PulmoSIMU A2.1 — source CT airspace contour review", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(nifti, out, downsample=2):
    out.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(nifti)
    if source_hash != SOURCE_SHA256:
        raise ValueError("Source SHA-256 differs from verified A1 NLST NIfTI")
    hu, affine = load_nifti(nifti, downsample)
    labels, ras_x, components = masks_from_hu(hu, affine, downsample)
    step_mm = np.linalg.norm(affine[:3, :3], axis=0) * [downsample, downsample, 1]
    voxel_mm3 = float(np.prod(step_mm))
    scene = trimesh.Scene()
    geometry = {}
    for name, code in (("right", 1), ("left", 2)):
        mesh = source_mesh(labels == code, affine, downsample)
        scene.add_geometry(mesh, geom_name=f"a2_{name}_reference_airspace")
        geometry[name] = {
            "triangles": len(mesh.faces), "vertices": len(mesh.vertices),
            "watertight": bool(mesh.is_watertight),
            "quest_25k_triangle_budget_met": len(mesh.faces) <= 25000,
            "airspace_volume_l": round(int(np.count_nonzero(labels == code))
                                       * voxel_mm3 / 1_000_000, 5),
            "model_x_m": mesh.bounds[:, 0].tolist()
        }
    glb = out / "PulmoSIMU_A2_1_NLST_Corrected_Airspace_Research.glb"
    scene.export(glb)
    maskfile = out / "PulmoSIMU_A2_1_Airspace_Mask_QA.npz"
    np.savez_compressed(maskfile, mask=labels, affine_ras=affine,
                        downsample_xy=np.array(downsample))
    review = out / "PulmoSIMU_A2_1_CT_Contours.png"
    plot_review(hu, labels, review)
    report = {
        "schema": "pulmosimu.a2.1.qa.v1", "algorithm": ALGORITHM,
        "source_sha256": source_hash, "source_shape": [512, 512, 150],
        "sform_RAS_mm": affine.tolist(),
        "downsample_xy": downsample, "spacing_mm": step_mm.tolist(),
        "HU_threshold_open_interval": [-1015, -400],
        "body_threshold_HU": -450, "RAS_midline_mm": 0,
        "components": components, "surfaces": geometry,
        "mesh_sha256": sha256(glb), "mask_sha256": sha256(maskfile),
        "preview_sha256": sha256(review),
        "scientific_status": {
            "bilateral_component_integrity": "PASS_BY_CONSERVATIVE_PARTITION",
            "surface_watertightness": "PASS",
            "expert_CT_review": "PENDING",
            "full_lung_boundaries": "NOT_VALIDATED",
            "fissures_and_lobes": "NOT_SEGMENTED",
            "pulmonary_vessels": "NOT_SEGMENTED",
            "airway_lumen_and_centreline": "NOT_SEGMENTED",
            "Quest_mesh": "BLOCKED_UNTIL_TOPOLOGY_PRESERVING_DECIMATION",
            "CT_linked_bronchoscopy": "BLOCKED"
        },
        "limitations": [
            "Airspace is not total pulmonary parenchyma.",
            "Midline partition removes observed A1 contralateral leakage but may clip medial lung.",
            "The largest component may include connected bronchial air.",
            "No reference masks: Dice/Hausdorff and clinical validity cannot be claimed.",
            "2 mm slice spacing limits thin anatomy.",
            "Never substitute this model into the original procedural VR engine without QA."
        ]
    }
    (out / "PulmoSIMU_A2_1_Segmentation_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"surfaces": geometry, "status": report["scientific_status"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--downsample", type=int, choices=(1, 2, 4), default=2)
    p = ap.parse_args()
    main(p.nifti, p.out, p.downsample)
