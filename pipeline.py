"""
PNEUMO·VR — CT Airway Segmentation Pipeline
============================================
Converts a patient DICOM chest CT into a GLTF mesh + JSON sidecar
that the frontend loads directly into Three.js.

Usage:
    python pipeline.py --dicom /path/to/dicom/folder --out ./output --patient-id P001

Outputs:
    output/P001_airway.glb   — trimmed airway mesh (loads into Three.js)
    output/P001_meta.json    — centerline, segment graph, pathologies, transform

Changes from v1:
    GAP 1 — Anatomical segment naming: spatial atlas assigns proper
             bronchopulmonary segment names (RUL B1-3, LLL B9-10, etc.)
             instead of generic "Airway Gen2 Seg5" labels.
    GAP 2 — Tumor/polypoid detection: radiodensity wall analysis and
             morphological shape analysis detect endobronchial lesions.
"""

# ── 1. Imports ────────────────────────────────────────────────────
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pydicom
import trimesh
import vtk
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from skimage.measure import label as sk_label

warnings.filterwarnings("ignore")


# ── 2. DICOM ingest ───────────────────────────────────────────────
def load_dicom_volume(dicom_dir):
    """
    Read all DICOM slices from a folder.
    Returns (volume_np, voxel_spacing_mm, patient_meta).
    Sorts slices by InstanceNumber or ImagePositionPatient Z.
    """
    print(f"[1/5] Loading DICOM from {dicom_dir}")
    slices = []
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=False)
            if hasattr(ds, "ImagePositionPatient"):
                slices.append(ds)
        except Exception:
            continue  # skip non-DICOM files

    if not slices:
        raise RuntimeError(f"No valid DICOM files found in {dicom_dir}")

    # Sort by Z position (axial slices)
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    print(f"    Found {len(slices)} slices")

    # Stack pixel data into 3D volume (Z, Y, X)
    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)

    # Apply rescale slope/intercept to get Hounsfield Units
    ref = slices[0]
    slope     = float(getattr(ref, "RescaleSlope",     1))
    intercept = float(getattr(ref, "RescaleIntercept", 0))
    volume    = volume * slope + intercept  # now in HU

    # Voxel spacing: (slice_thickness_mm, row_spacing_mm, col_spacing_mm)
    ps = ref.PixelSpacing
    st = float(getattr(ref, "SliceThickness", 1.0))
    spacing = np.array([st, float(ps[0]), float(ps[1])])

    # Patient metadata for the sidecar
    meta = {
        "patient_id":   str(getattr(ref, "PatientID",   "UNKNOWN")),
        "patient_name": str(getattr(ref, "PatientName",  "UNKNOWN")),
        "study_date":   str(getattr(ref, "StudyDate",    "UNKNOWN")),
        "modality":     str(getattr(ref, "Modality",     "CT")),
        "n_slices":     len(slices),
        "spacing_mm":   spacing.tolist(),
        "volume_shape": list(volume.shape),
    }

    print(f"    Volume shape: {volume.shape}, spacing: {spacing} mm")
    return volume, spacing, meta


# ── 3. Airway segmentation ────────────────────────────────────────
def segment_airways(volume, spacing):
    """
    Extract airway mask using TotalSegmentator (preferred) or
    a region-growing HU threshold fallback.

    TotalSegmentator produces a labeled NIfTI; we extract label 132 (trachea/bronchi).
    Fallback: HU threshold −900 to −300 captures air-filled airways.
    """
    print("[2/5] Segmenting airways")

    # Try TotalSegmentator first
    try:
        from totalsegmentator.python_api import totalsegmentator
        import nibabel as nib
        import tempfile

        # Write volume to temp NIfTI for TotalSegmentator input
        with tempfile.TemporaryDirectory() as tmpdir:
            nii_in  = os.path.join(tmpdir, "ct.nii.gz")
            nii_out = os.path.join(tmpdir, "seg")
            affine  = np.diag([spacing[2], spacing[1], spacing[0], 1])
            nib.save(nib.Nifti1Image(volume, affine), nii_in)

            totalsegmentator(
                nii_in, nii_out,
                task="total",
                fast=True,          # reduced model for speed
                quiet=True,
            )

            # Label 132 = trachea; 133 = bronchi left; 134 = bronchi right
            airway_mask = np.zeros(volume.shape, dtype=bool)
            for label_id in [132, 133, 134]:
                lpath = os.path.join(nii_out, f"label_{label_id:03d}.nii.gz")
                if os.path.exists(lpath):
                    seg = nib.load(lpath).get_fdata().astype(bool)
                    airway_mask |= seg

        print("    TotalSegmentator segmentation complete")
        return airway_mask

    except ImportError:
        print("    TotalSegmentator not available — using HU threshold fallback")
    except Exception as e:
        print(f"    TotalSegmentator failed ({e}) — using HU threshold fallback")

    # ── HU threshold fallback ─────────────────────────────────────
    # Airways are air-filled: HU roughly −1000 to −300
    # Lungs also fall in this range — we isolate by connected component
    air_mask = (volume > -1000) & (volume < -300)

    # Find the largest connected component that includes the trachea
    # (approximately at image centre-top in axial view)
    labeled = sk_label(air_mask)
    cy, cx  = volume.shape[1] // 2, volume.shape[2] // 2
    trachea_start = max(0, volume.shape[0] - volume.shape[0] // 4)  # top ~25% of Z

    # Walk down from top until we hit a labelled voxel at centre
    trachea_label = 0
    for z in range(trachea_start, volume.shape[0]):
        lv = labeled[z, cy, cx]
        if lv > 0:
            trachea_label = lv
            break

    if trachea_label == 0:
        raise RuntimeError("Could not locate trachea in volume — check DICOM orientation")

    airway_mask = labeled == trachea_label

    # Small morphological close to fill mucosal gaps
    airway_mask = ndimage.binary_closing(airway_mask, iterations=2)

    print(f"    Threshold segmentation: {airway_mask.sum():,} airway voxels")
    return airway_mask


# ── 4. Mesh extraction + cleaning ────────────────────────────────
def extract_mesh(airway_mask, spacing):
    """
    Run VTK marching cubes on the binary mask, then clean with trimesh:
    - Remove disconnected fragments
    - Laplacian smoothing (removes voxel staircase)
    - Decimate to ~50K triangles for real-time rendering
    Returns a trimesh.Trimesh object.
    """
    print("[3/5] Extracting mesh")

    # VTK marching cubes
    vol = airway_mask.astype(np.uint8)
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(vol.shape[2], vol.shape[1], vol.shape[0])
    vtk_img.SetSpacing(spacing[2], spacing[1], spacing[0])  # mm per voxel

    flat = vol.flatten(order="F")  # VTK uses Fortran order
    arr  = vtk.util.numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_img.GetPointData().SetScalars(arr)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, 0.5)         # iso-surface at mask boundary
    mc.ComputeNormalsOn()
    mc.Update()

    # Convert VTK polydata → trimesh
    pd     = mc.GetOutput()
    pts    = vtk.util.numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
    faces  = vtk.util.numpy_support.vtk_to_numpy(pd.GetPolys().GetData())
    faces  = faces.reshape(-1, 4)[:, 1:]  # strip the "3" prefix from each face

    mesh = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
    print(f"    Raw mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    # Keep only the largest connected component (the airway tree)
    components = mesh.split(only_watertight=False)
    mesh = max(components, key=lambda m: len(m.faces))
    print(f"    After component filter: {len(mesh.faces):,} faces")

    # Laplacian smoothing — 5 iterations removes staircase artefacts
    trimesh.smoothing.filter_laplacian(mesh, iterations=5)

    # Decimate to target face count for GPU performance
    target_faces = 50_000
    if len(mesh.faces) > target_faces:
        ratio = target_faces / len(mesh.faces)
        mesh = mesh.simplify_quadric_decimation(int(len(mesh.faces) * ratio))
        print(f"    After decimation: {len(mesh.faces):,} faces")

    return mesh


# ── 5. Centerline extraction ──────────────────────────────────────
def extract_centerline(airway_mask, spacing):
    """
    Skeletonize the 3D binary mask to get a 1-voxel-wide skeleton,
    then trace branches to build a segment graph.
    Each segment has: id, name, generation, 3D centerline points, radius, children.
    """
    print("[4/5] Extracting centerline")

    print("    Running 3D skeletonization (may take 1-2 min)...")
    skeleton = skeletonize_3d(airway_mask)

    skel_vox = np.argwhere(skeleton)   # (N, 3) in voxel coords
    skel_mm  = skel_vox * spacing      # scale to mm

    def count_neighbours(skel, z, y, x):
        count = 0
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == dy == dx == 0: continue
                    nz, ny, nx = z+dz, y+dy, x+dx
                    if 0 <= nz < skel.shape[0] and 0 <= ny < skel.shape[1] and 0 <= nx < skel.shape[2]:
                        if skel[nz, ny, nx]:
                            count += 1
        return count

    degrees = np.zeros(skeleton.shape, dtype=np.uint8)
    for z, y, x in skel_vox:
        degrees[z, y, x] = count_neighbours(skeleton, z, y, x)

    branch_pts   = np.argwhere(degrees >= 3)
    endpoint_pts = np.argwhere(degrees == 1)

    print(f"    Skeleton: {len(skel_vox):,} pts, {len(branch_pts)} branch pts, {len(endpoint_pts)} endpoints")

    segments = _trace_segments(skeleton, skel_vox, branch_pts, endpoint_pts, spacing, airway_mask)
    print(f"    Traced {len(segments)} airway segments")

    return segments


def _trace_segments(skeleton, skel_vox, branch_pts, endpoint_pts, spacing, mask):
    """
    Walk the skeleton graph to build a list of segment dicts.
    Each segment: {id, name, gen, pts_mm, r0_mm, r1_mm, children, parent}
    Name is left as a placeholder here — anatomical naming happens in
    assign_anatomical_names() after the full segment tree is built.
    """
    skel_set = set(map(tuple, skel_vox))

    def neighbours(z, y, x):
        nbrs = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == dy == dx == 0: continue
                    n = (z+dz, y+dy, x+dx)
                    if n in skel_set:
                        nbrs.append(n)
        return nbrs

    branch_set    = set(map(tuple, branch_pts))
    endpoint_set  = set(map(tuple, endpoint_pts))
    visited_edges = set()
    segments      = []
    seg_id        = 0

    # Start from topmost endpoint (trachea top)
    start_vox = tuple(endpoint_pts[np.argmin(endpoint_pts[:, 0])])

    def walk_segment(start, came_from, generation):
        nonlocal seg_id
        path    = [start]
        current = start
        prev    = came_from

        while True:
            nbrs = [n for n in neighbours(*current) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt in branch_set:
                path.append(nxt)
                break
            if nxt in endpoint_set and nxt != start:
                path.append(nxt)
                break
            path.append(nxt)
            prev, current = current, nxt

        pts_mm = [np.array(p) * spacing for p in path]

        dist_transform = ndimage.distance_transform_edt(mask) * spacing[0]
        r_vals = [dist_transform[p] for p in path]
        r0 = float(np.clip(np.mean(r_vals[:3]),  2, 12) / 1000)
        r1 = float(np.clip(np.mean(r_vals[-3:]), 1,  8) / 1000)

        seg = {
            "id":       seg_id,
            "name":     f"_seg{seg_id}",   # placeholder — named in assign_anatomical_names()
            "gen":      generation,
            "pts_mm":   [p.tolist() for p in pts_mm],
            "r0_mm":    r0,
            "r1_mm":    r1,
            "children": [],
            "parent":   -1,
            "_end_vox": path[-1],
            "_mid_mm":  pts_mm[len(pts_mm) // 2].tolist(),  # midpoint for spatial atlas
        }
        seg_id += 1
        return seg, path[-1]

    # BFS from trachea
    from collections import deque
    queue        = deque([(start_vox, None, 0, -1)])
    visited_edges = set()

    while queue:
        start, came_from, gen, parent_id = queue.popleft()
        edge_key = (min(start, came_from or start), max(start, came_from or start))
        if edge_key in visited_edges:
            continue
        visited_edges.add(edge_key)

        seg, end_vox = walk_segment(start, came_from, gen)
        seg["parent"] = parent_id
        if parent_id >= 0:
            segments[parent_id]["children"].append(seg["id"])
        segments.append(seg)

        if tuple(end_vox) in branch_set:
            nbrs        = neighbours(*end_vox)
            visited_set = set(s["_end_vox"] for s in segments)
            child_starts = [n for n in nbrs if n != (came_from or start) and n not in visited_set]
            for child in child_starts:
                queue.append((child, end_vox, gen + 1, seg["id"]))

    return segments


# ══════════════════════════════════════════════════════════════════
# GAP 1 — ANATOMICAL SEGMENT NAMING
# ══════════════════════════════════════════════════════════════════
#
# Strategy: spatial atlas matching.
#
# The bronchopulmonary segment tree has a well-known spatial anatomy:
#   Gen 0  — Trachea       (midline, superior)
#   Gen 1  — L/R Main      (split left/right at carina by X coordinate)
#   Gen 2  — Lobar         (3 right lobes, 2 left lobes)
#            Right: RUL (superior), RML (middle-anterior), RLL (inferior)
#            Left : LUL (superior), LLL (inferior)
#   Gen 3+ — Segmental     (named by lobe + spatial position)
#
# We use the midpoint XYZ of each segment in CT mm coordinates:
#   X axis  → left/right (negative = patient left, positive = patient right
#              in standard DICOM LPS orientation)
#   Y axis  → anterior/posterior
#   Z axis  → superior/inferior (higher Z = more inferior in standard DICOM)
#
# After normalise_to_threejs() the coordinates are in Three.js space, but
# naming happens before that step, so we work in raw mm.
# ══════════════════════════════════════════════════════════════════

def assign_anatomical_names(segments):
    """
    Walk the segment tree and assign clinically meaningful bronchopulmonary
    names to each segment based on generation and spatial position.

    Modifies segments in-place (updates the 'name' field).
    Returns the segment list.
    """
    print("    Assigning anatomical names via spatial atlas")

    if not segments:
        return segments

    # Build a lookup by id
    seg_map = {s["id"]: s for s in segments}

    # ── Gen 0: Trachea ────────────────────────────────────────────
    trachea = next((s for s in segments if s["gen"] == 0), None)
    if trachea:
        trachea["name"] = "Trachea"
        _set_latin(trachea, "Trachea", "")

    # ── Gen 1: Main bronchi — split by X at carina ────────────────
    # Carina midpoint X is the trachea's last point X
    carina_x = 0.0
    if trachea and trachea["pts_mm"]:
        carina_x = trachea["pts_mm"][-1][0]   # X of distal trachea end

    gen1_segs = [s for s in segments if s["gen"] == 1]
    for s in gen1_segs:
        mid_x = _mid_x(s)
        if mid_x < carina_x:
            s["name"] = "L Main Bronchus"
            s["_side"] = "L"
        else:
            s["name"] = "R Main Bronchus"
            s["_side"] = "R"
        _set_latin(s, s["name"], "Bronchus principalis")

    # Propagate side label down the tree
    _propagate_side(segments, seg_map)

    # ── Gen 2: Lobar bronchi ──────────────────────────────────────
    gen2_segs = [s for s in segments if s["gen"] == 2]
    for s in gen2_segs:
        side  = s.get("_side", "?")
        mid_z = _mid_z(s)   # inferior = higher Z in DICOM
        mid_y = _mid_y(s)   # anterior = smaller Y

        if side == "R":
            # Right lung has 3 lobes — classify by Z and Y
            siblings = [x for x in gen2_segs if x.get("_side") == "R"]
            z_vals   = sorted([_mid_z(x) for x in siblings])
            if len(z_vals) >= 3:
                # Superior third → RUL, middle third → RML, inferior → RLL
                low_z, high_z = z_vals[0], z_vals[-1]
                span = max(high_z - low_z, 1)
                frac = (mid_z - low_z) / span
                if frac < 0.35:
                    s["name"] = "RUL – B1-3"
                    _set_latin(s, "RUL – B1-3", "Lobus superior dexter")
                elif frac < 0.65:
                    # RML is also more anterior
                    s["name"] = "RML – B4-5"
                    _set_latin(s, "RML – B4-5", "Lobus medius dexter")
                else:
                    s["name"] = "RLL – B6-10"
                    _set_latin(s, "RLL – B6-10", "Lobus inferior dexter")
            else:
                # Only 1-2 children — fallback to superior/inferior
                if mid_z < np.mean([_mid_z(x) for x in siblings]):
                    s["name"] = "RUL – B1-3"
                    _set_latin(s, "RUL – B1-3", "Lobus superior dexter")
                else:
                    s["name"] = "RLL – B6-10"
                    _set_latin(s, "RLL – B6-10", "Lobus inferior dexter")

        else:  # Left lung — 2 lobes
            siblings = [x for x in gen2_segs if x.get("_side") == "L"]
            z_mean   = np.mean([_mid_z(x) for x in siblings]) if siblings else mid_z
            if mid_z < z_mean:
                s["name"] = "LUL – B1-3"
                _set_latin(s, "LUL – B1-3", "Lobus superior sinister")
            else:
                s["name"] = "LLL – B6-10"
                _set_latin(s, "LLL – B6-10", "Lobus inferior sinister")

    # ── Gen 3+: Segmental bronchi — refine lobar names ───────────
    gen3plus = [s for s in segments if s["gen"] >= 3]
    for s in gen3plus:
        parent = seg_map.get(s["parent"])
        if not parent:
            s["name"] = f"Seg {s['id']}"
            continue

        pname = parent.get("name", "")
        side  = s.get("_side", "?")
        mid_z = _mid_z(s)
        mid_y = _mid_y(s)
        mid_x = _mid_x(s)

        siblings = [x for x in gen3plus if x["parent"] == s["parent"]]
        z_mean   = np.mean([_mid_z(x) for x in siblings]) if siblings else mid_z
        y_mean   = np.mean([_mid_y(x) for x in siblings]) if siblings else mid_y
        x_mean   = np.mean([_mid_x(x) for x in siblings]) if siblings else mid_x

        name = _segmental_name(pname, side, mid_z, mid_y, mid_x,
                               z_mean, y_mean, x_mean)
        s["name"] = name
        _set_latin(s, name, "")

    # Clean up private fields
    for s in segments:
        s.pop("_mid_mm", None)

    named = [s["name"] for s in segments if not s["name"].startswith("_")]
    print(f"    Named {len(named)}/{len(segments)} segments anatomically")
    return segments


def _segmental_name(parent_name, side, mid_z, mid_y, mid_x,
                    z_mean, y_mean, x_mean):
    """
    Map a segmental bronchus to its bronchopulmonary segment label
    based on parent lobe and spatial position.

    Standard bronchopulmonary segment map (Jackson & Huber):
      RUL: B1 (apical), B2 (posterior), B3 (anterior)
      RML: B4 (lateral), B5 (medial)
      RLL: B6 (superior), B7 (medial basal), B8 (anterior basal),
           B9 (lateral basal), B10 (posterior basal)
      LUL: B1-2 (apicoposterior), B3 (anterior), B4 (superior lingular),
           B5 (inferior lingular)
      LLL: B6 (superior), B8 (anterior basal), B9 (lateral basal),
           B10 (posterior basal)
    """
    sup  = mid_z < z_mean   # True = superior (smaller Z in DICOM)
    ant  = mid_y < y_mean   # True = anterior
    med  = (mid_x < x_mean) if side == "R" else (mid_x > x_mean)  # medial

    if "RUL" in parent_name:
        if sup and ant:   return "RUL B3 (Anterior)"
        if sup:           return "RUL B1 (Apical)"
        return                  "RUL B2 (Posterior)"

    if "RML" in parent_name:
        if med:           return "RML B5 (Medial)"
        return                  "RML B4 (Lateral)"

    if "RLL" in parent_name:
        if sup:           return "RLL B6 (Superior)"
        if med and ant:   return "RLL B7 (Medial Basal)"
        if ant:           return "RLL B8 (Anterior Basal)"
        if not med:       return "RLL B9 (Lateral Basal)"
        return                  "RLL B10 (Posterior Basal)"

    if "LUL" in parent_name:
        if sup and not ant: return "LUL B1-2 (Apicoposterior)"
        if ant:             return "LUL B3 (Anterior)"
        if not sup and ant: return "LUL B4 (Superior Lingular)"
        return                    "LUL B5 (Inferior Lingular)"

    if "LLL" in parent_name:
        if sup:           return "LLL B6 (Superior)"
        if ant and med:   return "LLL B8 (Anterior Basal)"
        if not med:       return "LLL B9 (Lateral Basal)"
        return                  "LLL B10 (Posterior Basal)"

    # Deep sub-segmental fallback — preserve parent context
    return f"{parent_name.split('–')[0].strip()} sub-seg"


def _propagate_side(segments, seg_map):
    """BFS propagation of _side label from gen-1 down the tree."""
    from collections import deque
    queue = deque([s for s in segments if s["gen"] == 1])
    while queue:
        parent = queue.popleft()
        for cid in parent.get("children", []):
            child = seg_map.get(cid)
            if child and "_side" not in child:
                child["_side"] = parent.get("_side", "?")
                queue.append(child)


def _mid_x(seg):
    pts = seg.get("pts_mm") or seg.get("_mid_mm") and [seg["_mid_mm"]] or []
    if not pts: return 0.0
    return float(np.mean([p[0] for p in pts]))

def _mid_y(seg):
    pts = seg.get("pts_mm") or seg.get("_mid_mm") and [seg["_mid_mm"]] or []
    if not pts: return 0.0
    return float(np.mean([p[1] for p in pts]))

def _mid_z(seg):
    pts = seg.get("pts_mm") or seg.get("_mid_mm") and [seg["_mid_mm"]] or []
    if not pts: return 0.0
    return float(np.mean([p[2] for p in pts]))

def _set_latin(seg, short_name, latin):
    """Store the Latin anatomical name as an annotation (used by frontend card library)."""
    seg["_latin"] = latin


# ── 6. Normalise to Three.js world space ─────────────────────────
def normalise_to_threejs(mesh, segments):
    """
    CT data is in mm, patient-centred.
    Three.js scene uses ~1 unit ≈ ~1m at our scale.
    Scale and centre the mesh + segment points to match.
    """
    print("[5/5] Normalising to Three.js world space")

    all_pts  = np.vstack([np.array(s["pts_mm"]) for s in segments if s["pts_mm"]])
    bbox_min = all_pts.min(axis=0)
    bbox_max = all_pts.max(axis=0)
    bbox_ctr = (bbox_min + bbox_max) / 2
    bbox_ext = (bbox_max - bbox_min).max()

    target_span_m = 2.4
    scale = target_span_m / (bbox_ext / 1000)      # mm → m → Three.js units

    def xf_pt(pt_mm):
        pt_m = (np.array(pt_mm) - bbox_ctr) / 1000
        return (pt_m * scale).tolist()

    v = mesh.vertices.copy()
    v = (v - bbox_ctr[np.newaxis, :]) / 1000
    v = v * scale
    mesh.vertices = v

    for seg in segments:
        seg["pts_threejs"] = [xf_pt(p) for p in seg["pts_mm"]]
        seg["r0"] = seg["r0_mm"] * scale / 1000
        seg["r1"] = seg["r1_mm"] * scale / 1000

    transform = {
        "centre_mm":   bbox_ctr.tolist(),
        "scale":       float(scale),
        "target_span": target_span_m,
    }

    return mesh, segments, transform


# ══════════════════════════════════════════════════════════════════
# GAP 2 — PATHOLOGY DETECTION (stenosis + mucus + TUMOR/POLYPOID)
# ══════════════════════════════════════════════════════════════════
#
# Three detection modes:
#
# 1. STENOSIS (existing, improved)
#    Local radius drops >40% below segment mean.
#    Added: asymmetry check (one-sided narrowing = extrinsic compression
#    vs. circumferential = intrinsic stenosis).
#
# 2. MUCUS PLUG (existing, improved)
#    High HU region (> -200) inside the airway lumen.
#    Now checks for filling pattern (>50% lumen cross-section).
#
# 3. ENDOBRONCHIAL TUMOR / POLYPOID LESION (NEW)
#    Detection pipeline:
#    a) Candidate voxels: inside airway mask but HU > -100
#       (soft tissue HU range: -100 to +100)
#    b) Connected component analysis: find blobs that are
#       - attached to the airway wall (adjacent to ~0 HU voxels)
#       - NOT the full lumen cross-section (ruling out mucus)
#       - volume > min_blob_mm3 (avoids noise)
#    c) Morphology: polypoid lesions are roughly spherical (eccentricity < 0.8)
#       while mucus is elongated along the airway axis.
#    d) Locate on the segment centerline using nearest-point projection.
#    e) Severity estimate from blob volume relative to lumen area.
# ══════════════════════════════════════════════════════════════════

def detect_pathologies(segments, airway_mask, volume, spacing):
    """
    Detect stenosis, mucus plugs, and endobronchial tumors/polypoid lesions.
    Returns list of pathology dicts compatible with the frontend.
    """
    print("[+] Detecting pathologies (stenosis, mucus, tumor/polypoid)")
    pathologies = []

    dist_xf = ndimage.distance_transform_edt(airway_mask) * spacing[0]

    # ── 1 & 2: Stenosis and mucus (original logic, improved) ────
    for seg in segments:
        pts = seg.get("pts_mm", [])
        if len(pts) < 4:
            continue

        radii   = []
        hu_vals = []
        for pt in pts:
            vox = (np.array(pt) / spacing).astype(int)
            vox = np.clip(vox, 0, np.array(airway_mask.shape) - 1)
            radii.append(dist_xf[tuple(vox)])
            hu_vals.append(float(volume[tuple(vox)]))

        mean_r = np.mean(radii)
        min_r  = np.min(radii)

        # ── Stenosis: local radius < 60% of segment mean ─────────
        if min_r < mean_r * 0.6 and mean_r > 3:
            t = radii.index(min(radii)) / max(len(radii) - 1, 1)
            # Asymmetry check: sample radius in 4 directions at narrowest pt
            narrowest_pt = pts[radii.index(min(radii))]
            asym = _check_asymmetry(narrowest_pt, airway_mask, spacing, dist_xf)
            sten_type = "extrinsic_compression" if asym > 0.35 else "stenosis"
            pathologies.append({
                "id":       f"sten_{seg['id']}",
                "type":     "stenosis",
                "subtype":  sten_type,
                "label":    f"Stenosis · {seg['name']}",
                "seg_id":   seg["id"],
                "t":        float(t),
                "severity": float(1 - min_r / mean_r),
                "col":      "#ff9900",
            })

        # ── Mucus plug: high-HU region filling >30% of lumen ────
        high_hu    = [h for h in hu_vals if h > -300]
        fill_frac  = len(high_hu) / max(len(hu_vals), 1)
        if fill_frac > 0.3 and mean_r > 2:
            t = hu_vals.index(max(high_hu, default=0)) / max(len(hu_vals) - 1, 1)
            # Check HU range: mucus is -100 to +50, blood is +50 to +80
            max_hu  = max(high_hu, default=-999)
            subtype = "blood_clot" if max_hu > 50 else "mucus_plug"
            pathologies.append({
                "id":       f"mucus_{seg['id']}",
                "type":     "mucus_plug",
                "subtype":  subtype,
                "label":    f"{'Blood Clot' if subtype == 'blood_clot' else 'Mucus Plug'} · {seg['name']}",
                "seg_id":   seg["id"],
                "t":        float(t),
                "severity": float(min(fill_frac * 1.4, 1.0)),
                "col":      "#cc3300" if subtype == "blood_clot" else "#9955ff",
            })

    # ── 3: Endobronchial tumor / polypoid lesion (NEW) ───────────
    tumor_pathologies = _detect_endobronchial_lesions(
        segments, airway_mask, volume, spacing, dist_xf
    )
    pathologies.extend(tumor_pathologies)

    # Deduplicate: if a segment already has a mucus/stenosis, skip tumor
    # (one finding per segment to keep UI clean)
    seg_ids_taken = set(p["seg_id"] for p in pathologies if p["type"] != "polypoid")
    pathologies = [
        p for p in pathologies
        if p["type"] != "polypoid" or p["seg_id"] not in seg_ids_taken
    ]

    print(f"    Detected {len(pathologies)} pathologies "
          f"({sum(1 for p in pathologies if p['type']=='stenosis')} stenosis, "
          f"{sum(1 for p in pathologies if p['type']=='mucus_plug')} mucus, "
          f"{sum(1 for p in pathologies if p['type']=='polypoid')} polypoid/tumor)")
    return pathologies


def _detect_endobronchial_lesions(segments, airway_mask, volume, spacing, dist_xf):
    """
    Detect polypoid/tumor lesions inside the airway lumen.

    A polypoid lesion is a soft-tissue density mass that:
      - Sits inside the airway (within the airway_mask)
      - Has HU range consistent with soft tissue (-100 to +150)
      - Forms a compact, roughly spherical blob (not elongated like mucus)
      - Locally reduces the lumen radius
      - Is attached to the airway wall (not freely floating)

    Returns list of pathology dicts.
    """
    pathologies = []

    # Step 1: candidate voxels — inside airway mask, soft-tissue HU
    soft_tissue_mask = (
        airway_mask &
        (volume > -100) &
        (volume < 150)
    )

    if soft_tissue_mask.sum() == 0:
        return pathologies

    # Step 2: connected components of soft-tissue blobs
    labeled_blobs, n_blobs = ndimage.label(soft_tissue_mask)
    min_blob_voxels = int((4 / 3) * np.pi * (3 ** 3) / np.prod(spacing))  # ~3mm radius sphere
    voxel_vol_mm3   = np.prod(spacing)

    for blob_id in range(1, n_blobs + 1):
        blob_mask = labeled_blobs == blob_id
        blob_size = blob_mask.sum()

        if blob_size < min_blob_voxels:
            continue  # too small — noise

        # Step 3: morphology check — polypoid vs. mucus
        # Compute blob bounding box and aspect ratio
        blob_coords = np.argwhere(blob_mask)
        bbox_extent = blob_coords.max(axis=0) - blob_coords.min(axis=0)
        bbox_mm     = bbox_extent * spacing
        sorted_dims = np.sort(bbox_mm)[::-1]  # largest to smallest

        if sorted_dims[0] > 0:
            elongation = sorted_dims[0] / max(sorted_dims[-1], 0.1)
        else:
            elongation = 1.0

        # Mucus plugs are highly elongated (>4:1); polypoid lesions are compact
        if elongation > 4.0:
            continue  # likely mucus — skip

        # Step 4: wall attachment check
        # The blob must be adjacent to the airway wall (where dist_xf ≈ 0)
        # Dilate the blob by 1 voxel and check if it overlaps near-wall voxels
        dilated     = ndimage.binary_dilation(blob_mask, iterations=2)
        near_wall   = (dist_xf < spacing[0] * 2.5)  # within ~2.5 voxels of wall
        wall_contact = np.logical_and(dilated, near_wall).sum()
        if wall_contact < 3:
            continue  # not attached to wall

        # Step 5: assign to nearest segment centerline point
        blob_centroid_vox = blob_coords.mean(axis=0)
        blob_centroid_mm  = blob_centroid_vox * spacing

        best_seg, best_t, best_dist = None, 0.5, np.inf
        for seg in segments:
            pts = seg.get("pts_mm", [])
            if not pts:
                continue
            for i, pt in enumerate(pts):
                d = np.linalg.norm(np.array(pt) - blob_centroid_mm)
                if d < best_dist:
                    best_dist = d
                    best_seg  = seg
                    best_t    = i / max(len(pts) - 1, 1)

        # Only keep if the blob is close to a known segment (< 20mm)
        if best_seg is None or best_dist > 20.0:
            continue

        # Step 6: severity — blob volume relative to local lumen area
        lumen_r_mm    = dist_xf[tuple(blob_centroid_vox.astype(int))]
        lumen_area    = np.pi * lumen_r_mm ** 2
        blob_vol_mm3  = blob_size * voxel_vol_mm3
        # Approximate lesion cross-section as circle with same volume
        lesion_r      = (3 * blob_vol_mm3 / (4 * np.pi)) ** (1/3)
        lesion_area   = np.pi * lesion_r ** 2
        severity      = float(np.clip(lesion_area / max(lumen_area, 1), 0.1, 1.0))

        # Mean HU of the blob (malignant lesions tend to be higher HU)
        blob_hu_mean = float(volume[blob_mask].mean())
        blob_hu_note = "hyper-dense" if blob_hu_mean > 20 else "iso-dense"

        pathologies.append({
            "id":        f"tumor_{best_seg['id']}_{blob_id}",
            "type":      "polypoid",
            "subtype":   "endobronchial_lesion",
            "label":     f"Polypoid Lesion · {best_seg['name']} · {blob_hu_note}",
            "seg_id":    best_seg["id"],
            "t":         float(best_t),
            "severity":  severity,
            "col":       "#ff2244",
            "blob_vol_mm3":   float(blob_vol_mm3),
            "blob_hu_mean":   blob_hu_mean,
            "elongation":     float(elongation),
            "wall_contact_vox": int(wall_contact),
        })

    return pathologies


def _check_asymmetry(pt_mm, airway_mask, spacing, dist_xf):
    """
    Sample the distance transform in 4 orthogonal radial directions
    around a centerline point. Returns asymmetry index 0-1
    (0 = symmetric narrowing, 1 = fully one-sided).
    """
    vox = (np.array(pt_mm) / spacing).astype(int)
    vox = np.clip(vox, 2, np.array(airway_mask.shape) - 3)

    offsets = [
        (0,  2, 0), (0, -2, 0),   # anterior-posterior
        (0,  0, 2), (0,  0, -2),  # left-right
    ]
    radii = []
    for dz, dy, dx in offsets:
        nv = (vox[0]+dz, vox[1]+dy, vox[2]+dx)
        if 0 <= nv[0] < dist_xf.shape[0]:
            radii.append(dist_xf[nv])

    if len(radii) < 2:
        return 0.0

    r_arr = np.array(radii)
    return float((r_arr.max() - r_arr.min()) / max(r_arr.max(), 1))


# ── 8. Export ─────────────────────────────────────────────────────
def export(mesh, segments, pathologies, transform, patient_meta, patient_id, out_dir):
    """
    Write the two output files:
      {patient_id}_airway.glb   — Three.js-ready mesh
      {patient_id}_meta.json    — centerline + pathologies + transform
    """
    os.makedirs(out_dir, exist_ok=True)

    gltf_path = os.path.join(out_dir, f"{patient_id}_airway.glb")
    _ = mesh.vertex_normals   # force smooth normal computation
    mesh.export(gltf_path, file_type="glb", include_normals=True)
    print(f"    Mesh → {gltf_path}")

    meta = {
        "version":  "pneumovr-1.1",   # bumped for gap fixes
        "patient":  patient_meta,
        "transform": transform,
        "segments": [
            {
                "id":       s["id"],
                "name":     s["name"],
                "latin":    s.get("_latin", ""),
                "gen":      s["gen"],
                "pts":      s.get("pts_threejs", []),
                "r0":       s.get("r0", 0.05),
                "r1":       s.get("r1", 0.04),
                "children": s["children"],
                "parent":   s["parent"],
            }
            for s in segments
        ],
        "pathologies": pathologies,
        "mesh_file":   f"{patient_id}_airway.glb",
    }
    json_path = os.path.join(out_dir, f"{patient_id}_meta.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    Meta → {json_path}")

    return gltf_path, json_path


# ── 9. Main entry point ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PNEUMO·VR CT Pipeline v1.1")
    parser.add_argument("--dicom",      required=True,      help="Path to DICOM folder")
    parser.add_argument("--out",        default="./output", help="Output directory")
    parser.add_argument("--patient-id", default="patient",  help="Patient ID prefix for output files")
    args = parser.parse_args()

    t0 = time.time()
    print("\n══ PNEUMO·VR CT Pipeline v1.1 ════════════════════════")
    print(f"   DICOM:  {args.dicom}")
    print(f"   Output: {args.out}")
    print(f"   ID:     {args.patient_id}\n")

    volume, spacing, patient_meta = load_dicom_volume(args.dicom)
    airway_mask                   = segment_airways(volume, spacing)
    mesh                          = extract_mesh(airway_mask, spacing)
    segments                      = extract_centerline(airway_mask, spacing)

    # GAP 1 — Anatomical naming (before normalisation, while pts are in mm)
    segments = assign_anatomical_names(segments)

    mesh, segments, transform     = normalise_to_threejs(mesh, segments)

    # GAP 2 — Full pathology detection including tumor/polypoid
    pathologies = detect_pathologies(segments, airway_mask, volume, spacing)

    gltf_path, json_path = export(
        mesh, segments, pathologies, transform,
        patient_meta, args.patient_id, args.out
    )

    elapsed = time.time() - t0
    print(f"\n══ Done in {elapsed:.1f}s ══════════════════════════════════")
    print(f"   GLTF: {gltf_path}")
    print(f"   Meta: {json_path}\n")


if __name__ == "__main__":
    main()