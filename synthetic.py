"""
PNEUMO·VR — Synthetic Patient Generator
========================================
Generates a test GLTF + JSON sidecar from the existing procedural
bronchial tree — identical format to pipeline.py output.
Use this to test the frontend loader before real CT data exists.

Usage:
    python synthetic.py --out ./output --patient-id SYNTHETIC_001

No dependencies on TotalSegmentator or pydicom.
Requires: numpy, trimesh

v1.1 changes:
    - Segment names now use the full anatomical atlas from pipeline.py
      (e.g. "LLL B10 (Posterior Basal)" instead of "LLL B9-10")
    - Pathology labels updated to match new names
    - version bumped to pneumovr-1.1 to match pipeline.py output
"""

# ── 1. Imports ────────────────────────────────────────────────────
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import trimesh


# ── 2. Bronchial tree definition ──────────────────────────────────
# Mirrors the SEGS array from the frontend exactly.
# pts are in Three.js world units (approx metres).
# Names follow the full anatomical atlas from pipeline.py v1.1
# (Jackson & Huber bronchopulmonary segment nomenclature).

SEGS = [
    {"id":0,  "name":"Trachea",         "latin":"Trachea",                   "gen":0, "children":[1,4],
     "pts":[[0,1.2,.13],[0,1.0,.12],[0,.8,.12],[0,.6,.12],[0,.4,.12],[0,.32,.12]],
     "r0":.098, "r1":.090, "parent":-1},

    {"id":1,  "name":"L Main Bronchus", "latin":"Bronchus principalis",       "gen":1, "children":[2,3],
     "pts":[[0,.32,.12],[-.20,.24,.11],[-.42,.16,.10],[-.64,.09,.09],[-.80,.05,.08]],
     "r0":.082, "r1":.068, "parent":0},

    {"id":2,  "name":"LUL – B1-3",      "latin":"Lobus superior sinister",    "gen":2, "children":[20,21],
     "pts":[[-.80,.05,.08],[-.77,.17,.12],[-.70,.30,.14],[-.62,.44,.16],[-.55,.54,.17]],
     "r0":.054, "r1":.038, "parent":1,
     "pathology":{"id":"tumor","type":"polypoid","label":"Polypoid Lesion · LUL B1-3 · iso-dense",
                  "t":0.62,"col":"#ff2244","severity":0.72}},

    {"id":3,  "name":"LLL – B6-10",     "latin":"Lobus inferior sinister",    "gen":2, "children":[22,23],
     "pts":[[-.80,.05,.08],[-.88,-.07,.07],[-.96,-.20,.06],[-1.04,-.34,.05],[-1.08,-.48,.04]],
     "r0":.056, "r1":.040, "parent":1,
     "pathology":{"id":"mucus","type":"mucus_plug","label":"Mucus Plug · LLL B10 (Posterior Basal)",
                  "t":0.68,"col":"#9955ff","severity":0.58}},

    {"id":4,  "name":"R Main Bronchus", "latin":"Bronchus principalis",       "gen":1, "children":[5,6,7],
     "pts":[[0,.32,.12],[.20,.24,.11],[.44,.16,.10],[.66,.09,.09],[.82,.05,.08]],
     "r0":.086, "r1":.072, "parent":0},

    {"id":5,  "name":"RUL – B1-3",      "latin":"Lobus superior dexter",      "gen":2, "children":[24,25],
     "pts":[[.82,.05,.08],[.79,.17,.12],[.72,.30,.14],[.65,.43,.16],[.59,.53,.17]],
     "r0":.055, "r1":.040, "parent":4},

    {"id":6,  "name":"RML – B4-5",      "latin":"Lobus medius dexter",        "gen":2, "children":[26,27],
     "pts":[[.82,.05,.08],[.89,.02,.18],[.93,-.02,.28],[.96,-.08,.34]],
     "r0":.044, "r1":.030, "parent":4,
     "pathology":{"id":"sten","type":"stenosis","label":"Stenosis · RML B4-5",
                  "t":0.52,"col":"#ff9900","severity":0.65}},

    {"id":7,  "name":"RLL – B6-10",     "latin":"Lobus inferior dexter",      "gen":2, "children":[28,29],
     "pts":[[.82,.05,.08],[.90,-.10,.07],[.98,-.23,.05],[1.06,-.37,.04],[1.10,-.51,.02]],
     "r0":.058, "r1":.042, "parent":4},

    # Generation 3 — segmental bronchi (Jackson & Huber nomenclature)
    {"id":20, "name":"LUL B1-2 (Apicoposterior)", "latin":"", "gen":3, "children":[],
     "pts":[[-.55,.54,.17],[-.52,.64,.20],[-.50,.72,.22],[-.48,.78,.23]],
     "r0":.032, "r1":.022, "parent":2},
    {"id":21, "name":"LUL B3 (Anterior)",          "latin":"", "gen":3, "children":[],
     "pts":[[-.55,.54,.17],[-.62,.60,.22],[-.68,.65,.25],[-.72,.68,.26]],
     "r0":.030, "r1":.020, "parent":2},
    {"id":22, "name":"LLL B9 (Lateral Basal)",      "latin":"", "gen":3, "children":[],
     "pts":[[-1.08,-.48,.04],[-1.10,-.56,.03],[-1.12,-.64,.02],[-1.13,-.70,.02]],
     "r0":.030, "r1":.022, "parent":3},
    {"id":23, "name":"LLL B10 (Posterior Basal)",   "latin":"", "gen":3, "children":[],
     "pts":[[-1.08,-.48,.04],[-1.13,-.53,.10],[-1.16,-.58,.16],[-1.17,-.62,.20]],
     "r0":.028, "r1":.020, "parent":3},
    {"id":24, "name":"RUL B1 (Apical)",             "latin":"", "gen":3, "children":[],
     "pts":[[.59,.53,.17],[.56,.63,.20],[.54,.71,.22],[.52,.77,.23]],
     "r0":.030, "r1":.022, "parent":5},
    {"id":25, "name":"RUL B3 (Anterior)",           "latin":"", "gen":3, "children":[],
     "pts":[[.59,.53,.17],[.66,.59,.22],[.72,.64,.25],[.76,.67,.26]],
     "r0":.028, "r1":.020, "parent":5},
    {"id":26, "name":"RML B4 (Lateral)",            "latin":"", "gen":3, "children":[],
     "pts":[[.96,-.08,.34],[.99,-.12,.40],[1.01,-.17,.44],[1.02,-.22,.47]],
     "r0":.022, "r1":.015, "parent":6},
    {"id":27, "name":"RML B5 (Medial)",             "latin":"", "gen":3, "children":[],
     "pts":[[.96,-.08,.34],[1.00,-.08,.40],[1.04,-.09,.44],[1.06,-.10,.47]],
     "r0":.020, "r1":.014, "parent":6},
    {"id":28, "name":"RLL B9 (Lateral Basal)",      "latin":"", "gen":3, "children":[],
     "pts":[[1.10,-.51,.02],[1.12,-.59,.01],[1.13,-.67,0],[1.14,-.73,-.01]],
     "r0":.030, "r1":.022, "parent":7},
    {"id":29, "name":"RLL B10 (Posterior Basal)",   "latin":"", "gen":3, "children":[],
     "pts":[[1.10,-.51,.02],[1.14,-.55,.08],[1.17,-.59,.14],[1.18,-.62,.18]],
     "r0":.028, "r1":.020, "parent":7},
]


# ── 3. Tube mesh builder ───────────────────────────────────────────
def build_tube_mesh(pts, r0, r1, radial_segs=20, length_segs=None):
    """
    Build a tapered tube mesh along a polyline using Frenet frames.
    Returns (vertices, faces) as numpy arrays.
    pts: list of [x,y,z] control points
    r0:  start radius, r1: end radius
    """
    pts = np.array(pts, dtype=float)
    n   = len(pts)
    if length_segs is None:
        length_segs = max(n * 10, 20)

    # Interpolate points along the path
    # Simple linear interpolation between control points
    cumlen = np.zeros(n)
    for i in range(1, n):
        cumlen[i] = cumlen[i-1] + np.linalg.norm(pts[i] - pts[i-1])
    total_len = cumlen[-1]

    # Sample evenly along the curve
    t_samples = np.linspace(0, total_len, length_segs + 1)
    sampled   = np.zeros((length_segs + 1, 3))
    for i, t in enumerate(t_samples):
        # Find which segment t falls in
        idx = np.searchsorted(cumlen, t, side="right") - 1
        idx = np.clip(idx, 0, n - 2)
        local_t = (t - cumlen[idx]) / max(cumlen[idx+1] - cumlen[idx], 1e-9)
        sampled[i] = pts[idx] + local_t * (pts[idx+1] - pts[idx])

    # Compute Frenet frames (tangent, normal, binormal)
    tangents = np.zeros_like(sampled)
    for i in range(len(sampled)):
        if i == 0:
            tangents[i] = sampled[1] - sampled[0]
        elif i == len(sampled) - 1:
            tangents[i] = sampled[-1] - sampled[-2]
        else:
            tangents[i] = sampled[i+1] - sampled[i-1]
        norm = np.linalg.norm(tangents[i])
        tangents[i] = tangents[i] / norm if norm > 1e-9 else tangents[i]

    # Build normals using rotation minimising frames
    up = np.array([0, 1, 0])
    normals = np.zeros_like(sampled)
    normals[0] = np.cross(tangents[0], up)
    if np.linalg.norm(normals[0]) < 1e-9:
        up = np.array([1, 0, 0])
        normals[0] = np.cross(tangents[0], up)
    normals[0] /= np.linalg.norm(normals[0])

    for i in range(1, len(sampled)):
        normals[i] = normals[i-1] - np.dot(normals[i-1], tangents[i]) * tangents[i]
        norm = np.linalg.norm(normals[i])
        normals[i] = normals[i] / norm if norm > 1e-9 else normals[i-1]

    binormals = np.cross(tangents, normals)

    # Generate tube vertices
    rs  = radial_segs
    ls  = length_segs
    verts = []
    for i, (pos, N, B) in enumerate(zip(sampled, normals, binormals)):
        t_frac = i / ls
        r      = r0 + (r1 - r0) * t_frac
        for j in range(rs):
            angle = 2 * math.pi * j / rs
            v = pos + r * (math.cos(angle) * N + math.sin(angle) * B)
            verts.append(v)
    verts = np.array(verts)

    # Generate faces (quads split into triangles)
    faces = []
    for i in range(ls):
        for j in range(rs):
            a = i * rs + j
            b = i * rs + (j + 1) % rs
            c = (i + 1) * rs + (j + 1) % rs
            d = (i + 1) * rs + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces = np.array(faces)

    return verts, faces


# ── 4. Build full airway mesh ──────────────────────────────────────
def build_airway_mesh(segs):
    """
    Concatenate tube meshes for all segments into a single trimesh.
    """
    print("[1/3] Building synthetic airway mesh")
    all_verts = []
    all_faces = []
    vert_offset = 0

    for seg in segs:
        v, f = build_tube_mesh(seg["pts"], seg["r0"], seg["r1"])
        all_verts.append(v)
        all_faces.append(f + vert_offset)
        vert_offset += len(v)

    verts = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    print(f"    Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    return mesh


# ── 5. Build metadata sidecar ──────────────────────────────────────
def build_meta(segs, patient_id):
    """
    Construct the JSON sidecar in the same format pipeline.py produces.
    Includes segment graph, pathologies, and a neutral transform.
    """
    print("[2/3] Building metadata sidecar")

    # Collect pathologies from segments that have them
    pathologies = []
    for seg in segs:
        if "pathology" in seg:
            p = seg["pathology"].copy()
            p["seg_id"] = seg["id"]
            pathologies.append(p)

    meta = {
        "version": "pneumovr-1.1",
        "patient": {
            "patient_id":   patient_id,
            "patient_name": "SYNTHETIC PATIENT",
            "study_date":   "20240101",
            "modality":     "SYNTHETIC",
            "n_slices":     0,
            "note":         "Generated by synthetic.py — not real patient data",
        },
        "transform": {
            # No transform needed — already in Three.js world space
            "centre_mm":   [0, 0, 0],
            "scale":       1.0,
            "target_span": 2.4,
            "is_synthetic": True,
        },
        "segments": [
            {
                "id":       s["id"],
                "name":     s["name"],
                "latin":    s.get("latin", ""),
                "gen":      s["gen"],
                "pts":      s["pts"],          # already in Three.js coords
                "r0":       s["r0"],
                "r1":       s["r1"],
                "children": s["children"],
                "parent":   s["parent"],
            }
            for s in segs
        ],
        "pathologies": pathologies,
        "mesh_file":   f"{patient_id}_airway.glb",
    }

    print(f"    {len(meta['segments'])} segments, {len(pathologies)} pathologies")
    return meta


# ── 6. Export ──────────────────────────────────────────────────────
def export(mesh, meta, patient_id, out_dir):
    """Write GLTF + JSON to output directory."""
    print("[3/3] Exporting files")
    os.makedirs(out_dir, exist_ok=True)

    gltf_path = os.path.join(out_dir, f"{patient_id}_airway.glb")
    json_path = os.path.join(out_dir, f"{patient_id}_meta.json")

    # Ensure smooth vertex normals are computed and included in export
    # (trimesh computes them but doesn't write to GLB by default)
    import trimesh.exchange.gltf as gltf_ex
    _ = mesh.vertex_normals  # force computation
    mesh.export(gltf_path, file_type="glb", include_normals=True)
    print(f"    GLTF → {gltf_path}")

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    JSON → {json_path}")

    return gltf_path, json_path


# ── 7. Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PNEUMO·VR Synthetic Patient Generator")
    parser.add_argument("--out",        default="./output",        help="Output directory")
    parser.add_argument("--patient-id", default="SYNTHETIC_001",   help="Patient ID prefix")
    args = parser.parse_args()

    t0 = time.time()
    print("\n══ PNEUMO·VR Synthetic Generator ══════════════════════")
    print(f"   Output:     {args.out}")
    print(f"   Patient ID: {args.patient_id}\n")

    mesh            = build_airway_mesh(SEGS)
    meta            = build_meta(SEGS, args.patient_id)
    gltf_path, json_path = export(mesh, meta, args.patient_id, args.out)

    elapsed = time.time() - t0
    print(f"\n══ Done in {elapsed:.1f}s ══════════════════════════════════")
    print(f"\n   Drop these two files into your project folder and")
    print(f"   click 'LOAD CT' in the frontend to test the loader.\n")
    print(f"   GLTF: {gltf_path}")
    print(f"   Meta: {json_path}\n")


if __name__ == "__main__":
    main()