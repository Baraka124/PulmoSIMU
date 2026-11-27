"""Unreviewed airway/vascular graph candidates; preserve cycles and fragments."""
import itertools
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

MAX_VOXELS=8_000_000
OFFSETS=tuple(d for d in itertools.product((-1,0,1),repeat=3) if d!=(0,0,0))

def decompose(points,edges,xyz,radii):
    """Partition undirected voxel edges into maximal degree-two chains.
    A cycle without a junction has one deterministic anchor and a closed chain.
    No anatomical names, pruning, tree forcing or inferred continuity.
    """
    n=len(points);adj=[set() for _ in range(n)]
    for i,j in edges:
        if not 0<=i<j<n or j in adj[i]:raise ValueError("Invalid or duplicate edge")
        adj[i].add(j);adj[j].add(i)
    visited=set();branches=[]
    def walk(start,neighbor):
        path=[start,neighbor];visited.add(tuple(sorted((start,neighbor))))
        while path[-1]!=start and len(adj[path[-1]])==2:
            k=next(iter(adj[path[-1]]-{path[-2]}))
            e=tuple(sorted((path[-1],k)))
            if e in visited:raise ValueError("Unexpected chain collision")
            visited.add(e);path.append(k)
        p=xyz[path];lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
        length=float(lengths.sum());chord=float(np.linalg.norm(p[-1]-p[0]))
        branches.append({"id":len(branches),"start":start,"end":path[-1],"voxel_node_ids":path,
           "length_mm":length,"chord_mm":chord,"tortuosity":length/chord if chord>0 else None,
           "closed":start==path[-1],"mean_EDT_radius_mm":float(np.mean(radii[path])),
           "minimum_EDT_radius_mm":float(np.min(radii[path]))})
    for i in range(n):
        if len(adj[i])!=2:
            for j in sorted(adj[i]):
                if tuple(sorted((i,j))) not in visited:walk(i,j)
    for i,j in edges:
        if (i,j) not in visited:walk(i,j)
    if len(visited)!=len(edges):raise ValueError("Graph edge coverage mismatch")
    seen=set();components=[]
    for i in range(n):
        if i in seen:continue
        stack=[i];seen.add(i);members=[]
        while stack:
            p=stack.pop();members.append(p)
            for q in sorted(adj[p]):
                if q not in seen:seen.add(q);stack.append(q)
        ec=sum(len(adj[p]) for p in members)//2
        components.append({"id":len(components),"nodes":len(members),"edges":ec,
                           "cycle_rank":ec-len(members)+1,"anchor":min(members)})
    return branches,components,[len(a) for a in adj]

def build(mask,affine):
    m=np.asarray(mask);a=np.asarray(affine,dtype=float)
    if m.ndim!=3 or min(m.shape)<3 or m.size>MAX_VOXELS or not np.all((m==0)|(m==1)) or not m.any():
        raise ValueError("Nonempty bounded binary mask required")
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1]):raise ValueError("Invalid affine")
    s=np.linalg.norm(a[:3,:3],axis=0)
    if np.any(s<.1) or np.any(s>10) or not np.allclose((a[:3,:3]/s).T@(a[:3,:3]/s),np.eye(3),rtol=0,atol=1e-5):
        raise ValueError("Orthogonal physical grid required")
    labels,n=ndi.label(m,structure=ndi.generate_binary_structure(3,1))
    if n>256:raise ValueError("Too many mask components; inspect fragmentation")
    sk=np.zeros(m.shape,dtype=bool);missing=[]
    for k,region in enumerate(ndi.find_objects(labels),1):
        local=skeletonize(labels[region]==k,method="lee").astype(bool)
        if not local.any():missing.append(k)
        sk[region]|=local
    points=np.argwhere(sk)
    if not 0<len(points)<=100000:raise ValueError("Empty or oversized skeleton")
    xyz=points@a[:3,:3].T+a[:3,3]
    radii=ndi.distance_transform_edt(np.pad(m.astype(bool),1),sampling=s)[tuple((points+1).T)]
    lookup={tuple(p):i for i,p in enumerate(points)}
    source_labels=labels[tuple(points.T)]
    edges=[]
    for p,i in lookup.items():
        for d in OFFSETS:
            j=lookup.get(tuple(x+y for x,y in zip(p,d)))
            if j is not None and j>i and source_labels[i]==source_labels[j]:
                edges.append((i,j))
    edges.sort()
    branches,components,degrees=decompose(points,edges,xyz,radii)
    return {"schema":"pulmo.anatomical-branch-graph/1","nodes_voxel_ijk":points.tolist(),
       "nodes_ras_mm":xyz.tolist(),"node_EDT_radius_mm":radii.tolist(),"node_degree":degrees,
       "node_mask_component":source_labels.tolist(),"edges":edges,"branches":branches,"graph_components":components,
       "mask_components_6":int(n),"mask_components_without_skeleton":missing,
       "total_voxel_edge_length_mm":float(sum(b["length_mm"] for b in branches)),
       "cycle_rank":sum(c["cycle_rank"] for c in components),
       "isolated_nodes":int(sum(d==0 for d in degrees)),
       "convention":"Lee thinning per face-connected mask component; 26-neighbor skeleton edges; degree-two chain decomposition",
       "limitations":["Voxel skeleton can contain diagonal shortcut cycles and junction clusters","EDT radii are voxel-center proxies","No anatomical labels or rooted generations inferred"],
       "review":"PENDING","clinical_navigation_allowed":False}
