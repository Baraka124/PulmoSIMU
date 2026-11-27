"""Deterministic, expert-seeded ROI segmentation; no automatic anatomical labels."""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from geodesic_refinement import refine
from levelset_refinement import evolve

MAX_VOXELS=8_000_000
CONNECTIVITY=ndi.generate_binary_structure(3,1)

def seed_mask(shape,seeds):
    if not isinstance(seeds,(list,tuple)) or not 1<=len(seeds)<=4096:
        raise ValueError("Explicit nonempty seed list required")
    out=np.zeros(shape,dtype=bool)
    for p in seeds:
        if not isinstance(p,(list,tuple)) or len(p)!=3 or any(type(v) is not int or not 0<=v<shape[i] for i,v in enumerate(p)):
            raise ValueError("Seed outside ROI")
        if out[tuple(p)]: raise ValueError("Duplicate seed")
        out[tuple(p)]=True
    return out

def segment(data,spacing,*,foreground,lower_hu,upper_hu,method="region_growing",background=None,feature=None,min_feature=None,refinement=None,intermediates=None):
    x=np.asarray(data,dtype=np.float32);s=np.asarray(spacing,dtype=float)
    if x.ndim!=3 or min(x.shape)<3 or x.size>MAX_VOXELS or not np.isfinite(x).all():
        raise ValueError("Finite bounded 3D ROI required")
    if s.shape!=(3,) or not np.isfinite(s).all() or np.any(s<.1) or np.any(s>10):
        raise ValueError("Physical grid outside supported spacing")
    if not np.isfinite([lower_hu,upper_hu]).all() or lower_hu>upper_hu:
        raise ValueError("Explicit ordered HU bounds required")
    allowed=(x>=lower_hu)&(x<=upper_hu)
    if feature is not None:
        f=np.asarray(feature,dtype=np.float32)
        if f.shape!=x.shape or not np.isfinite(f).all() or np.any((f<0)|(f>1)) or min_feature is None or not np.isfinite(min_feature) or not 0<=min_feature<=1:
            raise ValueError("Aligned unitless feature and threshold required")
        allowed&=f>=min_feature
    elif min_feature is not None:
        raise ValueError("Feature threshold without feature")
    fg=seed_mask(x.shape,foreground)
    if np.any(fg&~allowed): raise ValueError("Foreground seed violates admissible region")
    details=None
    if method not in ("geodesic_discrete_contour","region_levelset") and refinement is not None:
        raise ValueError("Refinement parameters require a contour method")
    if method=="region_levelset":
        if not isinstance(refinement,dict): raise ValueError("Explicit level-set parameters required")
        bg=seed_mask(x.shape,background) if background is not None else np.zeros_like(fg)
        mask,fields,details=evolve(x,s,allowed,fg,bg,**refinement)
        if intermediates is not None: intermediates.update(fields)
    elif method=="geodesic_discrete_contour":
        if not isinstance(refinement,dict): raise ValueError("Explicit refinement parameters required")
        bg=seed_mask(x.shape,background) if background is not None else np.zeros_like(fg)
        mask,fields,details=refine(x,s,allowed,fg,bg,**refinement)
        if intermediates is not None: intermediates.update(fields)
    elif method=="region_growing":
        if background is not None: raise ValueError("Background markers require watershed")
        mask=ndi.binary_propagation(fg,structure=CONNECTIVITY,mask=allowed)
    elif method=="watershed":
        bg=seed_mask(x.shape,background)
        if np.any(fg&bg) or np.any(bg&~allowed): raise ValueError("Background seeds must be distinct and admissible")
        # Limit watershed to components reached by at least one marker.
        reached=ndi.binary_propagation(fg|bg,structure=CONNECTIVITY,mask=allowed)
        gradient=np.zeros_like(x)
        for axis,h in enumerate(s):
            g=np.gradient(x,h,axis=axis)
            gradient+=g*g
        np.sqrt(gradient,out=gradient)
        markers=fg.astype(np.int32)+2*bg.astype(np.int32)
        mask=watershed(gradient,markers,mask=reached,connectivity=CONNECTIVITY)==1
    else: raise ValueError("Unsupported segmentation method")
    # Signed distance between voxel centers; not exact interface distance.
    outside=ndi.distance_transform_edt(~mask,sampling=s)
    inside=ndi.distance_transform_edt(np.pad(mask,1),sampling=s)[1:-1,1:-1,1:-1]
    signed=(outside-inside).astype(np.float32)
    # Distances outside an entirely full mask are not defined within the ROI;
    # padded background establishes an explicit artificial boundary.
    labels,n=ndi.label(mask,structure=CONNECTIVITY)
    counts=np.bincount(labels.ravel())[1:]
    faces=[int(np.count_nonzero(np.take(mask,i,axis=a))) for a in range(3) for i in (0,-1)]
    report={"method":method,"foreground_seeds":len(foreground),"background_seeds":len(background or []),
      "lower_hu":float(lower_hu),"upper_hu":float(upper_hu),"min_feature":min_feature,
      "voxels":int(mask.sum()),"volume_ml":float(mask.sum()*np.prod(s)/1000),
      "components":int(n),"largest_component_fraction":float(counts.max()/counts.sum()),
      "roi_face_foreground_counts":faces,"touches_roi_boundary":any(faces),
      "distance_convention":"negative inside, positive outside; physical voxel-center EDT; padded ROI exterior",
      "review":"PENDING","anatomical_identity_verified":False,"clinical_navigation_allowed":False}
    if details is not None: report["refinement"]=details
    return mask.astype(np.uint8),signed,report
