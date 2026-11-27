"""Physical cross-sectional morphometry for candidate tubular branches."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import find_contours

def section(mask,affine,center,normal,*,radius_mm=8.,step_mm=.25):
    m=np.asarray(mask);a=np.asarray(affine,dtype=float)
    p=np.asarray(center,dtype=float);n=np.asarray(normal,dtype=float)
    if m.ndim!=3 or a.shape!=(4,4) or p.shape!=(3,) or n.shape!=(3,) or not all(np.isfinite(v).all() for v in (a,p,n)) or np.linalg.norm(n)<1e-8:
        raise ValueError("Valid physical plane required")
    if not np.isfinite([radius_mm,step_mm]).all() or not .5<=radius_mm<=30 or not .1<=step_mm<=2 or radius_mm/step_mm>150:
        raise ValueError("Section raster exceeds budget")
    n/=np.linalg.norm(n);seed=np.eye(3)[np.argmin(np.abs(n))]
    u=np.cross(n,seed);u/=np.linalg.norm(u);v=np.cross(n,u)
    k=int(np.ceil(radius_mm/step_mm));q=np.arange(-k,k+1)*step_mm
    xx,yy=np.meshgrid(q,q,indexing="ij");disk=xx**2+yy**2<=radius_mm**2
    ras=p+xx[...,None]*u+yy[...,None]*v
    inv=np.linalg.inv(a);vox=ras@inv[:3,:3].T+inv[:3,3]
    ijk=np.floor(vox+.5).astype(int)
    support=np.all((ijk>=0)&(ijk<np.asarray(m.shape)),axis=-1)
    hit=np.zeros(disk.shape,dtype=bool);hit[support]=m[tuple(ijk[support].T)]==1;hit&=disk
    labels,_=ndi.label(hit,structure=ndi.generate_binary_structure(2,1))
    label=int(labels[k,k])
    result={"center_RAS_mm":p.tolist(),"normal_RAS":n.tolist(),"radius_mm":radius_mm,"step_mm":step_mm,
      "area_mm2":None,"perimeter_mm":None,"equivalent_diameter_mm":None,"hydraulic_diameter_mm":None,
      "eccentricity":None,"centroid_offset_mm":None,"status":"CENTER_NOT_FOREGROUND"}
    if not label:return result
    region=labels==label
    rim=disk&~ndi.binary_erosion(disk,structure=ndi.generate_binary_structure(2,1),border_value=0)
    if np.any(region&rim) or np.any(disk&~support):
        result["status"]="TRUNCATED_OR_OUTSIDE_FOV";return result
    if region.sum()<4:
        result["status"]="UNDERRESOLVED";return result
    contours=find_contours(np.pad(region.astype(float),1),.5,fully_connected="low")
    perimeter=float(sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum()*step_mm for c in contours))
    if perimeter<=0 or any(not np.allclose(c[0],c[-1]) for c in contours):
        result["status"]="OPEN_CONTOUR";return result
    area=float(region.sum()*step_mm**2)
    samples=np.stack([xx[region],yy[region]],axis=-1)
    cov=np.cov(samples,rowvar=False,bias=True)+np.eye(2)*step_mm**2/12
    eig=np.linalg.eigvalsh(cov)
    result.update(status="ESTIMATED",area_mm2=area,perimeter_mm=perimeter,
       equivalent_diameter_mm=float(2*np.sqrt(area/np.pi)),hydraulic_diameter_mm=4*area/perimeter,
       eccentricity=float(np.sqrt(max(0,1-eig[0]/eig[1]))),centroid_offset_mm=float(np.linalg.norm(samples.mean(axis=0))))
    return result

def measure(mask,affine,graph,*,interval_mm=5.,tangent_window_mm=2.,radius_mm=8.,step_mm=.25,max_sections=256):
    if not np.isfinite([interval_mm,tangent_window_mm]).all() or not .5<=interval_mm<=50 or not .5<=tangent_window_mm<=20 or type(max_sections) is not int or not 1<=max_sections<=1024:
        raise ValueError("Bounded section settings required")
    xyz=np.asarray(graph["nodes_ras_mm"],dtype=float);results=[];omitted=[]
    for b in graph["branches"]:
        p=xyz[b["voxel_node_ids"]]
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        if b["closed"] or arc[-1]<2*tangent_window_mm:
            omitted.append({"branch_id":b["id"],"reason":"closed_or_short_branch"});continue
        targets=np.arange(tangent_window_mm,arc[-1]-tangent_window_mm+1e-9,interval_mm)
        for t in targets:
            if len(results)>=max_sections:raise ValueError("Section budget exceeded; use coarser interval or smaller ROI")
            def point(s):return np.array([np.interp(s,arc,p[:,i]) for i in range(3)])
            center=point(t);normal=point(t+tangent_window_mm)-point(t-tangent_window_mm)
            if np.linalg.norm(normal)<1e-8:
                omitted.append({"branch_id":b["id"],"reason":"undefined_tangent","arc_mm":float(t)});continue
            r=section(mask,affine,center,normal,radius_mm=radius_mm,step_mm=step_mm)
            results.append({"branch_id":b["id"],"arc_mm":float(t),**r})
    return {"schema":"pulmo.branch-morphometry/1","sections":results,"omitted":omitted,
      "estimated_sections":sum(r["status"]=="ESTIMATED" for r in results),
      "convention":"Nearest-voxel plane raster, center-connected section, marching-squares perimeter; physical mm",
      "limitations":["Voxel resolution and plane orientation limit accuracy","Hydraulic diameter is geometric, not flow resistance","Sections near branch junctions may include adjacent anatomy"],
      "review":"PENDING","clinical_navigation_allowed":False}
