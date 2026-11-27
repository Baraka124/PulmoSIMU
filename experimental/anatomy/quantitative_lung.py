"""Quantitative CT research features. No diagnosis or direct ventilation measurement."""
import numpy as np

def _mask(v,shape):
    v=np.asarray(v)
    if v.shape!=shape or not np.all((v==0)|(v==1)):raise ValueError("Aligned binary mask required")
    return v.astype(bool)

def _summary(v):
    if not v.size:return None
    return {"mean":float(v.mean(dtype=np.float64)),"median":float(np.median(v)),
            "p05":float(np.percentile(v,5)),"p95":float(np.percentile(v,95))}

def density(hu,mask,voxel_ml,*,phase):
    x=np.asarray(hu,dtype=np.float32)
    if x.ndim!=3 or not np.isfinite(x).all() or not np.isfinite(voxel_ml) or voxel_ml<=0:
        raise ValueError("Finite 3D calibrated HU and positive voxel volume required")
    if phase not in ("inspiration","expiration"):raise ValueError("Explicit phase required")
    m=_mask(mask,x.shape)
    if not m.any():raise ValueError("Empty analysis mask")
    v=x[m];threshold=-950 if phase=="inspiration" else -856
    # Two-compartment water-equivalent approximation; do not silently clip.
    valid=(v>=-1000)&(v<=0)
    fraction=(v[valid]+1000)/1000
    return {"phase":phase,"mask_voxels":int(v.size),"mask_volume_ml":float(v.size*voxel_ml),
            "mean_HU":float(v.mean(dtype=np.float64)),"std_HU":float(v.std(dtype=np.float64)),
            "percentile_15_HU":float(np.percentile(v,15)),
            "low_attenuation_threshold_HU":threshold,"low_attenuation_fraction":float(np.mean(v<threshold)),
            "two_compartment_valid_fraction":float(valid.mean()),
            "model_valid_volume_ml":float(valid.sum()*voxel_ml),
            "model_tissue_equivalent_volume_ml":float(fraction.sum(dtype=np.float64)*voxel_ml),
            "model_air_equivalent_volume_ml":float((1-fraction).sum(dtype=np.float64)*voxel_ml),
            "two_compartment_assumptions":"air -1000 HU, water-equivalent tissue 0 HU; no extrapolation beyond range"}

def paired(insp,exp_warped,jac,insp_mask,exp_mask_warped,support,voxel_ml):
    x=np.asarray(insp,dtype=np.float32);y=np.asarray(exp_warped,dtype=np.float32);j=np.asarray(jac,dtype=np.float32)
    if x.ndim!=3 or x.shape!=y.shape or x.shape!=j.shape or not all(np.isfinite(v).all() for v in (x,y,j)) or not np.isfinite(voxel_ml) or voxel_ml<=0:
        raise ValueError("Aligned finite 3D fields required")
    mi,me,sp=[_mask(v,x.shape) for v in (insp_mask,exp_mask_warped,support)]
    if not mi.any():raise ValueError("Empty inspiration mask")
    common=mi&me&sp
    valid=common&(j>0)
    # Four explicit categories plus indeterminate. Fixed inspiration reference.
    # Do not force inconsistent low-inspiration/normal-expiration pairs into normal.
    prm=np.zeros(x.shape,dtype=np.uint8)
    prm[valid&(x>=-950)&(x<-810)&(y>=-856)]=1
    prm[valid&(x>=-950)&(x<-810)&(y<-856)]=2
    prm[valid&(x<-950)&(y<-856)]=3
    prm[valid&(x>=-810)]=4
    prm[valid&(prm==0)]=5
    # Below ideal air or unusually dense expiratory voxels are kept in the
    # indeterminate class, not presented as validated disease classifications.
    prm[valid&((x<-1000)|(y<-1000)|(x>0)|(y>0))]=5
    model=valid&(x>=-1000)&(x<=0)&(y>=-1000)&(y<=0)
    ti=(x[model]+1000)/1000;te=(y[model]+1000)/1000
    jj=j[model]
    residual=jj*te-ti
    airchange=(1-ti)-jj*(1-te)
    def field(v):
        a=np.zeros(x.shape,dtype=np.float32);a[model]=v
        return a
    fields={"prm-labels":prm,"paired-valid-mask":valid.astype(np.uint8),"mass-model-valid-mask":model.astype(np.uint8),
            "tissue-fraction-inspiration":field(ti),"tissue-fraction-expiration":field(te),
            "tissue-volume-residual":field(residual),"air-volume-change-per-inspiration-volume":field(airchange)}
    shrink=np.zeros(x.shape,dtype=np.float32);shrink[valid]=1-j[valid]
    fields["volume-contraction-surrogate"]=shrink
    if not all(np.isfinite(v).all() for v in fields.values()):raise ValueError("Nonfinite quantitative field")
    n=int(valid.sum())
    names={1:"normal_pattern",2:"nonemphysematous_air_trapping_pattern",3:"emphysema_pattern",4:"higher_inspiratory_attenuation_pattern",5:"indeterminate"}
    counts={name:{"voxels":int(np.count_nonzero(prm==k)),
                  "fraction_of_valid_reference":float(np.count_nonzero(prm==k)/n) if n else None} for k,name in names.items()}
    tissue_fixed=float(ti.sum(dtype=np.float64)*voxel_ml)
    tissue_mapped=float((jj*te).sum(dtype=np.float64)*voxel_ml)
    report={"reference":"inspiration","direction":"inspiration_to_expiration","reference_mask_voxels":int(mi.sum()),
            "common_lung_supported_voxels":int(common.sum()),"paired_valid_voxels":n,"paired_valid_fraction":float(n/mi.sum()),
            "excluded_nonpositive_jacobian_voxels":int(np.count_nonzero(common&(j<=0))),
            "mass_model_valid_voxels":int(model.sum()),"mass_model_valid_fraction":float(model.sum()/mi.sum()),
            "prm_thresholds_HU":{"inspiration_emphysema":-950,"expiration_air_trapping":-856,"inspiration_high_attenuation":-810},
            "prm_classes":counts,"prm_label_dictionary":{"0":"not_evaluated",**{str(k):v for k,v in names.items()}},
            "tissue_equivalent_volume_fixed_ml":tissue_fixed,"tissue_equivalent_volume_mapped_ml":tissue_mapped,
            "relative_integrated_tissue_volume_error":(tissue_mapped-tissue_fixed)/tissue_fixed if tissue_fixed>0 else None,
            "local_tissue_volume_residual":_summary(residual),"absolute_local_tissue_volume_residual":_summary(np.abs(residual)),
            "air_volume_change_per_reference_volume":_summary(airchange),"volume_contraction_surrogate":_summary(shrink[valid]),
            "zero_fill":"consult paired-valid-mask or mass-model-valid-mask; zero outside support is not a measurement",
            "mass_conservation":"evaluation of J*tissue_fraction_exp - tissue_fraction_insp; no mass-preserving optimization",
            "review":"PENDING","clinical_navigation_allowed":False}
    return fields,report
