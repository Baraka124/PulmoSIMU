#!/usr/bin/env python3
"""A3.2-A3.4 source-mask quality gates. Never fabricates lobes, bronchi or vessels.
Inputs: one separately supplied CT-coframed .npz {mask, affine_ras}
per structure, alongside A3.1 labels and verified manifest.
Requires numpy scipy. Does NOT independently certify pulmonary anatomy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

GROUPS={
  'A3.2 - lobes':('RUL','RML','RLL','LUL','LLL'),
  'A3.3 - bronchial lumen':('airway_lumen',),
  'A3.4 - vessels':('pulmonary_artery','pulmonary_vein')
}
LUNG_SIDE={'RUL':2,'RML':2,'RLL':2,'LUL':1,'LLL':1}
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def validate(mask_dir,labels_file,manifest_file,output_file):
    meta=json.loads(Path(manifest_file).read_text())
    if digest(labels_file)!=meta['source_labelmap_sha256']:
        raise ValueError('A3.1 mask hash mismatch')
    with np.load(labels_file,allow_pickle=False) as z:
        labels=np.asarray(z['labels'],dtype=np.uint8)
        affine=np.asarray(z['affine_ras'],dtype=float)
    if not np.allclose(affine,meta['sform_RAS_mm'],atol=1e-4):
        raise ValueError('A3.1 CT affine mismatch')
    masks={};findings={};groups={}
    voxel_l=abs(np.linalg.det(affine[:3,:3]))/1e6
    for group,names in GROUPS.items():
        for name in names:
            path=Path(mask_dir)/(name+'.npz')
            if not path.exists():
                findings[name]={'status':'MISSING_SOURCE_SEGMENTATION'}
                continue
            with np.load(path,allow_pickle=False) as data:
                if 'mask' not in data or 'affine_ras' not in data:
                    raise ValueError(name+' must provide mask and affine_ras arrays')
                mask=np.asarray(data['mask'],dtype=bool)
                original_affine=np.asarray(data['affine_ras'],dtype=float)
            if mask.shape!=labels.shape or not np.allclose(original_affine,affine,atol=1e-4):
                raise ValueError(name+' is not registered to original source CT voxels')
            if not mask.any():raise ValueError(name+' has an empty mask')
            _,count=ndi.label(mask,structure=ndi.generate_binary_structure(3,1))
            inside=float((mask & (labels==LUNG_SIDE[name])).sum()/mask.sum()) if name in LUNG_SIDE else None
            masks[name]=mask
            findings[name]=dict(status='SOURCE_COFAMED_RESEARCH_MASK_UNREVIEWED',
                sha256=digest(path),voxels=int(mask.sum()),
                volume_l=float(mask.sum()*voxel_l),components=int(count),
                fraction_inside_candidate_lung=inside,
                independent_anatomical_review='NOT_PROVIDED')
        groups[group]={'complete':all(x in masks for x in names),
            'status':'AVAILABLE_FOR_RESEARCH_QA_ONLY' if all(x in masks for x in names) else 'MISSING_SOURCE_MASKS',
            'required_masks':list(names)}
    lobes=GROUPS['A3.2 - lobes']
    overlaps={}
    if all(x in masks for x in lobes):
        for i,a in enumerate(lobes):
            for b in lobes[i+1:]:
                n=int((masks[a]&masks[b]).sum())
                if n:overlaps[a+'+'+b]=n
    result=dict(
        protocol='PulmoSIMU-A3.2-to-A3.4-source-mask-QA-v1',
        CT_SHA256=meta['source_sha256'],
        A31_labelmap_SHA256=meta['source_labelmap_sha256'],
        source_dimensions_kji=list(labels.shape),
        structure_findings=findings,groups=groups,
        lobar_overlap_voxels=overlaps,
        lobar_topology_gate=('MISSING_SOURCE_MASKS' if not all(x in masks for x in lobes)
            else 'FAIL_OVERLAP' if overlaps else 'NONOVERLAPPING_UNREVIEWED'),
        artery_vein_overlap_voxels=(int((masks['pulmonary_artery']&masks['pulmonary_vein']).sum())
            if all(x in masks for x in GROUPS['A3.4 - vessels']) else None),
        clinical_segmentation_approval='PENDING_INDEPENDENT_REVIEW',
        ct_registered_bronchoscopy='BLOCKED_NO_REVIEWED_AIRWAY_CENTRELINE',
        patient_specific_flow='BLOCKED_UNVALIDATED_PHYSICS',
        note='CT coframing and source file hashes are NOT proof of lobar/airway/vascular anatomical correctness.'
    )
    Path(output_file).write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('masks','labels','manifest','out'):p.add_argument('--'+a,type=Path,required=True)
    opts=p.parse_args()
    report=validate(opts.masks,opts.labels,opts.manifest,opts.out)
    print(json.dumps({'groups':report['groups'],
        'lobar_topology_gate':report['lobar_topology_gate'],
        'ct_registered_bronchoscopy':report['ct_registered_bronchoscopy']},indent=2))
