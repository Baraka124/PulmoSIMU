"""Reusable method policy; case-specific seeds, HU and ROI must be supplied.
No automatic selection of pulmonary artery versus vein or disease labels.
"""
import hashlib
import json
from pathlib import Path
from computational_assets import canonical,file_hash
from segmentation_pipeline import KINDS

POLICY={
    "airway_lumen":("region_growing","geodesic_discrete_contour","region_levelset"),
    "pulmonary_artery":("geodesic_discrete_contour","watershed","region_levelset"),
    "pulmonary_vein":("geodesic_discrete_contour","watershed","region_levelset"),
    "lung":("region_growing","watershed","region_levelset"),
    **{k:("watershed","region_levelset") for k in ("RUL","RML","RLL","LUL","LLL")}
}
def compile_protocol(intake,requests):
    root=Path(intake).resolve(strict=True)
    m=json.loads((root/"intake.json").read_text())
    if m.get("schema")!="pulmo.imaging-intake/1" or m.get("modality")!="CT" or m.get("quantity")!="HU":
        raise ValueError("Verified CT intake required")
    digest=file_hash(root/"source.nii")
    if m["normalized"]["sha256"]!=digest: raise ValueError("Source changed")
    if not isinstance(requests,list) or not 1<=len(requests)<=len(KINDS): raise ValueError("Bounded structure requests required")
    tasks=[];seen=set()
    for req in requests:
        if not isinstance(req,dict) or set(req)!={"structure","roi","enhancement","segmentation"}:
            raise ValueError("Explicit structure, ROI, enhancement and segmentation required")
        name=req["structure"];r=req["roi"];e=req["enhancement"];p=req["segmentation"]
        if name not in POLICY or name in seen: raise ValueError("Unsupported or repeated structure")
        seen.add(name)
        if not isinstance(r,list) or len(r)!=6 or any(type(x) is not int for x in r) or any(not 0<=r[i*2]<r[i*2+1]<=m["normalized"]["shape"][i] or r[i*2+1]-r[i*2]<3 for i in range(3)):
            raise ValueError("Invalid case ROI")
        if not isinstance(e,dict) or set(e)!={"iterations","kappa","scales_mm","bright"}: raise ValueError("Explicit enhancement parameters required")
        if not isinstance(p,dict) or not {"foreground","lower_hu","upper_hu","method"}<=set(p) or set(p)-{"foreground","lower_hu","upper_hu","method","background","min_feature","refinement"}:
            raise ValueError("Explicit segmentation parameters required")
        if p["method"] not in POLICY[name]: raise ValueError("Method outside structure protocol policy")
        if name in ("pulmonary_artery","pulmonary_vein") and e["bright"] is not True: raise ValueError("Vascular protocol expects bright feature response")
        if name=="airway_lumen" and p.get("min_feature") is not None and e["bright"] is not False: raise ValueError("Airway feature constraint expects dark response")
        if p["method"] in ("region_levelset","geodesic_discrete_contour") and not isinstance(p.get("refinement"),dict):
            raise ValueError("Contour parameters required")
        task={"structure":name,"roi":r,"enhancement":e,"segmentation":p,
              "anatomical_identity":"expert_claim_pending_review"}
        tasks.append(task)
    payload={"schema":"pulmo.segmentation-protocol/1","study_id":m["study_id"],"source_sha256":digest,
             "tasks":tasks,"policy":"expert_seeded_CT_ROI","clinical_navigation_allowed":False}
    return {**payload,"plan_sha256":hashlib.sha256(canonical(payload)).hexdigest()}
