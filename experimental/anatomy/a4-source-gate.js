/* PulmoSIMU A4 · Source-specific anatomical gating.
 * CT file hashes & affine alignment do not prove segmentation accuracy.
 * A patient-specific bronchoscopy procedure is NOT available until signed
 * independent source review, lumen centreline and device geometry exist.
 * This module never upgrades procedural reference shapes into patient data.
 */
(function(global){
'use strict';
const required=['RUL','RML','RLL','LUL','LLL','airway_lumen',
 'pulmonary_artery','pulmonary_vein'];
const stages=Object.freeze([
 {id:'A3.1',name:'Bilateral airspace'},
 {id:'A3.2',name:'Five-lobe anatomy'},
 {id:'A3.3',name:'Bronchial lumen'},
 {id:'A3.4',name:'Pulmonary vessels'},
 {id:'A4',name:'CT-specific procedure'}
]);
function assess({manifest=null,ctSha=null,meshSha=null,structureQA=null,
  clinicianReview=null,sourceAirway=null,instrument=null}={}){
 const sourceIdentity=!!manifest?.source_sha256&&
   manifest.source_sha256===ctSha&&manifest.mesh_sha256===meshSha;
 const hasAffine=Array.isArray(manifest?.model_from_ras_mm)&&
   manifest.model_from_ras_mm.length===4&&
   Array.isArray(manifest?.sform_RAS_mm)&&manifest.sform_RAS_mm.length===4;
 const allSource=sourceIdentity&&hasAffine;
 const groups=structureQA?.groups||{};
 const integrity=allSource&&structureQA?.CT_SHA256===ctSha&&
   structureQA?.A31_labelmap_SHA256===manifest?.source_labelmap_sha256;
 const lobeMaskReady=!!(integrity&&groups['A3.2 - lobes']?.complete&&
   structureQA?.lobar_topology_gate==='NONOVERLAPPING_UNREVIEWED');
 const airwayMaskReady=!!(integrity&&groups['A3.3 - bronchial lumen']?.complete);
 const vesselMaskReady=!!(integrity&&groups['A3.4 - vessels']?.complete);
 const independentApproval=clinicianReview?.status==='APPROVED_FOR_RESEARCH_PROCEDURE'&&
   typeof clinicianReview?.reviewer==='string'&&clinicianReview.reviewer.length>2&&
   clinicianReview?.source_sha256===ctSha&&
   clinicianReview?.mesh_sha256===meshSha&&
   clinicianReview?.signed_report_sha256?.length===64;
 const lumenLinked=!!(airwayMaskReady&&sourceAirway?.centrelineCoframed===true&&
   sourceAirway?.topologyReviewed===true&&sourceAirway?.lumenWallValidated===true&&
   sourceAirway?.ct_source_sha256===ctSha);
 const instrumentLinked=!!(instrument?.sourceDimensionReviewed===true&&
   instrument?.calibratedDiameterMm>0&&
   instrument?.collisionModelReviewed===true);
 const a4Candidate=!!(allSource&&lobeMaskReady&&lumenLinked&&
   independentApproval&&instrumentLinked);
 const status=[
   {id:'A3.1',name:stages[0].name,status:allSource?
     'SOURCE_REGISTERED_RESEARCH_ONLY':'SOURCE_FILES_NOT_MATCHED'},
   {id:'A3.2',name:stages[1].name,status:lobeMaskReady?
     'SOURCE_MASKS_RECEIVED_UNREVIEWED':'FIVE_LOBE_MASKS_MISSING_OR_INVALID'},
   {id:'A3.3',name:stages[2].name,status:lumenLinked?
     'REVIEWED_SOURCE_CENTRELINE_AVAILABLE':
     airwayMaskReady?'SOURCE_AIRWAY_MASK_UNREVIEWED':'AIRWAY_LUMEN_MISSING'},
   {id:'A3.4',name:stages[3].name,status:vesselMaskReady?
     'SOURCE_VESSEL_MASKS_UNREVIEWED':'SOURCE_VESSEL_MASKS_MISSING'},
   {id:'A4',name:stages[4].name,status:a4Candidate?
     'RESEARCH_PROCEDURE_INTEGRATION_CANDIDATE':'CT_PROCEDURE_BLOCKED'}
 ];
 return Object.freeze({sourceIntegrityMatched:allSource,
    researchAnatomyInspectionAllowed:allSource,
    researchStageStatus:Object.freeze(status.map(x=>Object.freeze(x))),
    masksPresent:required.filter(x=>structureQA?.structure_findings?.[x]?.status===
      'SOURCE_COFAMED_RESEARCH_MASK_UNREVIEWED'),
    clinicalAnatomyValidated:false,
    CTLinkedBronchoscopyAllowed:a4Candidate,
    CTLinkedBronchoscopyStatus:a4Candidate?
      'RESEARCH_CANDIDATE_REQUIRES_QUEST_AND_CLINICAL_REVIEW':'BLOCKED',
    proceduralReferenceAvailableSeparately:true,
    noInferredFissuresVesselsOrAirways:true});
}
global.PulmoSourceAnatomyGate=Object.freeze({assess,stages,required});
})(typeof window!=='undefined'?window:globalThis);
