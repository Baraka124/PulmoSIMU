/* PulmoSIMU XR16-F — external clinical review packet gate.
 * No toggle, signature, or review form can turn reference geometry into a
 * validated clinical model. Designed to expose missing evidence honestly.
 */
(function(root){
'use strict';
const checks=Object.freeze([
 Object.freeze({id:'airway-map',label:'Reference airway graph and source branch labels reviewed'}),
 Object.freeze({id:'landmarks',label:'Trachea, carina, main bronchi and accessible junctions reviewed'}),
 Object.freeze({id:'visual-fidelity',label:'Endoscopic appearance compared with appropriate clinical reference'}),
 Object.freeze({id:'instrument-behavior',label:'Reference steering, insertion and branch transitions inspected'}),
 Object.freeze({id:'limitations',label:'Source, measurement and patient-specific limitations accepted'}),
 Object.freeze({id:'physical-bench',label:'Independent physical instrument bench report recorded'})
]);
const iso=x=>typeof x==='string'&&Number.isFinite(Date.parse(x));
const text=x=>typeof x==='string'&&x.trim().length>=3&&x.trim().length<=300;
function assess(packet){
 const missing=[];
 if(!packet||packet.version!=='pulmosimu-clinical-review/1')
   return Object.freeze({status:'REJECTED',missing:['REVIEW_FORMAT'],readyForIndependentReview:false,clinicalValidated:false});
 const c=packet.reviewer||{};
 if(!text(c.name))missing.push('REVIEWER_NAME');
 if(!text(c.role))missing.push('REVIEWER_ROLE');
 if(!iso(c.reviewedAt))missing.push('REVIEW_DATE');
 if(packet.model!=='original-reference')
   missing.push('REFERENCE_MODEL_ONLY_NO_PATIENT_SOURCE');
 const observations=packet.observations||{};
 for(const check of checks){
   const r=observations[check.id];
   if(!r||r.completed!==true||!text(r.evidence))
     missing.push('EVIDENCE_'+check.id);
 }
 if(!text(packet.limitationsAcknowledged))
   missing.push('LIMITATIONS_TEXT');
 // A bench pass is evidence of engineering trace quality only; recording
 // it does not establish a medically approved physical instrument.
 const bench=packet.bench||{};
 if(bench.status!=='BENCH_ENGINEERING_TARGET_MET'||bench.clinicalValidated!==false)
   missing.push('BENCH_ENGINEERING_EVIDENCE');
 return Object.freeze({
  status:missing.length?'INCOMPLETE_REVIEW_PACKET':'READY_FOR_INDEPENDENT_CLINICAL_REVIEW',
  missing:Object.freeze(missing),
  readyForIndependentReview:missing.length===0,
  clinicalValidated:false,patientSpecificValidated:false,
  regulatoryApproval:false,
  model:'original-reference',
  disclaimer:'Evidence checklist is administrative. Final clinical validity requires independent expert assessment against measured ground truth.'
 });
}
root.PulmoXRClinicalReview=Object.freeze({assess,checks});
})(typeof window!=='undefined'?window:globalThis);
