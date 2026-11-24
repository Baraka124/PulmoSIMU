/* PulmoSIMU E3 · Scientific linkage and provenance gate.
 * Explicitly separates reference physics, verified CT voxel identity,
 * independent geometry validation, and clinical-device evidence.
 */
(function(root){
'use strict';
const hash=x=>typeof x==='string'&&/^[0-9a-f]{64}$/i.test(x);
function classify({modelRole='reference',ctStudy=null,referenceSolverReady=false,
       geometryValidated=false,airwayReviewed=false,motionStudy=null}={}){
 const reference=modelRole==='reference';
 const imaging=Boolean(!reference&&ctStudy&&root.PulmoXRCTSpatial?.gate(ctStudy).ok);
 const motion=motionAcceptance(motionStudy).ok;
 return Object.freeze({
   source:reference?'AUTHORED_PROCEDURAL_REFERENCE':imaging?'HASH_VERIFIED_NIFTI_RESEARCH':'UNVERIFIED_IMPORT',
   imaging:imaging?'SOURCE_MATCHED_RESEARCH':'NOT_SOURCE_MATCHED',
   anatomy:geometryValidated?'EXTERNAL_REVIEW_ASSERTED':'NOT_CLINICALLY_VALIDATED',
   respiratoryPhysics:reference&&referenceSolverReady?'R3_ASSUMED_REFERENCE_ONLY':'NOT_ATTACHED',
   bronchoscopy:reference?'ORIGINAL_REFERENCE_SIMULATOR_ONLY':'CT_SPECIFIC_PROCEDURE_BLOCKED',
   patientSpecificPressureFlow:false,
   patientSpecificInstrumentSafety:false,
   fourDimensionalMotion:motion?'SOURCE_REVIEW_PRESENT_SEPARATE_VALIDATION':'NOT_AVAILABLE',
   sourceGeometryReviewed:Boolean(geometryValidated),
   airwayReviewed:Boolean(airwayReviewed),
   patientSpecificClinicalUse:false
 });
}
function motionAcceptance(study){
 if(!study)return Object.freeze({ok:false,reason:'No registered 4D motion study'});
 const {frames,sourceSha256,reviewed,seconds}=study;
 if(!hash(sourceSha256)||!Array.isArray(frames)||frames.length<2||frames.length>100||
    !reviewed||!Number.isFinite(seconds)||seconds<=0)
   return Object.freeze({ok:false,reason:'Measured source frames, duration and independent review required'});
 if(frames.some(f=>!hash(f?.sha256)||!Array.isArray(f?.voxelToRASmm)||f.voxelToRASmm.length!==4||
     f.voxelToRASmm.some(row=>!Array.isArray(row)||row.length!==4||
       row.some(x=>!Number.isFinite(x)))))
   return Object.freeze({ok:false,reason:'Phase frame identity or spatial affine invalid'});
 return Object.freeze({ok:true,reason:'4D source metadata reviewed; biomechanics not automatically validated'});
}
function canCoupleToCT({ctSha256,geometrySha256,solverSourceSha256,measuredBoundaryConditions,
                       independentValidation}={}){
 return Boolean(hash(ctSha256)&&hash(geometrySha256)&&hash(solverSourceSha256)&&
   geometrySha256.toLowerCase()===solverSourceSha256.toLowerCase()&&
   measuredBoundaryConditions===true&&independentValidation===true);
}
function measurementMM(a,b){
 if(!Array.isArray(a)||!Array.isArray(b)||a.length!==3||b.length!==3||
    ![...a,...b].every(x=>typeof x==='number'&&Number.isFinite(x)))
   throw Error('Both RAS landmarks must have finite millimetre coordinates');
 return Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
}
root.PulmoXRScience=Object.freeze({classify,motionAcceptance,canCoupleToCT,measurementMM});
})(typeof window!=='undefined'?window:globalThis);
