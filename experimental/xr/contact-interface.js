/* HIL05 numerical interface only: dimensionless geometric contact risk.
 * No conversion to physical force or motor command. */
(function(root){
'use strict';
function evaluate({shaft,reference,tracking,procedure}={}){
 if(!reference||!tracking||!procedure||!Number.isFinite(shaft?.minMarginMm))return {state:'UNAVAILABLE',riskIndex:null,hardwareCommand:false,physicalForceN:null};
 const interference=Math.max(0,-shaft.minMarginMm),risk=Math.min(1,interference/2);
 return {state:interference>0?'GEOMETRIC_INTERFERENCE':'NO_SAMPLED_INTERFERENCE',riskIndex:risk,interferenceMm:interference,physicalForceN:null,physicalTorqueNm:null,hardwareCommand:false,calibrated:false};
}
root.PulmoXRContactInterface=Object.freeze({evaluate});
})(typeof window!=='undefined'?window:globalThis);
