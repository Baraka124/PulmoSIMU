/* HIL04 passive bench analysis: numerical dissipation only.
 * No motor, brake command, physical force target or actual calibration. */
(function(root){
'use strict';
function sample({velocityMmS,angularRadS,axialDragN,rotationalDragNm,dtS}){
 if(![velocityMmS,angularRadS,axialDragN,rotationalDragNm,dtS].every(Number.isFinite)||axialDragN<0||rotationalDragNm<0||dtS<=0||dtS>.25)throw Error('INVALID_PASSIVE_SAMPLE');
 const forceN=velocityMmS===0?0:-Math.sign(velocityMmS)*axialDragN;
 const torqueNm=angularRadS===0?0:-Math.sign(angularRadS)*rotationalDragNm;
 const powerW=forceN*velocityMmS/1000+torqueNm*angularRadS;
 return {forceN,torqueNm,powerW,dissipatedJ:-powerW*dtS,hardwareCommand:false,measured:false};
}
function review(rows){
 if(!Array.isArray(rows)||!rows.length||rows.length>100000)throw Error('INVALID_TRACE');
 let dissipatedJ=0;for(const row of rows)dissipatedJ+=sample(row).dissipatedJ;
 return {samples:rows.length,dissipatedJ,passiveByConstruction:true,physicalValidation:false};
}
root.PulmoHILPassive=Object.freeze({sample,review});
})(typeof window!=='undefined'?window:globalThis);
