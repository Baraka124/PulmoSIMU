/* HIL03: independent reference sweep evaluation. Raw ADC is never
 * presented as BENCH/1 or used to drive either physical or virtual devices. */
(function(root){
'use strict';
function fit(rows,{minSpanAdc=200,maxCheckErrorDeg=5,maxHysteresisDeg=5}={}){
 if(!Array.isArray(rows)||rows.length<9||rows.length>10000||![minSpanAdc,maxCheckErrorDeg,maxHysteresisDeg].every(v=>Number.isFinite(v)&&v>0))throw Error('INVALID_SWEEP');
 for(const p of rows)if(!p||!Number.isInteger(p.adc)||p.adc<0||p.adc>4095||!Number.isFinite(p.referenceDeg)||Math.abs(p.referenceDeg)>180||!['fit','check'].includes(p.role)||!['up','down'].includes(p.direction))throw Error('INVALID_REFERENCE_ROW');
 const train=rows.filter(p=>p.role==='fit'),check=rows.filter(p=>p.role==='check');
 if(train.length<6||check.length<3||new Set(train.map(p=>p.referenceDeg)).size<3||new Set(check.map(p=>p.referenceDeg)).size<3||!['up','down'].every(dir=>train.some(p=>p.direction===dir)))throw Error('INSUFFICIENT_INDEPENDENT_SWEEPS');
 const min=Math.min(...train.map(p=>p.adc)),max=Math.max(...train.map(p=>p.adc));if(max-min<minSpanAdc)throw Error('INSUFFICIENT_ADC_SPAN');
 if(check.some(p=>p.adc<min||p.adc>max))throw Error('CHECK_OUTSIDE_FIT_RANGE');
 const avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
 const xm=avg(train.map(p=>p.adc)),ym=avg(train.map(p=>p.referenceDeg));
 const slope=train.reduce((s,p)=>s+(p.adc-xm)*(p.referenceDeg-ym),0)/train.reduce((s,p)=>s+(p.adc-xm)**2,0);
 if(!Number.isFinite(slope)||Math.abs(slope)<1e-6)throw Error('DEGENERATE_CALIBRATION');
 const offset=ym-slope*xm,predict=adc=>slope*adc+offset;
 const errors=check.map(p=>predict(p.adc)-p.referenceDeg),rmse=Math.sqrt(avg(errors.map(e=>e*e))),maxError=Math.max(...errors.map(Math.abs));
 const paired=[];for(const ref of new Set(train.map(p=>p.referenceDeg))){
  const a=train.filter(p=>p.referenceDeg===ref&&p.direction==='up'),b=train.filter(p=>p.referenceDeg===ref&&p.direction==='down');
  if(a.length&&b.length)paired.push(Math.abs(predict(avg(a.map(p=>p.adc)))-predict(avg(b.map(p=>p.adc)))));
 }
 if(paired.length<3)throw Error('INSUFFICIENT_PAIRED_SWEEP_LEVELS');
 const hysteresis=Math.max(...paired);
 return {schema:'pulmosimu-hil03-angle-calibration/1',status:maxError<=maxCheckErrorDeg&&hysteresis<=maxHysteresisDeg?'PROVISIONAL_TARGET_MET':'TARGET_NOT_MET',slopeDegPerAdc:slope,offsetDeg:offset,adcRange:[min,max],fitCount:train.length,checkCount:check.length,checkRmseDeg:rmse,maxCheckErrorDeg:maxError,maxHysteresisDeg:hysteresis,pairedLevels:paired.length,limits:{maxCheckErrorDeg,maxHysteresisDeg},calibratedAxis:'leverAngleOnly',benchProtocolEligible:false,independentMetrologySignoff:false,hardwareCommand:false};
}
function apply(report,adc){if(report?.schema!=='pulmosimu-hil03-angle-calibration/1'||report.status!=='PROVISIONAL_TARGET_MET'||!Number.isInteger(adc)||adc<report.adcRange[0]||adc>report.adcRange[1])throw Error('UNAVAILABLE_OR_OUTSIDE_CALIBRATION');return report.slopeDegPerAdc*adc+report.offsetDeg;}
root.PulmoHILAngleCalibration=Object.freeze({fit,apply});
})(typeof window!=='undefined'?window:globalThis);
