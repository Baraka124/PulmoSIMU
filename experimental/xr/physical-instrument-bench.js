/* PulmoSIMU XR-16F — physical-device bench trace acceptance.
 * Evidence-only, not a direct connection to a bronchoscope or a medical-device
 * certification. Private measurements remain in the browser.
 */
(function(root){
'use strict';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const limit=Object.freeze({
 version:'pulmosimu-instrument-trace/1',
 minSamples:50,minDurationMs:5000,maxSamples:30000,
 maxGapMs:250,
 // Initial engineering QA targets, NOT clinical tolerances.
 insertionP95Mm:5,tipP95Deg:8,rollP95Deg:12
});
const fields=Object.freeze([
 ['insertion','cmdInsertionMm','measInsertionMm','mm'],
 ['tipH','cmdTipHDeg','measTipHDeg','deg'],
 ['tipV','cmdTipVDeg','measTipVDeg','deg'],
 ['roll','cmdRollDeg','measRollDeg','deg']
]);
const percent=(values,p)=>{
 const v=values.slice().sort((a,b)=>a-b);
 const index=(v.length-1)*p,lo=Math.floor(index),hi=Math.ceil(index);
 return v[lo]+(v[hi]-v[lo])*(index-lo);
};
const angleError=(a,b)=>Math.abs(((a-b+180)%360+360)%360-180);
const normalError=(a,b)=>Math.abs(a-b);
function assess(trace){
 const defects=[];
 const reject=code=>{defects.push(code);return null};
 if(!trace||trace.version!==limit.version)
   return Object.freeze({status:'REJECTED',defects:['INVALID_FORMAT_VERSION'],engineeringPassed:false,clinicalValidated:false,metrics:null});
 const p=trace.provenance||{};
 for(const k of ['deviceModel','instrumentId','referenceSensor','operator','measuredAt']){
   if(typeof p[k]!=='string'||p[k].trim().length<2||p[k].length>160)reject('PROVENANCE_'+k.toUpperCase());
 }
 if(!Number.isFinite(Date.parse(p.measuredAt)))reject('INVALID_MEASUREMENT_DATE');
 if(p.measurementSource!=='external-bench')reject('EXTERNAL_BENCH_MEASUREMENT_REQUIRED');
 if(!trace.units||trace.units.time!=='ms'||trace.units.insertion!=='mm'||
    trace.units.angles!=='deg')reject('UNSUPPORTED_UNITS');
 const samples=trace.samples;
 if(!Array.isArray(samples)||samples.length<limit.minSamples||samples.length>limit.maxSamples)
   reject('INVALID_SAMPLE_COUNT');
 if(defects.length)return Object.freeze({status:'REJECTED',defects,engineeringPassed:false,clinicalValidated:false,metrics:null});
 const errors=Object.fromEntries(fields.map(([id])=>[id,[]]));
 const timestamps=[];
 let longestGapMs=0,trackingLosses=0;
 for(let i=0;i<samples.length;i++){
   const x=samples[i];
   if(!x||!finite(x.tMs)||x.tMs<0||x.tMs>3600000||
      (i&&x.tMs<=samples[i-1].tMs))return Object.freeze({
        status:'REJECTED',defects:['INVALID_TIMESTAMP_AT_'+i],engineeringPassed:false,clinicalValidated:false,metrics:null});
   for(const [name,c,m] of fields){
     if(!finite(x[c])||!finite(x[m])||Math.abs(x[c])>100000||Math.abs(x[m])>100000)
       return Object.freeze({status:'REJECTED',defects:['INVALID_'+name.toUpperCase()+'_AT_'+i],engineeringPassed:false,clinicalValidated:false,metrics:null});
     errors[name].push(name==='roll'?angleError(x[c],x[m]):normalError(x[c],x[m]));
   }
   if(typeof x.tracking!=='boolean')
     return Object.freeze({status:'REJECTED',defects:['TRACKING_FIELD_REQUIRED_AT_'+i],engineeringPassed:false,clinicalValidated:false,metrics:null});
   if(!x.tracking)trackingLosses++;
   if(i)longestGapMs=Math.max(longestGapMs,x.tMs-samples[i-1].tMs);
   timestamps.push(x.tMs);
 }
 const durationMs=timestamps[timestamps.length-1]-timestamps[0];
 const metrics={
   sampleCount:samples.length,durationMs,longestGapMs,trackingLosses,
   insertionP95Mm:percent(errors.insertion,.95),
   tipHP95Deg:percent(errors.tipH,.95),
   tipVP95Deg:percent(errors.tipV,.95),
   rollP95Deg:percent(errors.roll,.95)
 };
 if(durationMs<limit.minDurationMs)defects.push('INSUFFICIENT_DURATION');
 if(longestGapMs>limit.maxGapMs)defects.push('SAMPLE_GAPS_EXCEED_TARGET');
 if(trackingLosses)defects.push('TRACKING_LOSS_RECORDED');
 if(metrics.insertionP95Mm>limit.insertionP95Mm)defects.push('INSERTION_P95_OUTSIDE_ENGINEERING_TARGET');
 if(Math.max(metrics.tipHP95Deg,metrics.tipVP95Deg)>limit.tipP95Deg)
   defects.push('TIP_P95_OUTSIDE_ENGINEERING_TARGET');
 if(metrics.rollP95Deg>limit.rollP95Deg)defects.push('ROLL_P95_OUTSIDE_ENGINEERING_TARGET');
 return Object.freeze({
   status:defects.length?'BENCH_TARGET_NOT_MET':'BENCH_ENGINEERING_TARGET_MET',
   engineeringPassed:!defects.length,
   clinicalValidated:false,patientUseApproved:false,deviceCertified:false,
   defects:Object.freeze(defects),metrics:Object.freeze(metrics),
   thresholds:limit,provenance:Object.freeze({
     deviceModel:p.deviceModel,instrumentId:p.instrumentId,
     referenceSensor:p.referenceSensor,operator:p.operator,measuredAt:p.measuredAt
   }),
   interpretation:'QA of supplied external measurements only; no proof of sensor authenticity or clinical instrument safety'
 });
}
root.PulmoXRInstrumentBench=Object.freeze({assess,limit});
})(typeof window!=='undefined'?window:globalThis);
