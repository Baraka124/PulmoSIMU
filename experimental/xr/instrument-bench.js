/* XR16-FINAL — physical bronchoscope instrument bench model.
 * Hardware-independent, no network access, no patient data, no clinical control.
 * Device telemetry is NEVER attached to the original scope automatically.
 */
(function(root){
'use strict';
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const clip=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
const EXPECTED=['insertionMm','shaftRollDeg','tipPitchDeg','tipYawDeg'];
const LIMITS=Object.freeze({insertionMm:[0,2000],shaftRollDeg:[-180,180],
 tipPitchDeg:[-180,180],tipYawDeg:[-180,180]});
const FIELDS=Object.freeze(EXPECTED.slice());
function parseFrame(line){
 if(typeof line!=='string'||line.length>512)throw Error('INVALID_FRAME_LENGTH');
 let obj;try{obj=JSON.parse(line)}catch{throw Error('INVALID_JSON')}
 if(!obj||Array.isArray(obj)||typeof obj!=='object'||obj.protocol!=='PULMOSIMU-BENCH/1')
  throw Error('UNRECOGNIZED_PROTOCOL');
 if(!Number.isSafeInteger(obj.seq)||obj.seq<0)throw Error('INVALID_SEQUENCE');
 const values={};
 for(const key of EXPECTED){
  if(!finite(obj[key]))throw Error('MISSING_OR_NONFINITE_'+key);
  const [lo,hi]=LIMITS[key];
  if(obj[key]<lo||obj[key]>hi)throw Error('OUT_OF_BOUNDS_'+key);
  values[key]=obj[key];
 }
 return Object.freeze({protocol:obj.protocol,seq:obj.seq,...values});
}
function makeBench({maxSampleAgeMs=250}={}){
 if(!finite(maxSampleAgeMs)||maxSampleAgeMs<50||maxSampleAgeMs>2000)throw Error('INVALID_AGE_GATE');
 let last=null,lastAt=-Infinity,accepted=0,rejected=0;
 let calibration=null,lastSequence=-1;
 function ingest(frame,nowMs){
  if(!finite(nowMs)||nowMs<0)throw Error('INVALID_TIME');
  let sample;try{sample=typeof frame==='string'?parseFrame(frame):parseFrame(JSON.stringify(frame))}
  catch(e){rejected++;return Object.freeze({accepted:false,reason:e.message})}
  if(sample.seq<=lastSequence){rejected++;return Object.freeze({accepted:false,reason:'STALE_OR_REPLAYED_SEQUENCE'})}
  last=sample;lastAt=nowMs;lastSequence=sample.seq;accepted++;
  return Object.freeze({accepted:true,seq:sample.seq});
 }
 function calibrate(reference,nowMs){
  if(!last||!finite(nowMs)||nowMs-lastAt>maxSampleAgeMs)return Object.freeze({ok:false,reason:'NO_FRESH_SAMPLE'});
  if(!reference||typeof reference!=='object')return Object.freeze({ok:false,reason:'REFERENCE_REQUIRED'});
  const zeros={};
  for(const key of EXPECTED){
   if(!finite(reference[key]))return Object.freeze({ok:false,reason:'REFERENCE_MISSING_'+key});
   zeros[key]=last[key]-reference[key];
  }
  calibration=Object.freeze({offsets:Object.freeze(zeros),seq:last.seq,atMs:nowMs,
    basis:'OPERATOR_ENTERED_REFERENCE_NOT_METROLOGICALLY_VALIDATED'});
  return Object.freeze({ok:true,calibration});
 }
 function snapshot(nowMs){
  const freshness=finite(nowMs)&&nowMs>=lastAt&&nowMs-lastAt<=maxSampleAgeMs;
  const corrected=last&&calibration?Object.freeze(Object.fromEntries(
    EXPECTED.map(k=>[k,last[k]-calibration.offsets[k]]))):null;
  return Object.freeze({
   status:!last?'NO_INPUT':!freshness?'STALE_INPUT':!calibration?'UNCALIBRATED':'BENCH_ONLY',
   sample:last,fresh:Boolean(freshness),calibration,corrected,
   accepted,rejected,maxSampleAgeMs,
   clinicalUse:false,deviceToScopeCoupled:false,
   validation:'PHYSICAL_BENCH_AND_CLINICIAN_EVIDENCE_REQUIRED'
  });
 }
 return Object.freeze({ingest,calibrate,snapshot,reset(){
  last=null;lastAt=-Infinity;lastSequence=-1;calibration=null;accepted=0;rejected=0;
 }});
}
root.PulmoXRInstrumentTelemetry=Object.freeze({create:makeBench,parseFrame,FIELDS,LIMITS});
})(typeof window!=='undefined'?window:globalThis);
